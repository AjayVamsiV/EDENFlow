from src.models import load_model
from src.datasets import load_dataset
from src.utils import CalMetrics, InputPadder, one_iter_for_dit
from src.transport import create_transport, Sampler
from RAFT.raft_utils import load_raft_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import transformers
import diffusers
import torch
import argparse
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import logging
import os
from glob import glob
import yaml
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_dit.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        update_args = yaml.unsafe_load(f)
    parser.set_defaults(**update_args)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Evaluation currently requires at least one GPU!"
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    model_name = args.model_name  # "EDEN_DiT"
    output_dir = f"{args.output_dir}/eval-{model_name}"
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
        experiment_index = len(glob(f"{output_dir}/*"))
        experiment_dir = f"{output_dir}/{experiment_index:03d}"
        visualization_dir = f"{experiment_dir}/visualization_results"
        os.makedirs(visualization_dir, exist_ok=True)
        evaluation_dir = f"{experiment_dir}/evaluation_results"
        os.makedirs(evaluation_dir, exist_ok=True)
        logging.basicConfig(
            format="[\033[34m%(asctime)s\033[0m] - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")]
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        logger.info(f"Experiment directory created at {experiment_dir}")

    if args.global_seed is not None:
        set_seed(args.global_seed)

    # load dataset
    local_batch_size = args.dataloader["batch_size"]
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name, **args.dataset_args[dataset_name])
    dataloader = DataLoader(dataset, **args.dataloader)
    dataset_len = len(dataset)
    steps_one_epoch = dataset_len // (local_batch_size * accelerator.num_processes)
    logger.info(f"Dataset {dataset_name} contains {dataset_len:,} triplets")

    # load DiT model
    model = load_model(model_name, **args.model_args)
    logger.info(f"{model_name} Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dit_ckpt = torch.load(args.pretrained_dit_path, map_location="cpu")
    #print the checkpoint keys to verify
    logger.info(f"DiT checkpoint keys: {list(dit_ckpt.keys())}")
    #print model architecture to verify compatibility
    logger.info(f"DiT model architecture: {model}")
    model.load_state_dict(dit_ckpt["eden_dit"])
    logger.info(f"Loaded DiT checkpoint from {args.pretrained_dit_path}")
    del dit_ckpt

    # load VAE model
    vae = load_model("EDEN_VAE", **args.vae_args)
    vae_ckpt = torch.load(args.pretrained_vae_path, map_location="cpu")["eden_vae"]
    vae.load_state_dict(vae_ckpt)
    logger.info(f"Loaded VAE checkpoint from {args.pretrained_vae_path}")
    del vae_ckpt

    # load RAFT model (frozen, inference only)
    raft_model = load_raft_model(args.raft_model_path)

    # transport / sampler
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)
    cal_metrics = CalMetrics()

    model, vae, raft_model, dataloader = accelerator.prepare(model, vae, raft_model, dataloader)

    model.eval()
    vae.requires_grad_(False)
    vae.eval()
    raft_model.requires_grad_(False)
    raft_model.eval()

    vae_shift  = args.vae_shift
    vae_scaler = args.vae_scaler

    steps = 0
    results = {"PSNR": 0., "SSIM": 0., "LPIPS": 0., "FloLPIPS": 0., "L1": 0.}
    logger.info(f"Evaluating for {steps_one_epoch} steps...")

    for _, batch in enumerate(dataloader):
        generated_frames = one_iter_for_dit(
            model, vae, raft_model, batch, transport, sample_fn,
            vae_shift, vae_scaler,
            args.cos_sim_mean, args.cos_sim_std,
            is_train=False
        )

        # ground-truth for metrics — same /255 normalization as preprocess_frames
        frames  = batch / 255.
        frame_0 = frames[:, 0, ...]
        frame_1 = frames[:, 1, ...]
        gt      = frames[:, 2, ...]

        psnr     = cal_metrics.cal_psnr(generated_frames, gt)
        ssim     = cal_metrics.cal_ssim(generated_frames, gt)
        lpips    = cal_metrics.cal_lpips(generated_frames, gt)
        flolpips = cal_metrics.cal_flolpips(generated_frames, gt, frame_0, frame_1)
        l1       = torch.abs(generated_frames - gt)

        results["PSNR"]     += accelerator.gather(psnr).sum().item()
        results["SSIM"]     += accelerator.gather(ssim).sum().item()
        results["LPIPS"]    += accelerator.gather(lpips).sum().item()
        results["FloLPIPS"] += accelerator.gather(flolpips).sum().item()
        results["L1"]       += accelerator.gather(l1).sum().item()
        steps += 1

        logger.info(
            f"(step={steps:04d}) [PSNR: {psnr.mean():.4f}, SSIM: {ssim.mean():.4f}, "
            f"LPIPS: {lpips.mean():.4f}, FloLPIPS: {flolpips.mean():.4f}, L1: {l1.mean():.4f}]"
        )

        if args.save_generated_frames:
            if accelerator.is_local_main_process:
                blended_input = frame_0 * 0.5 + frame_1 * 0.5
                gt_generated_frames = torch.cat((blended_input, gt, generated_frames), dim=0)
                save_image(gt_generated_frames, f"{visualization_dir}/steps{steps:07d}.png")
                logger.info(f"Saved visualization results to {visualization_dir}")

    print(
        "PSNR total: ",     results["PSNR"],
        "SSIM total: ",     results["SSIM"],
        "LPIPS total: ",    results["LPIPS"],
        "FloLPIPS total: ", results["FloLPIPS"],
        "L1 total: ",       results["L1"]
    )

    if accelerator.num_processes > 1:
        total_samples = steps * local_batch_size * accelerator.num_processes
    else:
        total_samples = dataset.__len__()

    print("total samples evaluated: ", total_samples)
    for key in results.keys():
        results[key] /= total_samples
    format_results = (
        f"PSNR: {results['PSNR']:.4f},  SSIM: {results['SSIM']:.4f}, "
        f"LPIPS: {results['LPIPS']:.4f}, FloLPIPS: {results['FloLPIPS']:.4f}, "
        f"L1: {results['L1']:.4f}"
    )
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        with open(f"{evaluation_dir}/evaluation_results.txt", mode="w+", encoding="utf-8") as f:
            f.write(format_results)
    logger.info(
        f"DiT ({args.pretrained_dit_path}) + VAE ({args.pretrained_vae_path}) "
        f"evaluation results on {dataset_name}: {format_results}"
    )
    accelerator.end_training()
    logger.info("Done!")


if __name__ == "__main__":
    main()