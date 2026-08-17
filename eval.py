from src.models import load_model
from src.datasets import load_dataset
from src.utils import CalMetrics, InputPadder
from src.transport import create_transport, Sampler
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


from src.utils import InputPadder

from src.utils import preprocess_frames

from RAFT.raft_utils import (
    load_raft_model,
    compute_raft_warp
)

from gmflow.gmflow_utils import load_gmflow_model, compute_gmflow_warp_batched


warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        update_args = yaml.unsafe_load(f)
    parser.set_defaults(**update_args)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Training currently requires at least one GPU!"
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    model_name = args.model_name
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

    # load model
    model = load_model(model_name, **args.model_args)
    logger.info(f"{model_name} Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ckpt = torch.load(args.pretrained_eden_path, map_location="cpu")
    print(args.pretrained_eden_path)
    print("ckecpint keys: ",ckpt.keys())
    model.load_state_dict(ckpt["eden"])
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)
    cal_metrics = CalMetrics()

    # flow_model = load_raft_model(
    #     "./checkpoints/raft.pth"
    # )

    flow_model = load_gmflow_model(
        "./checkpoints/gmflow.pth",
        with_refine=True
    )

    flow_model.requires_grad_(False)

    flow_model.eval()

    model, flow_model, dataloader = (
        accelerator.prepare(
            model,
            flow_model,
            dataloader
        )
    )

    # begin training
    model.eval()
    #model = model if model.module is None else model.module
    model = model.module if hasattr(model, "module") else model
    steps = 0
    results = {"PSNR": 0., "SSIM": 0., "LPIPS": 0., "FloLPIPS": 0., "L1": 0.}
    logger.info(f"Evaluating for {steps_one_epoch} steps...")
    for _, batch in enumerate(dataloader):
        frames = batch / 1.

        rgb_0 = frames[:, 0]
        rgb_1 = frames[:, 1]
        gt = frames[:, 2]

        # -------------------------------------------------
        # FLOW
        # -------------------------------------------------

        # with torch.no_grad():

        #     (
        #         fwd_flow,
        #         bwd_flow,
        #         _,
        #         _
        #     ) = compute_raft_warp(
        #         flow_model.module
        #         if hasattr(flow_model, "module")
        #         else flow_model,
        #         rgb_0,
        #         rgb_1
        #     )
        with torch.no_grad():
            fwd_flow, bwd_flow = compute_gmflow_warp_batched(
                flow_model,
                rgb_0,
                rgb_1,
                sub_bs=1
            )

        fwd_flow = fwd_flow.detach()

        bwd_flow = bwd_flow.detach()

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        (
            frames,
            padder,
            frame_0,
            frame_1,
            gt
        ) = preprocess_frames(frames)

        # -------------------------------------------------
        # PAD FLOWS
        # -------------------------------------------------

        fwd_flow = padder.pad(fwd_flow)

        bwd_flow = padder.pad(bwd_flow)

        # -------------------------------------------------
        # CONDITIONING
        # -------------------------------------------------

        cond_frames = torch.cat(
            (frame_0, frame_1),
            dim=0
        )

        cond_frames = padder.pad(cond_frames)

        difference = (
            (
                torch.mean(
                    torch.cosine_similarity(
                        frame_0,
                        frame_1
                    ),
                    dim=[1, 2]
                )
                - args.cos_sim_mean
            )
            / args.cos_sim_std
        ).unsqueeze(1).to(
            accelerator.device
        )

        # -------------------------------------------------
        # SAMPLE
        # -------------------------------------------------

        with torch.no_grad():

            b, _, h, w = cond_frames.shape

            noise = torch.randn(
                [
                    b // 2,
                    (h // 32) * (w // 32),
                    args.model_args["latent_dim"]
                ]
            ).to(accelerator.device)

            denoise_kwargs = {

                "cond_frames": cond_frames,

                "difference": difference,

                "flow_fwd": fwd_flow,

                "flow_bwd": bwd_flow
            }

            model_unwrap = (
                model.module
                if hasattr(model, "module")
                else model
            )

            samples = sample_fn(
                noise,
                model_unwrap.denoise,
                **denoise_kwargs
            )[-1]

            denoise_latents = (
                samples
                / args.vae_scaler
                + args.vae_shift
            )

            generated_frames = (
                model_unwrap.decode(
                    denoise_latents
                )
            )

            generated_frames = (
                padder.unpad(
                    generated_frames.clamp(
                        0.,
                        1.
                    )
                )
            )
        psnr = cal_metrics.cal_psnr(generated_frames, gt)
        ssim = cal_metrics.cal_ssim(generated_frames, gt)
        lpips = cal_metrics.cal_lpips(generated_frames, gt)
        flolpips = cal_metrics.cal_flolpips(generated_frames, gt, frame_0, frame_1)
        #l1 = torch.mean(torch.abs(generated_frames - gt))
        l1 = torch.abs(generated_frames - gt)
        cur_batch_size = frame_0.shape[0]
        #print("cur batch size: ", cur_batch_size, accelerator.gather(psnr.repeat(cur_batch_size)).shape)
        #print(accelerator.gather(psnr.repeat(cur_batch_size)))
        #print("PSNR sum:", psnr.shape, accelerator.gather(psnr).shape)
        results["PSNR"] += accelerator.gather(psnr).sum().item()
        results["SSIM"] += accelerator.gather(ssim).sum().item()
        results["LPIPS"] += accelerator.gather(lpips).sum().item()
        results["FloLPIPS"] += accelerator.gather(flolpips).sum().item()
        results["L1"] += accelerator.gather(l1).sum().item()
        steps += 1
        logger.info(f"(step={steps:04d}) [PSNR: {psnr.mean():.4f}, SSIM: {ssim.mean():.4f}, "
                    f"LPIPS: {lpips.mean():.4f}, FloLPIPS: {flolpips.mean():.4f}, L1: {l1.mean():.4f}")
        if args.save_generated_frames:
            if accelerator.is_local_main_process:
                blended_input = frame_0 * 0.5 + frame_1 * 0.5
                gt_generated_frames = torch.cat((blended_input, gt, generated_frames), dim=0)
                save_image(gt_generated_frames, f"{visualization_dir}/steps{steps:07d}.png")
                logger.info(f"Saved visualization results to {visualization_dir}")

    if accelerator.num_processes > 1:
        total_samples = steps * local_batch_size * accelerator.num_processes
    else:
        total_samples = dataloader.dataset.__len__()
    print("PSNR total: ", results["PSNR"], "SSIM total: ", results["SSIM"], "LPIPS total: ", results["LPIPS"], "FloLPIPS total: ", results["FloLPIPS"], "L1 total: ", results["L1"])
    print(f"Total samples evaluated: {total_samples}. ", dataloader.dataset.__len__(), accelerator.num_processes)
    for key in results.keys():
        results[key] /= total_samples
    format_results = (f"PSNR: {results['PSNR']:.4f},  SSIM: {results['SSIM']:.4f}, LPIPS: {results['LPIPS']:.4f}, "
                      f"FloLPIPS: {results['FloLPIPS']:.4f}, L1: {results['L1']:.4f}")
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        with open(f"{evaluation_dir}/evaluation_results.txt", mode="w+", encoding="utf-8") as f:
            f.write(format_results)
    logger.info(f"Pretrained {model_name} (ckpt:{args.pretrained_eden_path}) evaluation results on {dataset_name}: {format_results}")
    accelerator.end_training()
    model.eval()
    logger.info("Done!")


if __name__ == "__main__":
    main()