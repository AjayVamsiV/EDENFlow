
from pathlib import Path
import argparse
import re
import os

import torch
import torchvision
import yaml

from src.models import load_model
from src.utils import InputPadder
from src.transport import create_transport, Sampler
from gmflow.gmflow_utils import (
    load_gmflow_model,
    compute_gmflow_warp_batched,
)


def natural_sort_key(path):
    """Sort frame2.png before frame10.png."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def interpolate(frame0, frame1):
    h, w = frame0.shape[2:]
    padder = InputPadder([h, w])

    with torch.no_grad():
        fwd_flow, bwd_flow = compute_gmflow_warp_batched(
            flow_model, frame0, frame1
        )

        fwd_flow = padder.pad(fwd_flow.detach())
        bwd_flow = padder.pad(bwd_flow.detach())

        cosine_sim = torch.cosine_similarity(frame0, frame1, dim=1)
        difference = (
            (cosine_sim.mean(dim=[1, 2]) - args.cos_sim_mean)
            / args.cos_sim_std
        ).unsqueeze(1).to(device)

        cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))
        new_h, new_w = cond_frames.shape[2:]

        noise = torch.randn(
            [
                1,
                (new_h // 32) * (new_w // 32),
                args.model_args["latent_dim"],
            ],
            device=device,
        )

        denoise_kwargs = {
            "cond_frames": cond_frames,
            "flow_fwd": fwd_flow,
            "flow_bwd": bwd_flow,
            "difference": difference,
        }

        samples = sample_fn(
            noise, eden.denoise, **denoise_kwargs
        )[-1]

        denoise_latents = samples / args.vae_scaler + args.vae_shift
        generated_frame = eden.decode(denoise_latents)
        generated_frame = padder.unpad(generated_frame.clamp(0.0, 1.0))

    return generated_frame


def load_frame(path):
    """Load an image as a normalized [1, C, H, W] tensor."""
    return (
        torchvision.io.read_image(str(path))
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )


def save_frame(frame, path):
    """Save a [1, C, H, W] tensor as an image."""
    torchvision.utils.save_image(frame, str(path))


def interpolate_video(video_path, output_dir, save_originals=True):
    """
    Interpolate a folder of extracted video frames.

    The output contains original frames and generated midpoint frames,
    unless save_originals=False.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_dir():
        raise NotADirectoryError(
            f"Video frames directory does not exist: {video_path}"
        )

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    frames = sorted(
        [
            path for path in video_path.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ],
        key=natural_sort_key,
    )

    if len(frames) < 2:
        raise ValueError(
            f"Expected at least two image frames in {video_path}; "
            f"found {len(frames)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input frames: {len(frames)}")
    print(f"Output directory: {output_dir}")

    # Save the first original frame.
    output_index = 0
    if save_originals:
        save_frame(
            load_frame(frames[0]),
            output_dir / f"frame_{output_index:06d}.png",
        )
        output_index += 1

    for i in range(len(frames) - 1):
        print(
            f"[{i + 1}/{len(frames) - 1}] "
            f"Interpolating {frames[i].name} -> {frames[i + 1].name}"
        )

        frame0 = load_frame(frames[i])
        frame1 = load_frame(frames[i + 1])

        # Generate the midpoint between consecutive input frames.
        interpolated_frame = interpolate(frame0, frame1)

        save_frame(
            interpolated_frame,
            output_dir / f"frame_{output_index:06d}.png",
        )
        output_index += 1

        # Append the next original frame to preserve temporal order.
        if save_originals:
            save_frame(
                frame1,
                output_dir / f"frame_{output_index:06d}.png",
            )
            output_index += 1

    print(f"Done! Saved {output_index} frames to {output_dir}")


def main():
    global args, device, eden, flow_model, sample_fn

    parser = argparse.ArgumentParser(
        description="EDENFlow image-pair and video-frame interpolation"
    )
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--frame_0_path", type=str, default=None)
    parser.add_argument("--frame_1_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument(
        "--interpolated_results_dir",
        type=str,
        default="interpolation_outputs",
    )
    parser.add_argument(
        "--interpolated-only",
        action="store_true",
        help="Save only generated frames, without original frames.",
    )

    # Load config defaults, while allowing CLI arguments to override them.
    initial_args, _ = parser.parse_known_args()
    with open(initial_args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.video_path is not None:
        if args.frame_0_path or args.frame_1_path:
            parser.error(
                "Use either --video_path or the two image paths, not both."
            )
    elif not (args.frame_0_path and args.frame_1_path):
        parser.error(
            "Provide --video_path or both --frame_0_path and --frame_1_path."
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load EDENFlow.
    model_name = args.model_name
    eden = load_model(model_name, **args.model_args)

    ckpt = torch.load(args.pretrained_eden_path, map_location="cpu", weights_only=False)
    eden.load_state_dict(ckpt["eden"])
    eden.to(device)
    eden.eval()
    del ckpt

    # Load GMFlow.
    print(f"Loading GMFlow model from {args.flow_path} ...")
    flow_model = load_gmflow_model(
        args.flow_path,
        with_refine=True,
    )
    flow_model.to(device)
    flow_model.requires_grad_(False)
    flow_model.eval()

    # Configure the sampler.
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method="euler",
        num_steps=2,
        atol=1e-6,
        rtol=1e-3,
    )

    output_dir = args.interpolated_results_dir

    if args.video_path is not None:
        interpolate_video(
            args.video_path,
            output_dir,
            save_originals=not args.interpolated_only,
        )
    else:
        print(
            f"Interpolating {args.frame_0_path} and "
            f"{args.frame_1_path} ..."
        )

        frame0 = load_frame(args.frame_0_path)
        frame1 = load_frame(args.frame_1_path)
        result = interpolate(frame0, frame1)

        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / "interpolated.png"
        save_frame(result, output_path)
        print(f"Saved interpolated image to {output_path}")


if __name__ == "__main__":
    main()
