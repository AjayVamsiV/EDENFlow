# EDEN/RAFT/raft_utils.py
import os
import sys
import argparse
import torch
import numpy as np

# ensure RAFT core is importable
CORE_DIR = os.path.join(os.path.dirname(__file__), "core")
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

from raft import RAFT
# RAFT's own InputPadder lives under core/utils or core/utils/utils depending on RAFT version.
# The original RAFT demo uses: from utils.utils import InputPadder
from utils.utils import InputPadder as RAFTPadder

DEVICE = "cuda"


def load_raft_model(model_path, small=False, mixed_precision=False, alternate_corr=False):
    args = argparse.Namespace()
    args.small = small
    args.mixed_precision = mixed_precision
    args.alternate_corr = alternate_corr
    args.dropout = 0.0
    args.dropout2 = 0.0

    model = RAFT(args)

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=DEVICE)

    # --- FIX HERE ---
    # If keys start with "module.", strip them.
    if any(k.startswith("module.") for k in ckpt.keys()):
        print("Stripping 'module.' prefix from RAFT checkpoint...")
        state_dict = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    else:
        state_dict = ckpt
    # --- END FIX ---

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE).eval()
    return model



def _ensure_tensor_on_device(x, device, dtype=torch.float32):
    """
    Convert x to a contiguous torch tensor on device.
    Handles: list of tensors, numpy arrays, torch tensors, etc.
    """
    # If it's already a torch tensor but on CPU or wrong dtype -> convert
    if isinstance(x, torch.Tensor):
        x = x.contiguous().to(device=device, dtype=dtype)
        return x

    # If it's a numpy array
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype).contiguous()

    # If it's a list/tuple of tensors or ndarrays -> stack
    if isinstance(x, (list, tuple)):
        # convert inner items to tensors then stack
        items = []
        for it in x:
            if isinstance(it, torch.Tensor):
                items.append(it)
            elif isinstance(it, np.ndarray):
                items.append(torch.from_numpy(it))
            else:
                # last resort: torch.tensor
                items.append(torch.tensor(it))
        stacked = torch.stack(items, dim=0).to(device=device, dtype=dtype).contiguous()
        return stacked

    # fallback
    return torch.tensor(x, dtype=dtype, device=device).contiguous()


def warp(image, flow):
    """
    Backward warp: sample from `image` using `flow` (pixel units).
    image: (N,3,H,W), flow: (N,2,H,W)
    returns warped image (N,3,H,W)
    """
    B, C, H, W = image.size()
    device = flow.device

    xx = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)

    # convert pixel flow -> normalized [-1,1]
    u = flow[:, 0:1, :, :] / ((W - 1) / 2.0)
    v = flow[:, 1:2, :, :] / ((H - 1) / 2.0)

    grid = torch.cat([xx + u, yy + v], dim=1)  # Bx2xHxW
    grid = grid.permute(0, 2, 3, 1)  # BxHxWx2

    warped = torch.nn.functional.grid_sample(image, grid, align_corners=True)
    return warped


@torch.no_grad()
def compute_raft_warp(raft_model, frame0, frame1, iters=20, test_mode=True):
    device = next(raft_model.parameters()).device
    frame0 = _ensure_tensor_on_device(frame0, device=device, dtype=torch.float32)
    frame1 = _ensure_tensor_on_device(frame1, device=device, dtype=torch.float32)

    if frame0.ndim == 3:
        frame0 = frame0.unsqueeze(0)
    if frame1.ndim == 3:
        frame1 = frame1.unsqueeze(0)

    if frame0.shape[1] == 1:
        frame0 = frame0.repeat(1, 3, 1, 1)
    if frame1.shape[1] == 1:
        frame1 = frame1.repeat(1, 3, 1, 1)

    # pad BOTH frames and keep using padded tensors for RAFT and warp
    pad = RAFTPadder(frame0.shape)
    f0_padded, f1_padded = pad.pad(frame0, frame1)

    # make sure RAFT gets padded tensors
    _, fwd_padded = raft_model(f0_padded, f1_padded, iters=iters, test_mode=test_mode)
    _, bwd_padded = raft_model(f1_padded, f0_padded, iters=iters, test_mode=test_mode)

    # Warp while still padded (so image and flow align)
    warp_fwd_padded = warp(f1_padded, fwd_padded)  # frame1 -> frame0 (padded)
    warp_bwd_padded = warp(f0_padded, bwd_padded)  # frame0 -> frame1 (padded)

    # Now unpad everything back to original spatial dims
    fwd = pad.unpad(fwd_padded).contiguous()
    bwd = pad.unpad(bwd_padded).contiguous()
    warp_fwd = pad.unpad(warp_fwd_padded).contiguous()
    warp_bwd = pad.unpad(warp_bwd_padded).contiguous()

    return fwd, bwd, warp_fwd, warp_bwd
