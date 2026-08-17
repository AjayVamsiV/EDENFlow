# EDEN/GMFlow/gmflow_utils.py

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

# ---------------------------------------------------
# Ensure GMFlow repo importable
# ---------------------------------------------------
ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from gmflow.gmflow.gmflow import GMFlow

DEVICE = "cuda"


# ---------------------------------------------------
# Padder (same role as RAFTPadder)
# ---------------------------------------------------
class GMFlowPadder:
    """
    Pads images so H,W divisible by divisor (default 32).
    GMFlow refine models usually prefer /32 compatibility.
    """
    def __init__(self, dims, divisor=32):
        self.ht, self.wd = dims[-2:]

        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor

        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2
        ]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [
            self._pad[2],
            ht - self._pad[3],
            self._pad[0],
            wd - self._pad[1]
        ]
        return x[..., c[0]:c[1], c[2]:c[3]]


# ---------------------------------------------------
# Tensor helper
# ---------------------------------------------------
def _ensure_tensor_on_device(x, device, dtype=torch.float32):
    """
    Convert x to contiguous tensor on device.
    Supports tensor / numpy / list / tuple.
    """
    if isinstance(x, torch.Tensor):
        return x.contiguous().to(device=device, dtype=dtype)

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype).contiguous()

    if isinstance(x, (list, tuple)):
        items = []
        for it in x:
            if isinstance(it, torch.Tensor):
                items.append(it)
            elif isinstance(it, np.ndarray):
                items.append(torch.from_numpy(it))
            else:
                items.append(torch.tensor(it))
        return torch.stack(items, dim=0).to(device=device, dtype=dtype).contiguous()

    return torch.tensor(x, dtype=dtype, device=device).contiguous()


# ---------------------------------------------------
# Warp helper (same as RAFT util)
# ---------------------------------------------------
def warp(image, flow):
    """
    Backward warp:
    image : (B,C,H,W)
    flow  : (B,2,H,W) in pixel units
    """
    B, C, H, W = image.size()
    device = flow.device

    xx = torch.linspace(-1, 1, W, device=device).view(1,1,1,W).expand(B,1,H,W)
    yy = torch.linspace(-1, 1, H, device=device).view(1,1,H,1).expand(B,1,H,W)

    u = flow[:, 0:1] / ((W - 1) / 2.0)
    v = flow[:, 1:2] / ((H - 1) / 2.0)

    grid = torch.cat([xx + u, yy + v], dim=1)
    grid = grid.permute(0,2,3,1)

    warped = F.grid_sample(image, grid, align_corners=True)
    return warped


# ---------------------------------------------------
# Load GMFlow model
# ---------------------------------------------------
def load_gmflow_model(model_path, with_refine=True):
    """
    Load pretrained GMFlow checkpoint.

    with_refine=True:
        num_scales=2, upsample_factor=4

    with_refine=False:
        num_scales=1, upsample_factor=8
    """

    if with_refine:
        model = GMFlow(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            attention_type='swin',
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        )
    else:
        model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type='swin',
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        )

    ckpt = torch.load(model_path, map_location=DEVICE)
    weights = ckpt["model"] if "model" in ckpt else ckpt

    if any(k.startswith("module.") for k in weights.keys()):
        print("Stripping 'module.' prefix from GMFlow checkpoint...")
        weights = {k.replace("module.", "", 1): v for k, v in weights.items()}

    model.load_state_dict(weights, strict=True)
    model.to(DEVICE).eval()

    return model


# ---------------------------------------------------
# Compute flow + warped images
# ---------------------------------------------------
@torch.no_grad()
def compute_gmflow_warp(
    gmflow_model,
    frame0,
    frame1,
    attn_splits_list=[2, 8],
    corr_radius_list=[-1, 4],
    prop_radius_list=[-1, 1],
):
    """
    Same API spirit as compute_raft_warp()

    Inputs:
        frame0, frame1 : (B,C,H,W) or (C,H,W)

    Returns:
        fwd       : frame0 -> frame1 flow
        bwd       : frame1 -> frame0 flow
        warp_fwd  : warp frame1 toward frame0
        warp_bwd  : warp frame0 toward frame1
    """

    device = next(gmflow_model.parameters()).device

    frame0 = _ensure_tensor_on_device(frame0, device=device)
    frame1 = _ensure_tensor_on_device(frame1, device=device)

    if frame0.ndim == 3:
        frame0 = frame0.unsqueeze(0)

    if frame1.ndim == 3:
        frame1 = frame1.unsqueeze(0)

    if frame0.shape[1] == 1:
        frame0 = frame0.repeat(1,3,1,1)

    if frame1.shape[1] == 1:
        frame1 = frame1.repeat(1,3,1,1)

    # pad
    pad = GMFlowPadder(frame0.shape, divisor=32)
    f0_padded, f1_padded = pad.pad(frame0, frame1)

    # forward flow
    out_fwd = gmflow_model(
        f0_padded,
        f1_padded,
        attn_splits_list=attn_splits_list,
        corr_radius_list=corr_radius_list,
        prop_radius_list=prop_radius_list,
    )

    # backward flow
    out_bwd = gmflow_model(
        f1_padded,
        f0_padded,
        attn_splits_list=attn_splits_list,
        corr_radius_list=corr_radius_list,
        prop_radius_list=prop_radius_list,
    )

    fwd_padded = out_fwd["flow_preds"][-1]
    bwd_padded = out_bwd["flow_preds"][-1]

    # warped images
    warp_fwd_padded = warp(f1_padded, fwd_padded)
    warp_bwd_padded = warp(f0_padded, bwd_padded)

    # unpad
    fwd = pad.unpad(fwd_padded).contiguous()
    bwd = pad.unpad(bwd_padded).contiguous()

    warp_fwd = pad.unpad(warp_fwd_padded).contiguous()
    warp_bwd = pad.unpad(warp_bwd_padded).contiguous()

    return fwd, bwd, warp_fwd, warp_bwd

@torch.no_grad()
def compute_gmflow_warp_batched(model, frame0, frame1, sub_bs=16):
    fwd_list = []
    bwd_list = []

    B = frame0.shape[0]

    for i in range(0, B, sub_bs):
        f0 = frame0[i:i+sub_bs]
        f1 = frame1[i:i+sub_bs]

        fwd, bwd, _, _ = compute_gmflow_warp(model, f0, f1)

        fwd_list.append(fwd)
        bwd_list.append(bwd)

        del fwd, bwd
        torch.cuda.empty_cache()

    fwd_all = torch.cat(fwd_list, dim=0)
    bwd_all = torch.cat(bwd_list, dim=0)

    return fwd_all, bwd_all