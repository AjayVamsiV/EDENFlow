from src.utils.embedding import TimestepEmbedder, get_pos_embedding
from src.utils.klperceptual import KLLPIPSWithDiscriminator
from src.utils.distributions import DiagonalGaussianDistribution
from src.utils.cal_metrics import CalMetrics
import torch
from RAFT.raft_utils import compute_raft_warp

class InputPadder:
    def __init__(self, img_size, divisor=32):
        self.ht, self.wd = img_size
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, x):
        return torch.nn.functional.pad(x, self._pad, mode="replicate")

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def preprocess_cond(x, eps=1e-8):
    x_flat = x.flatten(1)
    x_mean, x_std = torch.mean(x_flat, dim=-1), torch.std(x_flat, dim=-1) + eps
    while len(x_mean.shape) < len(x.shape):
        x_mean, x_std = x_mean.unsqueeze(-1), x_std.unsqueeze(-1)
    x_norm = (x - x_mean) / x_std
    x_mean_0, x_mean_1 = x_mean.chunk(2, dim=0)
    x_std_0, x_std_1 = x_std.chunk(2, dim=0)
    stats = ((x_mean_0 + x_mean_1) / 2, (x_std_0 + x_std_1) / 2)
    return x_norm, stats


def preprocess_frames(frames):
    frames = frames / 255.
    frame_0, frame_1, gt = frames[:, 0, ...], frames[:, 1, ...], frames[:, 2, ...]
    frames = torch.cat((frame_0, frame_1, gt), dim=0)
    img_size = [frames.shape[2], frames.shape[3]]
    padder = InputPadder(img_size)
    return frames, padder, frame_0, frame_1, gt

def one_iter_for_vae(model, frames, is_train=True):
    frames, padder, _, _, gt = preprocess_frames(frames)
    if not is_train:
        with torch.no_grad():
            recon, posterior = model(padder.pad(frames))
    else:
        recon, posterior = model(padder.pad(frames))
    recon = padder.unpad(recon.clamp(0., 1.))
    return recon, gt, posterior


def one_iter_for_dit(model, vae, raft_model, frames, transport, sample_fn,
                     vae_mean, vae_scaler, cos_sim_mean, cos_sim_std,
                     is_train=True):

    vae_model = vae.module if hasattr(vae, "module") else vae
    model_fwd = model.module.forward if hasattr(model, "module") else model.forward
    raft_model = raft_model.module if hasattr(raft_model, "module") else raft_model

    frames = frames / 1.
    rgb_0 = frames[:, 0]  # (B, 3, H, W)
    rgb_1 = frames[:, 1]  # (B, 3, H, W)
    gt    = frames[:, 2]  # (B, 3, H, W)

    # Compute RAFT on ORIGINAL resolution (BEFORE padding)
    with torch.no_grad():
        fwd_flow, bwd_flow, warp_fwd, warp_bwd = compute_raft_warp(
            raft_model, rgb_0, rgb_1
        )

    # detach & free graph
    fwd_flow = fwd_flow.detach()
    bwd_flow = bwd_flow.detach()
    warp_fwd = warp_fwd.detach()
    warp_bwd = warp_bwd.detach()

    torch.cuda.empty_cache()
    
    # NOW preprocess/pad frames
    frames, padder, frame_0, frame_1, gt = preprocess_frames(frames)

    #print("frames shape", gt.shape)
    
    # Pad flows to match
    fwd_flow = padder.pad(fwd_flow)
    bwd_flow = padder.pad(bwd_flow)
    warp_fwd = padder.pad(warp_fwd)
    warp_bwd = padder.pad(warp_bwd)
    
    cond_frames = torch.cat((frame_0, frame_1), dim=0)

    # IMPORTANT: Flows are already computed on padded frames, so they should match
    fwd_flow = fwd_flow.to(frames.device)
    bwd_flow = bwd_flow.to(frames.device)

    denoise_args = {
        "cond_frames": padder.pad(cond_frames),  # Apply same padding to cond_frames
        # "difference": difference,
        "flow_fwd": fwd_flow,  # Already padded from compute_raft_warp
        "flow_bwd": bwd_flow   # Already padded from compute_raft_warp
    }

    with torch.no_grad():
        posterior, cond_tokens = vae_model.encode(padder.pad(frames))
        latent = (posterior.sample() - vae_mean).mul_(vae_scaler)

    if is_train:
        loss_dict = transport.training_losses(model, latent, **denoise_args)
        return loss_dict, latent, cond_tokens, denoise_args
    else:
        with torch.no_grad():
            noise = torch.randn_like(latent)
            samples = sample_fn(noise, model_fwd, **denoise_args)[-1]
            generated = vae_model.decode(
                samples / vae_scaler + vae_mean,
                cond_tokens
            )
            generated = padder.unpad(generated.clamp(0., 1.))
        return generated

