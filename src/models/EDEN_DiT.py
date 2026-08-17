from torch import nn
import torch
from functools import partial
from src.modules import DiTBlock
from src.utils import TimestepEmbedder, get_pos_embedding, preprocess_cond


def preprocess_flow(flow_fwd, flow_bwd, eps=1e-8):
    """
    Normalize forward and backward flows jointly
    
    Args:
        flow_fwd: (B, 2, H, W)
        flow_bwd: (B, 2, H, W)
    Returns:
        flow_fwd_norm, flow_bwd_norm, stats
    """
    # Concatenate flows for joint statistics
    flows = torch.cat([flow_fwd, flow_bwd], dim=0)  # Changed from nn.cat
    
    # Compute statistics
    flows_flat = flows.flatten(1)
    flow_mean = torch.mean(flows_flat, dim=-1)  # Changed from nn.mean
    flow_std = torch.std(flows_flat, dim=-1) + eps  # Changed from nn.std


    # flows = nn.cat([flow_fwd, flow_bwd], dim=0)  # (2B, 2, H, W)
    
    # # Compute statistics
    # flows_flat = flows.flatten(1)  # (2B, 2*H*W)
    # flow_mean = nn.mean(flows_flat, dim=-1)  # (2B,)
    # flow_std = nn.std(flows_flat, dim=-1) + eps  # (2B,)
    
    # Reshape for broadcasting
    while len(flow_mean.shape) < len(flows.shape):
        flow_mean = flow_mean.unsqueeze(-1)
        flow_std = flow_std.unsqueeze(-1)
    
    # Normalize
    flows_norm = (flows - flow_mean) / flow_std
    
    # Split back
    flow_fwd_norm, flow_bwd_norm = flows_norm.chunk(2, dim=0)
    
    # Average statistics (like in preprocess_cond)
    flow_mean_fwd, flow_mean_bwd = flow_mean.chunk(2, dim=0)
    flow_std_fwd, flow_std_bwd = flow_std.chunk(2, dim=0)
    stats = ((flow_mean_fwd + flow_mean_bwd) / 2, (flow_std_fwd + flow_std_bwd) / 2)
    
    return flow_fwd_norm, flow_bwd_norm, stats


class DiT(nn.Module):
    def __init__(
        self,
        latent_dim=16,
        dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=12,
        qkv_bias=False,
        attn_drop_rate=0.,
        proj_drop_rate=0.,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_xformers=True
    ):
        super().__init__()
        self.dim = dim

        # --- Latent projection ---
        self.proj_in = nn.Linear(latent_dim, dim)

        # --- Frame conditioning ---
        self.proj_cond = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.norm_cond = norm_layer(dim)

        # --- Flow conditioning ---
        self.proj_flow = nn.Conv2d(2, dim, kernel_size=16, stride=16)
        self.norm_flow = norm_layer(dim)

        # --- Flow → global conditioning path ---
        self.flow_cond_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.flow_cond_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim, num_heads, mlp_ratio,
                qkv_bias, attn_drop_rate, proj_drop_rate,
                act_layer, norm_layer, use_xformers
            )
            for _ in range(depth)
        ])

        # --- Output ---
        self.norm_out = norm_layer(dim)
        self.proj_out = nn.Linear(dim, 2 * latent_dim)

        # --- AdaLN ---
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

        # --- Timestep ---
        self.denoise_timestep_embedder = TimestepEmbedder(dim)

        self.ph, self.pw = None, None
        self.init_weights()

    def init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

        # Conv init
        for conv in [self.proj_cond, self.proj_flow, self.flow_cond_conv]:
            w = conv.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            nn.init.constant_(conv.bias, 0)

        # Timestep embedder
        nn.init.normal_(self.denoise_timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.denoise_timestep_embedder.mlp[2].weight, std=0.02)

        # Zero output heads
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    # --------------------------------------------------

    def patch_cond(self, x):
        x, _ = preprocess_cond(x)
        x = self.proj_cond(x).flatten(2).transpose(1, 2)
        pos = get_pos_embedding(self.ph, self.pw, 1, self.dim).to(x.device)
        return self.norm_cond(x + pos)

    def patch_flow(self, flow_fwd, flow_bwd):
        fwd = self.proj_flow(flow_fwd).flatten(2).transpose(1, 2)
        bwd = self.proj_flow(flow_bwd).flatten(2).transpose(1, 2)

        pos = get_pos_embedding(self.ph, self.pw, 1, self.dim).to(fwd.device)
        fwd = self.norm_flow(fwd + pos)
        bwd = self.norm_flow(bwd + pos)
        return fwd, bwd

    def flow_global_embedding(self, flow_fwd_tokens, flow_bwd_tokens):
        """
        flow_*_tokens: (B, ph*pw, D)
        """

        B, N, D = flow_fwd_tokens.shape

        # Forward flow
        x_fwd = flow_fwd_tokens.transpose(1, 2).reshape(B, D, self.ph, self.pw)
        x_fwd = self.flow_cond_conv(x_fwd)
        x_fwd = x_fwd.mean(dim=[2, 3])  # GAP

        # Backward flow
        x_bwd = flow_bwd_tokens.transpose(1, 2).reshape(B, D, self.ph, self.pw)
        x_bwd = self.flow_cond_conv(x_bwd)
        x_bwd = x_bwd.mean(dim=[2, 3])  # GAP

        # Combine
        x = 0.5 * (x_fwd + x_bwd)

        return self.flow_cond_mlp(x)


    # --------------------------------------------------

    def forward(
        self,
        query_latents,
        denoise_timestep,
        cond_frames,
        flow_fwd,
        flow_bwd
    ):
        self.ph = cond_frames.shape[-2] // 16
        self.pw = cond_frames.shape[-1] // 16

        # --- Frame tokens ---
        tokens_0, tokens_1 = self.patch_cond(cond_frames).chunk(2, dim=0)

        # --- Flow tokens ---
        flow_fwd_tokens, flow_bwd_tokens = self.patch_flow(flow_fwd, flow_bwd)

        # --- Flow → global conditioning ---
        flow_global = self.flow_global_embedding(
            flow_fwd_tokens,
            flow_bwd_tokens
        )


        # --- AdaLN conditioning ---
        t_embed = self.denoise_timestep_embedder(denoise_timestep)
        condition = t_embed + flow_global
        modulations = self.adaLN_modulation(condition)

        # --- Query tokens ---
        pos = get_pos_embedding(self.ph, self.pw, 2, self.dim).to(query_latents.device)
        query = self.proj_in(query_latents) + pos

        for blk in self.blocks:
            query = blk(
                query,
                tokens_0,
                tokens_1,
                flow_fwd_tokens,
                flow_bwd_tokens,
                self.ph,
                self.pw,
                modulations
            )

        out = self.proj_out(self.norm_out(query))
        out, _ = out.chunk(2, dim=-1)
        return out
