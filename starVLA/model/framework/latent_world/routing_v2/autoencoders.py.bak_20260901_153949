from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTokenAutoencoder(nn.Module):
    """Task memory over Base-VLM latent-action-query hidden tokens.

    Input shape: [B, Q, D_vlm].  The same bottleneck is applied token-wise, so
    the original query structure is preserved instead of mean-pooling it away.
    Reconstruction is performed in layer-normalized feature space to make the
    error less sensitive to feature scale.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        if self.input_dim <= 0 or self.bottleneck_dim <= 0:
            raise ValueError("SemanticTokenAutoencoder dimensions must be positive.")
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.bottleneck_dim),
            nn.GELU(),
        )
        self.decoder = nn.Linear(self.bottleneck_dim, self.input_dim)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), (self.input_dim,))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = self.normalize(x)
        recon = self.decoder(self.encoder(target))
        per_token = (recon - target).pow(2).mean(dim=-1)
        return {
            "reconstruction": recon,
            "target": target,
            "loss": per_token.mean(),
            "per_sample_error": per_token.mean(dim=-1),
        }


class SpatialDynamicsAutoencoder(nn.Module):
    """Task memory for latent world-action transitions.

    The current state h_t is used as conditioning context, while the module
    reconstructs the normalized spatial transition Delta h and latent action z.
    This avoids letting the shared current-state reconstruction dominate the
    task-discriminative future dynamics signal.
    """

    def __init__(
        self,
        *,
        vision_dim: int,
        latent_dim: int,
        num_tokens: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 6,
        ffn_dim: int = 768,
        z_loss_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.vision_dim = int(vision_dim)
        self.latent_dim = int(latent_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.z_loss_weight = float(z_loss_weight)
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(
                f"Dynamics AE hidden_dim={self.hidden_dim} must be divisible by num_heads={num_heads}."
            )

        input_dim = 2 * self.vision_dim + self.latent_dim
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.delta_decoder = nn.Linear(self.hidden_dim, self.vision_dim)
        self.z_decoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def _norm_vision(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), (self.vision_dim,))

    def _norm_z(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3 and z.shape[1] == 1:
            z = z[:, 0, :]
        if z.dim() != 2 or int(z.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"Expected z [B,{self.latent_dim}] or [B,1,{self.latent_dim}], got {tuple(z.shape)}"
            )
        return F.layer_norm(z.float(), (self.latent_dim,))

    def forward(
        self,
        *,
        h_t: torch.Tensor,
        h_future: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if h_t.dim() != 3 or h_future.dim() != 3:
            raise ValueError(
                f"Dynamics AE expects [B,N,D] features, got h_t={tuple(h_t.shape)}, "
                f"h_future={tuple(h_future.shape)}"
            )
        if h_t.shape != h_future.shape:
            raise ValueError(
                f"Dynamics AE current/future shape mismatch: {tuple(h_t.shape)} vs {tuple(h_future.shape)}"
            )
        if int(h_t.shape[1]) != self.num_tokens or int(h_t.shape[2]) != self.vision_dim:
            raise ValueError(
                f"Dynamics AE feature geometry mismatch: got {tuple(h_t.shape[1:])}, "
                f"expected ({self.num_tokens},{self.vision_dim})"
            )

        h_cur_n = self._norm_vision(h_t)
        delta_n = self._norm_vision(h_future - h_t)
        z_n = self._norm_z(z)
        z_tokens = z_n.unsqueeze(1).expand(-1, self.num_tokens, -1)
        x = torch.cat([h_cur_n, delta_n, z_tokens], dim=-1)
        hidden = self.input_proj(x) + self.pos_embed.to(dtype=x.dtype, device=x.device)
        hidden = self.encoder(hidden)

        delta_recon = self.delta_decoder(hidden)
        z_recon = self.z_decoder(hidden.mean(dim=1))
        delta_loss = F.mse_loss(delta_recon, delta_n)
        z_loss = F.mse_loss(z_recon, z_n)
        loss = delta_loss + self.z_loss_weight * z_loss

        per_sample_delta = (delta_recon - delta_n).pow(2).mean(dim=(1, 2))
        per_sample_z = (z_recon - z_n).pow(2).mean(dim=1)
        return {
            "loss": loss,
            "loss_delta": delta_loss,
            "loss_z": z_loss,
            "per_sample_error": per_sample_delta + self.z_loss_weight * per_sample_z,
            "delta_reconstruction": delta_recon,
            "z_reconstruction": z_recon,
        }


class RoutingV2Memory(nn.Module):
    def __init__(
        self,
        *,
        semantic_dim: int,
        semantic_bottleneck: int,
        vision_dim: int,
        latent_dim: int,
        num_tokens: int,
        dynamics_hidden: int,
        dynamics_layers: int,
        dynamics_heads: int,
        dynamics_ffn: int,
        dynamics_z_weight: float,
    ) -> None:
        super().__init__()
        self.semantic = SemanticTokenAutoencoder(
            input_dim=int(semantic_dim),
            bottleneck_dim=int(semantic_bottleneck),
        )
        self.dynamics = SpatialDynamicsAutoencoder(
            vision_dim=int(vision_dim),
            latent_dim=int(latent_dim),
            num_tokens=int(num_tokens),
            hidden_dim=int(dynamics_hidden),
            num_layers=int(dynamics_layers),
            num_heads=int(dynamics_heads),
            ffn_dim=int(dynamics_ffn),
            z_loss_weight=float(dynamics_z_weight),
        )

    def parameter_summary(self) -> dict[str, int]:
        sem = sum(p.numel() for p in self.semantic.parameters())
        dyn = sum(p.numel() for p in self.dynamics.parameters())
        return {"semantic_ae": sem, "dynamics_ae": dyn, "total": sem + dyn}
