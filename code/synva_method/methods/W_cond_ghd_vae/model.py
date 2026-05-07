from __future__ import annotations

import torch
import torch.nn as nn


def build_norm_1d(num_features: int, norm_type: str = "batch") -> nn.Module:
    norm_type = str(norm_type).lower()
    if norm_type == "batch":
        return nn.BatchNorm1d(num_features)
    if norm_type == "layer":
        return nn.LayerNorm(num_features)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, norm_type: str = "batch"):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = build_norm_1d(out_features, norm_type)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = build_norm_1d(out_features, norm_type)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return self.relu(out + identity)


class ConditionalGHDVAE(nn.Module):
    """ConditionalGHDVAE matching vessel-mesh-editing-master's baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        cond_dim: int,
        cond_embed_dim: int | None = None,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.cond_embed_dim = cond_dim if cond_embed_dim is None else cond_embed_dim
        self.mu_clamp = 30.0
        self.logvar_min = -10.0
        self.logvar_max = 10.0

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim, hidden_dim, norm_type=norm_type)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        if cond_embed_dim is None:
            self.cond_encoder = nn.Identity()
        else:
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.cond_embed_dim),
                nn.ReLU(inplace=True),
            )

        self.fc3 = nn.Linear(latent_dim + self.cond_embed_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim, norm_type=norm_type)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor):
        x = self.res1(self.fc1(x))
        mu = torch.clamp(self.fc21(x), min=-self.mu_clamp, max=self.mu_clamp)
        logvar = torch.clamp(self.fc22(x), min=self.logvar_min, max=self.logvar_max)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, noise_scale: float = 1.0):
        if noise_scale == 0:
            return mu
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std * noise_scale

    def encode_condition(self, cond: torch.Tensor):
        return self.cond_encoder(cond)

    def decode(self, z: torch.Tensor, cond: torch.Tensor):
        cond = self.encode_condition(cond)
        x = torch.cat((z, cond), dim=1)
        return self.fc4(self.res2(self.fc3(x)))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, noise_scale: float = 1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, noise_scale=noise_scale)
        return self.decode(z, cond), mu, logvar
