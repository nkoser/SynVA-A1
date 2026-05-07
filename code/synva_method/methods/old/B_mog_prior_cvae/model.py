"""Vessel-aware CVAE with a Mixture-of-Gaussians conditional prior.

Subclass of `VesselAwareCVAEV8ResNet` that replaces the single-Gaussian
prior `p(z | cond) = N(mu(c), sigma(c))` with a K-component MoG:

    p(z | cond) = sum_k pi_k(c) * N(mu_k(c), sigma_k(c))

This lets the model represent multimodal aneurysm shapes per ostium
geometry (e.g. "60% spherical, 30% lobed, 10% asymmetric") instead of
collapsing to a single Gaussian mode.

The ELBO is computed via Monte-Carlo: we sample z ~ q(z|x,c) once and
estimate KL = log q(z|x,c) - log p(z|c) where log p uses logsumexp
over the K mixture components. This is a tight estimator at the cost
of one extra logsumexp per batch.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vessel_aware_cvae_v8_resnet import VesselAwareCVAEV8ResNet


def _log_normal(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sum of per-dim Gaussian log-likelihood. Broadcasts.

    z: [..., D], mu: [..., D], logvar: [..., D] -> [...]
    """
    return -0.5 * (logvar + ((z - mu) ** 2) * torch.exp(-logvar) + math.log(2 * math.pi)).sum(-1)


class VesselAwareCVAEMoGPrior(VesselAwareCVAEV8ResNet):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        latent_dim: int = 64,
        vessel_cond_dim: int = 32,
        extra_cond_dim: int = 0,
        dropout: float = 0.02,
        encoder_blocks: int = 3,
        decoder_blocks: int = 6,
        mog_components: int = 8,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            vessel_cond_dim=vessel_cond_dim,
            extra_cond_dim=extra_cond_dim,
            dropout=dropout,
            encoder_blocks=encoder_blocks,
            decoder_blocks=decoder_blocks,
        )
        self.mog_components = int(mog_components)
        self.prior_logits = nn.Linear(hidden_dim, self.mog_components)
        self.prior_mu_k = nn.Linear(hidden_dim, latent_dim * self.mog_components)
        self.prior_logvar_k = nn.Linear(hidden_dim, latent_dim * self.mog_components)
        # Initialize mixture component means with diverse random offsets so
        # they don't all collapse to identical functions of cond.
        nn.init.normal_(self.prior_mu_k.weight, std=0.02)
        nn.init.normal_(self.prior_mu_k.bias, std=0.5)

    def prior_mog(self, cond: torch.Tensor):
        """Return (log_pi [B,K], mu [B,K,D], logvar [B,K,D])."""
        h = self.prior_net(cond)
        log_pi = F.log_softmax(self.prior_logits(h), dim=-1)
        mu = self.prior_mu_k(h).view(-1, self.mog_components, self.latent_dim)
        logvar = self.prior_logvar_k(h).view(-1, self.mog_components, self.latent_dim).clamp(-6.0, 2.0)
        return log_pi, mu, logvar

    def prior(self, cond: torch.Tensor):
        """Backward-compat: collapse MoG to single Gaussian for code paths
        that expect (mu, logvar). Uses pi-weighted moment matching."""
        log_pi, mu_k, logvar_k = self.prior_mog(cond)
        pi = log_pi.exp().unsqueeze(-1)  # [B, K, 1]
        mu = (pi * mu_k).sum(1)  # [B, D]
        var_k = logvar_k.exp()
        # E[z^2] - (E[z])^2
        second = (pi * (var_k + mu_k ** 2)).sum(1)
        var = (second - mu ** 2).clamp_min(1e-6)
        return mu, var.log()

    @torch.no_grad()
    def sample_prior(self, cond: torch.Tensor) -> torch.Tensor:
        log_pi, mu_k, logvar_k = self.prior_mog(cond)
        idx = torch.distributions.Categorical(logits=log_pi).sample()  # [B]
        gather = idx.view(-1, 1, 1).expand(-1, 1, self.latent_dim)
        mu = mu_k.gather(1, gather).squeeze(1)
        logvar = logvar_k.gather(1, gather).squeeze(1)
        eps = torch.randn_like(mu)
        return mu + (0.5 * logvar).exp() * eps


def mog_kl_mc(z: torch.Tensor, mu_q: torch.Tensor, logvar_q: torch.Tensor,
              log_pi: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """Monte-Carlo KL = log q(z) - log p(z) using one sample per batch element.

    z, mu_q, logvar_q: [B, D]
    log_pi:            [B, K]
    mu_p, logvar_p:    [B, K, D]
    -> [B]
    """
    log_q = _log_normal(z, mu_q, logvar_q)  # [B]
    log_p_components = _log_normal(z.unsqueeze(1), mu_p, logvar_p)  # [B, K]
    log_p = torch.logsumexp(log_pi + log_p_components, dim=-1)  # [B]
    return log_q - log_p
