"""Finite Scalar Quantization VAE for GHD coefficients.

Encoder produces L token vectors (each of dim D_tok). Each scalar in a
token is quantized to one of `levels[i]` discrete bins, giving a
per-token codebook of size prod(levels). With L tokens you get
discrete latents ready for an autoregressive prior (no codebook
collapse, no commitment loss — see Mentzer et al. 2023).

This file defines:
- `FSQuantizer`            : straight-through scalar quantization
- `FSQVAE`                 : encoder + quantizer + decoder for GHD vectors

Two-stage training:
1. Train FSQVAE with reconstruction MSE only (this file's scope).
2. Train AR Transformer in `ar_prior.py` over discrete tokens,
   conditioned on cond as prefix.
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator."""
    return x + (torch.round(x) - x).detach()


class FSQuantizer(nn.Module):
    """Per-scalar finite quantization. Input shape [..., D_tok] where
    D_tok == len(levels). Each dim i is squashed to [-1, 1] then mapped to
    one of `levels[i]` evenly spaced bins."""

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = list(levels)
        self.register_buffer("_levels", torch.tensor(self.levels, dtype=torch.long))
        # half-step scale per dim
        steps = torch.tensor([(L - 1) / 2.0 for L in self.levels])
        self.register_buffer("_steps", steps)
        # codebook size per token
        cb = 1
        for L in self.levels:
            cb *= L
        self.codebook_size = cb
        # base for token-id encoding
        bases = [1]
        for L in self.levels[:-1]:
            bases.append(bases[-1] * L)
        self.register_buffer("_bases", torch.tensor(bases, dtype=torch.long))

    def forward(self, z: torch.Tensor):
        """Returns (z_q, indices). z_q is differentiable (STE), shape unchanged.
        indices is long tensor with shape z.shape[:-1] giving token id in [0, codebook_size).
        """
        # squash to (-1, 1) softly so gradients flow at the boundaries
        z_squashed = torch.tanh(z)
        # scale to integer grid centered at 0
        scaled = z_squashed * self._steps  # [..., D_tok]
        q = _round_ste(scaled)
        # encode index per dim: shift to [0, L-1]
        idx_per_dim = (torch.round(scaled.detach()) + self._steps.long()).long()
        idx_per_dim = idx_per_dim.clamp(min=torch.zeros_like(self._levels), max=self._levels - 1)
        token_id = (idx_per_dim * self._bases).sum(-1)
        # decode back to floats in [-1, 1]
        z_q = q / self._steps.clamp_min(1e-6)
        return z_q, token_id

    def decode_indices(self, token_id: torch.Tensor) -> torch.Tensor:
        """Inverse of token_id encoding -> z_q in [-1, 1]."""
        rem = token_id.clone()
        idx_per_dim = []
        for L in self.levels:
            idx_per_dim.append(rem % L)
            rem = rem // L
        idx = torch.stack(idx_per_dim, dim=-1)  # [..., D_tok]
        scaled = idx.float() - self._steps
        return scaled / self._steps.clamp_min(1e-6)


class _MLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(F.silu(self.norm(x)))
        h = self.dropout(h)
        return x + self.fc2(F.silu(h))


class FSQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 32,
        hidden_dim: int = 384,
        num_tokens: int = 8,
        levels: List[int] = (8, 8, 5, 5, 5),
        encoder_blocks: int = 3,
        decoder_blocks: int = 6,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.num_tokens = int(num_tokens)
        self.levels = list(levels)
        self.tok_dim = len(self.levels)
        self.latent_dim = self.num_tokens * self.tok_dim

        self.enc_in = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.enc_blocks = nn.ModuleList([_MLPBlock(hidden_dim, dropout) for _ in range(encoder_blocks)])
        self.enc_norm = nn.LayerNorm(hidden_dim)
        self.enc_out = nn.Linear(hidden_dim, self.latent_dim)

        self.quant = FSQuantizer(self.levels)

        self.dec_in = nn.Linear(self.latent_dim + cond_dim, hidden_dim)
        self.dec_blocks = nn.ModuleList([_MLPBlock(hidden_dim, dropout) for _ in range(decoder_blocks)])
        self.dec_norm = nn.LayerNorm(hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor, cond: torch.Tensor):
        h = F.silu(self.enc_in(torch.cat([x, cond], dim=-1)))
        for blk in self.enc_blocks:
            h = blk(h)
        h = F.silu(self.enc_norm(h))
        z = self.enc_out(h)
        z = z.view(-1, self.num_tokens, self.tok_dim)
        z_q, ids = self.quant(z)  # [B, T, Dtok], [B, T]
        return z_q, ids

    def decode(self, z_q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z_flat = z_q.reshape(z_q.size(0), -1)
        h = F.silu(self.dec_in(torch.cat([z_flat, cond], dim=-1)))
        for blk in self.dec_blocks:
            h = blk(h)
        h = F.silu(self.dec_norm(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        z_q, ids = self.encode(x, cond)
        recon = self.decode(z_q, cond)
        return recon, ids
