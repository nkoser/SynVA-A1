"""Vector-Quantized VAE for GHD coefficients with EMA codebook updates.

Encoder produces T continuous token vectors (each of dim D_code). Each
token is replaced by its nearest codebook entry (codebook of size K, dim D_code).
Codebook is updated by EMA (van den Oord et al. 2017, with EMA from
Razavi et al. 2019). Dead-code reset is supported.

Loss = recon + beta * commitment.
Stage 2 (in `train.py`) trains an autoregressive transformer prior over
discrete code indices, conditioned on `cond` as additive prefix.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim); self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.fc1(F.silu(self.norm(x))); h = self.drop(h)
        return x + self.fc2(F.silu(h))


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, decay: float = 0.99, eps: float = 1e-5,
                 dead_thresh: float = 0.01):
        super().__init__()
        self.K = int(num_codes); self.D = int(code_dim)
        self.decay = float(decay); self.eps = float(eps)
        self.dead_thresh = float(dead_thresh)
        emb = torch.randn(self.K, self.D) * 0.02
        self.register_buffer("embedding", emb)
        self.register_buffer("ema_count", torch.zeros(self.K))
        self.register_buffer("ema_sum", emb.clone())

    def reset_dead_codes(self, encodings_flat: torch.Tensor):
        """Replace codes whose usage rate is below `dead_thresh` with random
        encoder outputs from the current batch."""
        if not self.training:
            return 0
        usage = self.ema_count / (self.ema_count.sum() + 1e-9)
        dead = (usage < self.dead_thresh / self.K).nonzero(as_tuple=False).flatten()
        if dead.numel() == 0:
            return 0
        n = encodings_flat.size(0)
        if n == 0:
            return 0
        idx = torch.randint(0, n, (dead.numel(),), device=encodings_flat.device)
        new_codes = encodings_flat[idx]
        with torch.no_grad():
            self.embedding[dead] = new_codes
            self.ema_sum[dead] = new_codes
            self.ema_count[dead] = 1.0
        return int(dead.numel())

    def forward(self, z: torch.Tensor):
        """z: [B, T, D] -> (z_q [B,T,D] STE, ids [B,T] long, vq_loss scalar, perplexity scalar)."""
        B, T, D = z.shape
        flat = z.reshape(-1, D)
        # nearest neighbor in L2
        dist = (flat.pow(2).sum(-1, keepdim=True)
                - 2 * flat @ self.embedding.t()
                + self.embedding.pow(2).sum(-1))
        ids = dist.argmin(-1)  # [B*T]
        z_q_flat = self.embedding[ids]
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(ids, self.K).float()
                count = onehot.sum(0)
                self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
                n = self.ema_count.sum()
                cluster = (self.ema_count + self.eps) / (n + self.K * self.eps) * n
                self.embedding.copy_(self.ema_sum / cluster.unsqueeze(1))
        commitment = F.mse_loss(flat, z_q_flat.detach())
        z_q = flat + (z_q_flat - flat).detach()
        z_q = z_q.view(B, T, D)
        avg = (F.one_hot(ids, self.K).float().mean(0) + 1e-10)
        perp = torch.exp(-(avg * avg.log()).sum())
        return z_q, ids.view(B, T), commitment, perp, flat


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 32,
        hidden_dim: int = 384,
        num_tokens: int = 8,
        code_dim: int = 32,
        num_codes: int = 256,
        encoder_blocks: int = 3,
        decoder_blocks: int = 6,
        dropout: float = 0.05,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = input_dim; self.cond_dim = cond_dim
        self.num_tokens = int(num_tokens); self.code_dim = int(code_dim)
        self.num_codes = int(num_codes)

        self.enc_in = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.enc_blocks = nn.ModuleList([_MLPBlock(hidden_dim, dropout) for _ in range(encoder_blocks)])
        self.enc_norm = nn.LayerNorm(hidden_dim)
        self.enc_out = nn.Linear(hidden_dim, num_tokens * code_dim)

        self.vq = VectorQuantizerEMA(num_codes, code_dim, decay=ema_decay)

        self.dec_in = nn.Linear(num_tokens * code_dim + cond_dim, hidden_dim)
        self.dec_blocks = nn.ModuleList([_MLPBlock(hidden_dim, dropout) for _ in range(decoder_blocks)])
        self.dec_norm = nn.LayerNorm(hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor, cond: torch.Tensor):
        h = F.silu(self.enc_in(torch.cat([x, cond], dim=-1)))
        for blk in self.enc_blocks:
            h = blk(h)
        h = F.silu(self.enc_norm(h))
        z = self.enc_out(h).view(-1, self.num_tokens, self.code_dim)
        return self.vq(z)

    def decode(self, z_q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z_flat = z_q.reshape(z_q.size(0), -1)
        h = F.silu(self.dec_in(torch.cat([z_flat, cond], dim=-1)))
        for blk in self.dec_blocks:
            h = blk(h)
        h = F.silu(self.dec_norm(h))
        return self.dec_out(h)

    def decode_indices(self, ids: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z_q = self.vq.embedding[ids]
        return self.decode(z_q, cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        z_q, ids, commitment, perp, flat = self.encode(x, cond)
        recon = self.decode(z_q, cond)
        return recon, ids, commitment, perp, flat
