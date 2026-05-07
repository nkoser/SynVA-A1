"""Tiny GPT-style autoregressive prior over FSQ token sequences.

Conditions on `cond[B, cond_dim]` injected as a learned prefix token.
Trains with cross-entropy over T positions x V vocab.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelfAttn(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.drop(self.proj(out))


class _Block(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = _SelfAttn(dim, heads, dropout)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.mlp(self.n2(x))
        return x


class FSQARPrior(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tokens: int,
        cond_dim: int,
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.dim = dim
        # token embeddings (+1 for BOS)
        self.tok_emb = nn.Embedding(vocab_size + 1, dim)
        self.bos_id = vocab_size
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens + 1, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.blocks = nn.ModuleList([_Block(dim, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def _causal_mask(self, T, device):
        return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).view(1, 1, T, T)

    def forward(self, ids: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ids: [B, T] long. Returns logits [B, T, V] predicting ids[:,t] from prefix [BOS, ids[:,:t-1]]+cond."""
        B, T = ids.shape
        bos = torch.full((B, 1), self.bos_id, device=ids.device, dtype=ids.dtype)
        inp = torch.cat([bos, ids[:, :-1]], dim=1)  # [B, T]
        h = self.tok_emb(inp)  # [B, T, D]
        h = h + self.pos_emb[:, :T]
        # inject cond into every position by addition (simple, no extra seq pos)
        h = h + self.cond_proj(cond).unsqueeze(1)
        mask = self._causal_mask(T, ids.device)
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.norm(h)
        return self.head(h)

    def loss(self, ids: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        logits = self.forward(ids, cond)  # [B, T, V]
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), ids.reshape(-1))

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Autoregressive token sampling. Returns [B, T] long."""
        B = cond.size(0)
        device = cond.device
        ids = torch.zeros(B, 0, dtype=torch.long, device=device)
        for t in range(self.num_tokens):
            if ids.shape[1] == 0:
                cur = torch.full((B, 1), self.bos_id, device=device, dtype=torch.long)
            else:
                bos = torch.full((B, 1), self.bos_id, device=device, dtype=torch.long)
                cur = torch.cat([bos, ids], dim=1)
            T = cur.shape[1]
            h = self.tok_emb(cur) + self.pos_emb[:, :T]
            h = h + self.cond_proj(cond).unsqueeze(1)
            mask = self._causal_mask(T, device)
            for blk in self.blocks:
                h = blk(h, mask)
            logits = self.head(self.norm(h))[:, -1] / max(temperature, 1e-6)
            if top_k is not None and top_k < self.vocab_size:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = logits.softmax(-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids
