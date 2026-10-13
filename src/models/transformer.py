import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Plain multi-head self-attention written with Linear + matmul.

    nn.MultiheadAttention exports to ONNX through a fused path full of
    reshape/gather nodes that INT8 quantizers handle poorly. Spelling the
    attention out keeps the graph to MatMul/Softmax, which quantizes cleanly
    and runs well on small CPUs.
    """

    def __init__(self, d_model, heads, dropout):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.h, self.dk = heads, d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # (B, N, D)
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk), dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model, heads, ff_mult, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = Attention(d_model, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_model * ff_mult, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # pre-norm residual block
        x = x + self.drop(self.attn(self.n1(x)))
        return x + self.drop(self.ff(self.n2(x)))


class HARTransformer(nn.Module):
    """Patch-based transformer encoder for sensor windows.

    A strided conv turns the window into patch tokens (like ViT on 1D),
    which keeps the sequence short and attention cheap.
    """

    def __init__(self, n_channels, n_classes, seq_len=128, patch=8, d_model=64,
                 heads=4, layers=2, ff_mult=2, dropout=0.1):
        super().__init__()
        if seq_len % patch:
            raise ValueError("seq_len must be divisible by patch")
        self.embed = nn.Conv1d(n_channels, d_model, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, seq_len // patch + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, heads, ff_mult, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):            # x: (B, T, C)
        tok = self.embed(x.transpose(1, 2)).transpose(1, 2)
        tok = torch.cat([self.cls.expand(tok.shape[0], -1, -1), tok], dim=1) + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(self.norm(tok[:, 0]))
