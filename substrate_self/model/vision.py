"""Vision module — tiny ViT encoder + fusion adapter into TinyGPT.

The substrate's solo vision faculty. Trained on (image, caption) pairs
from `substrate_self.teach.vision`. Once trained, the substrate can describe
images on its own — no Groq call at inference.

Architecture (LLaVA-style, scaled way down):
  1. ViTEncoder: PatchEmbed + N transformer blocks → image embedding
  2. VisionAdapter: linear projection from image embedding into the
     language model's embedding space
  3. AT INFERENCE: prepend projected image tokens to the text token
     sequence, then run TinyGPT as usual. The text model conditions on
     the visual context via the prepended tokens.

This is small enough to train on a single GPU (RTX 4060, 8GB VRAM)
in single-digit hours given a small dataset.

Usage:
    from substrate_self.model.vision import ViTEncoder, VisionAdapter, VLModel

Training and inference scripts: vision_train.py / vision_generate.py.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionConfig:
    image_size: int = 64       # 64x64 RGB by default — tiny, demo-scale
    patch_size: int = 8        # 8 patches per side → 64 patches per image
    in_channels: int = 3
    n_embd: int = 192          # match TinyGPT default for clean fusion
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    bias: bool = True


class PatchEmbed(nn.Module):
    """Image → patch embeddings via a single conv (kernel=patch, stride=patch)."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        assert cfg.image_size % cfg.patch_size == 0
        self.cfg = cfg
        self.n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.proj = nn.Conv2d(cfg.in_channels, cfg.n_embd, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, n_embd, H/p, W/p) → (B, n_patches, n_embd)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    """Standard pre-norm transformer block (no causal mask — full attention)."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = nn.MultiheadAttention(
            cfg.n_embd, cfg.n_head, dropout=cfg.dropout, bias=cfg.bias, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ViTEncoder(nn.Module):
    """Tiny ViT image encoder.

    Output: (B, n_patches + 1, n_embd) where the first token is a CLS token
    summarizing the image. CLS embedding is what the VisionAdapter projects.
    """

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, cfg.n_embd))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        x = self.patch_embed(images)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return x  # (B, n_patches+1, n_embd)


class VisionAdapter(nn.Module):
    """Projects vision encoder output into the language model's embedding
    space and reshapes into a small number of "vision tokens" that get
    prepended to the text sequence.

    Default: take all patch+CLS tokens, project per-token. The fused
    sequence is [vision_tokens..., text_tokens...] going into TinyGPT.
    """

    def __init__(self, vision_dim: int, text_dim: int, n_vision_tokens: Optional[int] = None):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.n_vision_tokens = n_vision_tokens  # None = pass through all
        self.proj = nn.Linear(vision_dim, text_dim, bias=True)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # vision_features: (B, n_patches+1, vision_dim)
        x = self.proj(vision_features)  # (B, n_patches+1, text_dim)
        if self.n_vision_tokens is not None and self.n_vision_tokens < x.size(1):
            # Average-pool down to n_vision_tokens
            B, T, D = x.size()
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.n_vision_tokens).transpose(1, 2)
        return x  # (B, n_vision_tokens, text_dim)


class VLModel(nn.Module):
    """Vision+Language model: ViT encoder → adapter → prepend to TinyGPT.

    During training: image and tokens come in; loss is computed on text
    tokens only (vision tokens have ignore_index=-1 in targets).

    During inference: encode image → adapter → prepend → generate.
    """

    def __init__(self, vit: ViTEncoder, adapter: VisionAdapter, gpt):
        super().__init__()
        self.vit = vit
        self.adapter = adapter
        self.gpt = gpt  # TinyGPT instance

    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        v = self.vit(images)
        v = self.adapter(v)  # (B, n_vis, n_embd)
        B, T_text = text_ids.shape
        # Manually build the input embedding sequence: vision tokens + text token embeddings
        text_emb = self.gpt.tok_emb(text_ids)  # (B, T_text, n_embd)
        full = torch.cat([v, text_emb], dim=1)  # (B, n_vis + T_text, n_embd)
        T_full = full.size(1)
        assert T_full <= self.gpt.cfg.block_size, f"sequence {T_full} > block_size {self.gpt.cfg.block_size}"
        pos = torch.arange(0, T_full, dtype=torch.long, device=full.device).unsqueeze(0)
        x = self.gpt.drop(full + self.gpt.pos_emb(pos))
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.ln_f(x)
        logits = self.gpt.head(x)
        loss = None
        if targets is not None:
            # targets shape must match logits T_full; vision positions get -1 (ignored)
            n_vis = v.size(1)
            ignore = torch.full((B, n_vis), -1, dtype=targets.dtype, device=targets.device)
            full_targets = torch.cat([ignore, targets], dim=1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), full_targets.view(-1), ignore_index=-1)
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.85,
        top_k: Optional[int] = 40,
    ) -> torch.Tensor:
        """Autoregressive generation conditioned on an image."""
        # Encode vision once
        v = self.adapter(self.vit(images))
        n_vis = v.size(1)
        idx = text_ids
        for _ in range(max_new_tokens):
            B, T = idx.size()
            full_T = n_vis + T
            if full_T > self.gpt.cfg.block_size:
                # Crop text from the front to keep within block size
                idx = idx[:, full_T - self.gpt.cfg.block_size:]
                T = idx.size(1)
                full_T = n_vis + T
            text_emb = self.gpt.tok_emb(idx)
            x = torch.cat([v, text_emb], dim=1)
            pos = torch.arange(0, full_T, dtype=torch.long, device=x.device).unsqueeze(0)
            x = self.gpt.drop(x + self.gpt.pos_emb(pos))
            for block in self.gpt.blocks:
                x = block(x)
            x = self.gpt.ln_f(x)
            logits = self.gpt.head(x[:, -1, :]) / max(temperature, 1e-6)
            if top_k is not None:
                v_top, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v_top[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
