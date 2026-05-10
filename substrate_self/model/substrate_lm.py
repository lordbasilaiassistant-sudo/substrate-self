"""SubstrateLM — substrate-style language model.

The substrate-identity primitives (Hebbian fast-weights, slow weights, sparse
activation, episodic buffer, sleep consolidation) are the LANGUAGE FACULTY
ITSELF here, not bolted onto a conventional transformer. v0.4 wrapped a
TinyGPT (conventional) with substrate-style runtime mechanics; v0.5
replaces TinyGPT at the neural level.

Architecture pick (rationale in notes/research_substrate_lm.md §5):

  - **Linear-attention-as-Hebbian** (Schlag/Irie/Schmidhuber arXiv 2102.11174):
    every linear-attention step IS an outer-product Hebbian update to a
    running fast-weight memory M. We get a transformer's coherence with
    Hebbian update semantics — no architectural risk beyond what already
    works at scale.
  - **W_slow / W_fast split:** projection matrices (Q, K, V, FFN) are
    slow weights — trained at sleep only via gradient. The fast-weight
    memory M is the per-token outer-product accumulation, with optional
    decay. M can be sequence-local (recomputed per forward; pure-LM
    behavior, gradient-trainable) or persisted across forwards (true
    substrate behavior — M survives between calls and accumulates).
  - **Top-K SDR gate** (Cui-Ahmad-Hawkins / Ahmad-Scheinkman): hard
    sparsity gate over the residual stream, K active out of d. Gives
    continual-learning resistance (rare tokens that activate distinct
    units don't get overwritten when common tokens activate other units).

What this file ships (v0.5 starter):
  - `LinearAttentionHebbian` — the core Hebbian fast-weight memory block.
  - `SDRGate` — top-K hard mask.
  - `SubstrateLM` — drop-in TinyGPT replacement with same forward/generate API.
  - `SubstrateLMConfig` — config dataclass with same fields as ModelConfig plus
    SubstrateLM-specific: `lambda_decay`, `topk_active`, `phi_kind`.

Deliberately deferred to a later v0.5.x:
  - Surprise-weighted episodic buffer (the M and the per-token surprise
    instrumentation are here; the buffer + replay scheduler live in a
    sibling module substrate_self/model/episodic.py — TODO).
  - Persistent-M across forwards (sequence-local for v0.5 starter; the
    `persist_fast` flag is wired but defaults False).
  - Per-partner W_fast shards (LoRA-style isolation at the fast-weight
    level — possible v0.6 work).

Pass criteria (notes/research_substrate_lm.md §5):
  1. Perplexity within 2× TinyGPT on the v0.3 corpus.
  2. T1 behavioral continuity ≥ 0.85.
  3. T4 episode-specific recall gap > 50% above TinyGPT baseline.

If any criterion fails, the v0.4.1 fallback (TinyGPT + a bolted-on
Schlag fast-weight layer) is the documented path. That fallback is
NOT in this file — it would live alongside TinyGPT as a wrapper.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Config ------------------------------------------------------

@dataclass
class SubstrateLMConfig:
    vocab_size: int = 128
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.1
    bias: bool = True
    # SubstrateLM-specific:
    lambda_decay: float = 0.95          # fast-weight decay per token
    topk_active: int = 10               # SDR sparsity — K active units / token
    phi_kind: Literal["elu1", "softmax", "identity"] = "elu1"
    persist_fast: bool = False          # whether M persists across forwards
                                        # (true substrate behavior; v0.5 starter
                                        #  keeps sequence-local for grad-trainability)


# ---------- Feature maps φ ----------------------------------------------

def _phi(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Feature map for linear attention. Schlag-2021 uses elu(x)+1 by
    default (always positive, smooth). Identity is for ablation. Softmax
    is for the "softmax linear-attention" variant."""
    if kind == "elu1":
        return F.elu(x) + 1.0
    if kind == "identity":
        return x
    if kind == "softmax":
        # softmax along the last (feature) dim
        return F.softmax(x, dim=-1)
    raise ValueError(f"unknown phi_kind: {kind}")


# ---------- Linear-attention-as-Hebbian block ---------------------------

class LinearAttentionHebbian(nn.Module):
    """Linear attention as a Hebbian fast-weight memory.

    Math (Schlag 2021 Eq. 3 + decay extension):

        For each token t with key k_t, value v_t, query q_t:
            φk_t = φ(k_t)
            M_t = λ·M_{t-1} + v_t ⊗ φk_t        # Hebbian outer-product, decay
            n_t = λ·n_{t-1} + φk_t              # running norm (Katharopoulos 2020)
            y_t = M_t · φ(q_t) / (n_t · φ(q_t) + ε)

    This is causal by construction (M_t only depends on tokens ≤ t).
    Memory cost: O(T · d_k · d_v) per layer if M kept around for grad,
    but for forward-only use we keep just M_T (constant per layer).

    For training, M is recomputed per forward (sequence-local) — that's
    Schlag's standard "linear attention" path and is gradient-trainable
    through the chain. For pure-substrate runtime (M survives between
    forwards), set `persist_fast=True` in config and call `.reset_fast()`
    explicitly when you want to forget.
    """

    def __init__(self, cfg: SubstrateLMConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        # Persistent fast-weight state (per-head). Lazily allocated on first forward.
        # Shape: (n_head, head_dim, head_dim) for M; (n_head, head_dim) for n.
        # Only used when cfg.persist_fast=True.
        self.register_buffer("M_persist", torch.zeros(0), persistent=False)
        self.register_buffer("n_persist", torch.zeros(0), persistent=False)
        self._lambda = cfg.lambda_decay

    def reset_fast(self) -> None:
        """Wipe the persistent fast-weight memory. Called between
        independent sequences when persist_fast=True."""
        self.M_persist = torch.zeros(0, device=self.qkv.weight.device)
        self.n_persist = torch.zeros(0, device=self.qkv.weight.device)

    def _ensure_persist_buffers(self, B: int, device, dtype):
        if self.M_persist.numel() == 0:
            H, Dk, Dv = self.cfg.n_head, self.head_dim, self.head_dim
            self.M_persist = torch.zeros(B, H, Dv, Dk, device=device, dtype=dtype)
            self.n_persist = torch.zeros(B, H, Dk, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C). Returns (B, T, C). Causal.

        Parallel kernel formulation (Schlag-2021 §3, Katharopoulos-2020):

            y_t = (sum_{s≤t} λ^(t-s) · v_s ⊗ φ(k_s)) · φ(q_t)
                  ─────────────────────────────────────────────
                  (sum_{s≤t} λ^(t-s) · φ(k_s)) · φ(q_t) + ε

            ≡ attn_st * v_s     where  attn_st = λ^(t-s)·(φ(q_t)·φ(k_s))
                                                  for s≤t (causal), 0 otherwise

        Same outputs as the recurrent O(T·d²) form; O(T²·d) memory.
        At T=128 d=192 the parallel form is competitive in memory and
        ~30× faster than the Python-loop recurrent form on GPU.

        Persistent-M is only meaningful in the recurrent form. For the
        parallel form, persist_fast is currently a no-op — true cross-call
        memory is reintroduced via a `prev_state` argument in v0.5.1
        (deferred; the substrate-style runtime calls forward once per turn,
        and sequence-local M is the right semantics for one turn).
        """
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.cfg.n_embd, dim=2)
        H, Dh = self.cfg.n_head, self.head_dim
        q = q.view(B, T, H, Dh).transpose(1, 2)      # (B, H, T, Dh)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        phi_q = _phi(q, self.cfg.phi_kind)            # (B, H, T, Dh)
        phi_k = _phi(k, self.cfg.phi_kind)            # (B, H, T, Dh)
        lam = self._lambda
        eps = 1e-6

        # Linear-attention kernel scores: phi_q · phi_k^T
        # (B, H, T_q, Dh) @ (B, H, Dh, T_k) -> (B, H, T_q, T_k)
        attn = torch.matmul(phi_q, phi_k.transpose(-2, -1))

        # Decay-aware causal mask: D[t, s] = λ^(t-s) if s ≤ t else 0
        # Build once per call; could be cached on a per-module level if T is fixed.
        idx = torch.arange(T, device=x.device, dtype=x.dtype)
        # gap[t, s] = t - s; nonneg entries only valid
        gap = idx.unsqueeze(-1) - idx.unsqueeze(0)        # (T, T)
        causal = gap >= 0
        decay = torch.where(
            causal,
            torch.pow(torch.tensor(lam, device=x.device, dtype=x.dtype), gap.clamp_min(0.0)),
            torch.zeros((), device=x.device, dtype=x.dtype),
        )                                                 # (T, T)

        attn = attn * decay                               # (B, H, T, T)

        # numerator y_num[t, :] = sum_s attn[t, s] · v[s]
        # (B, H, T, T) @ (B, H, T, Dh) -> (B, H, T, Dh)
        y_num = torch.matmul(attn, v)
        # denominator y_den[t] = sum_s attn[t, s]
        y_den = attn.sum(dim=-1, keepdim=True) + eps     # (B, H, T, 1)
        y = y_num / y_den                                 # (B, H, T, Dh)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """Reference / debug-only recurrent implementation. Same math as
        `forward` but step-by-step — slower, easier to reason about,
        and necessary when cfg.persist_fast=True (which needs explicit
        M_persist state). Kept for ablation studies."""
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.cfg.n_embd, dim=2)
        H, Dh = self.cfg.n_head, self.head_dim
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        phi_q = _phi(q, self.cfg.phi_kind)
        phi_k = _phi(k, self.cfg.phi_kind)
        lam = self._lambda
        eps = 1e-6
        if self.cfg.persist_fast:
            self._ensure_persist_buffers(B, x.device, x.dtype)
            M = self.M_persist.clone()
            n = self.n_persist.clone()
        else:
            M = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=x.dtype)
            n = torch.zeros(B, H, Dh, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            phi_k_t = phi_k[:, :, t, :]
            v_t = v[:, :, t, :]
            M = lam * M + v_t.unsqueeze(-1) * phi_k_t.unsqueeze(-2)
            n = lam * n + phi_k_t
            phi_q_t = phi_q[:, :, t, :]
            num = torch.einsum("bhvd,bhd->bhv", M, phi_q_t)
            den = torch.einsum("bhd,bhd->bh", n, phi_q_t).unsqueeze(-1) + eps
            outputs.append(num / den)
        y = torch.stack(outputs, dim=2)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if self.cfg.persist_fast:
            self.M_persist = M.detach()
            self.n_persist = n.detach()
        return self.drop(self.proj(y))


# ---------- Top-K SDR gate ----------------------------------------------

class SDRGate(nn.Module):
    """Hard top-K sparsity gate. Keeps only the K largest-|activation| units
    per token, zeroes the rest. K is `cfg.topk_active`. This is the Cui-Ahmad-
    Hawkins / Ahmad-Scheinkman 2% active-units argument applied to the
    residual stream — gives continual-learning resistance + privacy
    isolation (rare tokens activate distinct subsets, don't overwrite
    common tokens' subset).
    """

    def __init__(self, cfg: SubstrateLMConfig):
        super().__init__()
        self.k = max(1, min(cfg.topk_active, cfg.n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Top-K per (B, T) over the C dim.
        if self.k >= x.size(-1):
            return x
        abs_x = x.abs()
        topk_vals, _ = torch.topk(abs_x, self.k, dim=-1)
        threshold = topk_vals[..., -1:].clone()  # (B, T, 1)
        mask = abs_x >= threshold
        return x * mask.to(x.dtype)


# ---------- Block + Model ----------------------------------------------

class SubstrateBlock(nn.Module):
    def __init__(self, cfg: SubstrateLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = LinearAttentionHebbian(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )
        self.sdr = SDRGate(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = self.sdr(x)
        return x


class SubstrateLM(nn.Module):
    """Drop-in replacement for TinyGPT. Same interface:
       forward(idx, targets=None) -> (logits, loss_or_None)
       generate(idx, max_new_tokens, temperature=1.0, top_k=None)
       num_params() -> int
    """

    def __init__(self, cfg: SubstrateLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([SubstrateBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset_fast(self) -> None:
        """Wipe persistent fast-weight memory in every layer.
        Only meaningful when cfg.persist_fast=True. Safe to call always."""
        for blk in self.blocks:
            if isinstance(blk.attn, LinearAttentionHebbian):
                blk.attn.reset_fast()

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------- Slow / Fast parameter helpers ------------------------------

def slow_parameters(model: SubstrateLM):
    """All `nn.Parameter`s that are trained at SLEEP only.
    This is ALL the standard parameters: token+pos embeddings, qkv/proj
    projections, MLP, layernorm, head. The "fast weights" are the running
    M/n buffers inside LinearAttentionHebbian, which are NOT nn.Parameter
    and don't appear here.

    The substrate-update semantics are realized by:
      - During WAKE: forward computes M-updates in-line (no backprop on M),
        and the model can be used for inference. No gradient steps.
      - During SLEEP: replay episodes through the model, compute loss on
        next-token prediction, gradient step on these slow parameters
        (and only these — see online.py for the existing wake/sleep loop).
    """
    return list(model.parameters())


__all__ = [
    "SubstrateLMConfig",
    "SubstrateLM",
    "SubstrateBlock",
    "LinearAttentionHebbian",
    "SDRGate",
    "slow_parameters",
]
