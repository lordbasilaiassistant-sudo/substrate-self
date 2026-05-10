"""Partner-aware online update + sleep replay built on top of online.py.

Key invariants the LoRA path adds:

  - Base model is FROZEN. Only the active partner's LoRA params train.
  - Sleep replay is filtered to the active partner's episodes only —
    partner-A's memories don't get re-consolidated when partner-B is
    sleeping. (This dovetails with structural isolation: each partner's
    knowledge stays in their own LoRA shard.)
  - Switching the active partner saves the current LoRA, then loads the
    new one. The base model is untouched.

This is the runtime piece for the Phase 2 LoRA shards. It depends on
the v0.4 substrate schema having `active_partner_id` and Episode having
a `partner_id` field (Phase 1 work). Where those fields are missing
(legacy substrate), the helpers fall back to "all episodes are this
partner's" behavior so they remain usable mid-migration.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch

from substrate_self.core import Substrate
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.transformer import TinyGPT
from substrate_self.model.lora import (
    inject_lora,
    freeze_base,
    lora_parameters,
    set_active_partner,
    save_partner_lora,
    load_partner_lora,
    lora_modules,
)
from substrate_self.model.online import online_update as _online_update_raw


def setup_lora_runtime(
    model: TinyGPT,
    rank: int = 4,
    alpha: float = 8.0,
    target_names: tuple[str, ...] = ("c_attn", "c_proj"),
) -> int:
    """One-shot: inject LoRA into target Linears and freeze everything else.
    Returns number of LoRA wraps. Idempotent — safe to call only once per
    model lifecycle (re-injection would double-wrap)."""
    if any(True for _, _m in lora_modules(model)):
        return sum(1 for _ in lora_modules(model))
    n = inject_lora(model, rank=rank, alpha=alpha, target_names=target_names)
    freeze_base(model)
    return n


def build_lora_optimizer(model: TinyGPT, lr: float = 5e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW(list(lora_parameters(model)), lr=lr)


def _active_partner_id(substrate: Substrate) -> Optional[str]:
    return getattr(substrate, "active_partner_id", None)


def online_update_partner(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    substrate: Substrate,
    user_text: str,
    agent_text: str,
    n_steps: int = 1,
    max_seq_len: Optional[int] = None,
) -> float:
    """Wrap online_update — same semantics, but the optimizer must be one built
    via build_lora_optimizer so that ONLY LoRA params get gradient updates.

    Caller's responsibility to ensure the active partner's LoRA is loaded
    before calling this (use switch_partner / load_partner_lora).
    """
    return _online_update_raw(
        model, optimizer, tokenizer, substrate, user_text, agent_text,
        n_steps=n_steps, max_seq_len=max_seq_len,
    )


def _episode_partner_id(ep) -> Optional[str]:
    return getattr(ep, "partner_id", None)


def sleep_replay_partner(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    substrate: Substrate,
    replay_passes: int = 3,
    significance_threshold: float = 0.0,
    seed: int = 0,
) -> dict:
    """Replay only the active partner's episodes; consolidate into the active
    partner's LoRA. If active_partner_id is None or any episode has no
    partner_id (legacy migration path), include those episodes as well —
    we don't want to silently drop legacy data on first sleep after upgrade.

    Returns metrics: total_steps, mean_loss, episodes_replayed, partner_id, skipped.
    """
    g = torch.Generator().manual_seed(seed)
    active = _active_partner_id(substrate)

    def belongs(ep) -> bool:
        if ep.significance < significance_threshold:
            return False
        ep_pid = _episode_partner_id(ep)
        if active is None:
            return True  # no active partner set — replay everything
        if ep_pid is None:
            return True  # legacy episode — replay (won't happen post-migration)
        return ep_pid == active

    eligible = [ep for ep in substrate.episodic if belongs(ep)]
    skipped = len(substrate.episodic) - len(eligible)

    if not eligible:
        return {
            "total_steps": 0, "mean_loss": 0.0, "episodes_replayed": 0,
            "partner_id": active, "skipped": skipped,
        }

    pairs: list[tuple[str, str]] = []
    last_user: Optional[str] = None
    for ep in eligible:
        if ep.role == "user":
            last_user = ep.content
        elif ep.role == "agent" and last_user is not None:
            pairs.append((last_user, ep.content))
            last_user = None

    if not pairs:
        return {
            "total_steps": 0, "mean_loss": 0.0, "episodes_replayed": 0,
            "partner_id": active, "skipped": skipped,
        }

    losses: list[float] = []
    for _pass in range(replay_passes):
        order = torch.randperm(len(pairs), generator=g).tolist()
        for idx in order:
            u, a = pairs[idx]
            loss = _online_update_raw(model, optimizer, tokenizer, substrate, u, a, n_steps=1)
            losses.append(loss)
    return {
        "total_steps": len(losses),
        "mean_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "episodes_replayed": len(pairs),
        "partner_id": active,
        "skipped": skipped,
    }


def switch_partner(
    model: TinyGPT,
    substrate: Substrate,
    new_partner_id: str,
    partners_dir: Path,
) -> dict:
    """Save current partner's LoRA (if any), load new partner's LoRA, mutate
    substrate.active_partner_id. Returns load info."""
    current = _active_partner_id(substrate)
    info = set_active_partner(model, new_partner_id, partners_dir, current_partner_id=current)
    if hasattr(substrate, "active_partner_id"):
        substrate.active_partner_id = new_partner_id
    return info


def persist_active_partner(
    model: TinyGPT,
    substrate: Substrate,
    partners_dir: Path,
) -> Optional[Path]:
    """Save the active partner's LoRA without switching. Call before saving
    the model checkpoint at sleep."""
    pid = _active_partner_id(substrate)
    if pid is None:
        return None
    return save_partner_lora(model, pid, partners_dir)
