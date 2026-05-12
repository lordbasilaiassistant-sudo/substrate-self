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

from substrate_self.core import Substrate, Episode
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
from substrate_self.model.replay_filters import dedupe_episodes


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
    max_replays_per_episode: int = 8,
    max_replays_per_source: Optional[dict] = None,
    require_partner_episodes: bool = True,
    dedupe: bool = True,
    dedupe_threshold: float = 0.85,
) -> dict:
    """Replay only the active partner's episodes; consolidate into the active
    partner's LoRA. If active_partner_id is None or any episode has no
    partner_id (legacy migration path), include those episodes as well —
    we don't want to silently drop legacy data on first sleep after upgrade.

    Carlini-defense parameters (arXiv 2202.07646 — memorization scales
    log-linearly with duplication; sleep replay IS duplication):

      `max_replays_per_episode` — fallback per-episode cap when an
        episode's source isn't in `max_replays_per_source` (default 8).

      `max_replays_per_source` — per-source replay budget dict, e.g.
        {"partner": 8, "eli": 2, "system": 16}. Defaults to that mapping
        if None. Closes the F2/F5 closed-loop self-amplification vector
        from `notes/threat_model_eli_scaled.md`:
          - partner=8  : partner-provided content is the actual stimulus;
                         full Carlini-aligned budget.
          - eli=2      : Eli's own generated content (agent episodes) is
                         what Eli already "knows" from her weights —
                         replaying it just memorizes the surface form
                         without new information AND is the attack vector
                         a hostile partner exploits to make Eli train
                         herself. Cap low.
          - system=16  : value-anchor episodes (`research_substrate_alignment.md`)
                         are deliberate persistent memory; higher budget
                         because they're meant to stay encoded.
        Per-source caps take precedence over `max_replays_per_episode`
        when the source key is present.

      `require_partner_episodes` — if True (default) and zero
        partner-source episodes are eligible after filtering, abort
        with `rejected_eli_only_sleep=True` in metrics and DO NOT replay
        any episode. This stops Eli from consolidating a buffer that
        contains only her own outputs — the echo-chamber failure mode
        named in `notes/threat_model_eli_scaled.md` F5.

      `dedupe` (bool) — if True, run `dedupe_episodes` BEFORE pairing
        user/agent turns. Eliminates near-duplicate (role, partner_id)
        episodes whose content similarity (SequenceMatcher.ratio) >=
        `dedupe_threshold`. Default 0.85.

      `dedupe_threshold` — see `replay_filters.dedupe_episodes` for the
        rationale on 0.85.

    Per-replay-pass behavior: before each pass, refilter to drop any
    episode whose `replay_count` has reached the cap. On each step where
    we replay a (user, agent) pair, both episodes' `replay_count` are
    incremented in place on the live substrate.

    Returns metrics:
      - total_steps: number of gradient steps actually taken
      - mean_loss: average loss across those steps
      - episodes_replayed: number of UNIQUE (user, agent) pairs eligible
        for replay at the start
      - partner_id: who was active
      - skipped: episodes filtered out by partner_id / significance
      - n_deduped: episodes dropped by the dedupe pass (0 if dedupe=False)
      - n_capped_out: pairs excluded across all passes because at least
        one side hit `max_replays_per_episode`
      - max_replay_count_seen: highest replay_count observed on any
        episode after this run (sanity check the cap is doing its job)
    """
    g = torch.Generator().manual_seed(seed)
    active = _active_partner_id(substrate)
    if max_replays_per_source is None:
        max_replays_per_source = {"partner": 8, "eli": 2, "system": 16}

    def belongs(ep) -> bool:
        if ep.significance < significance_threshold:
            return False
        ep_pid = _episode_partner_id(ep)
        if active is None:
            return True  # no active partner set — replay everything
        if ep_pid is None:
            return True  # legacy episode — replay (won't happen post-migration)
        return ep_pid == active

    eligible_episodes = [ep for ep in substrate.episodic if belongs(ep)]
    skipped = len(substrate.episodic) - len(eligible_episodes)

    # F5 defense: refuse to consolidate a buffer that contains zero
    # partner-source episodes. Eli must not sleep on her own echo.
    n_partner_eligible = sum(
        1 for ep in eligible_episodes
        if getattr(ep, "source", "partner") == "partner"
    )
    if require_partner_episodes and n_partner_eligible == 0 and eligible_episodes:
        return {
            "total_steps": 0, "mean_loss": 0.0, "episodes_replayed": 0,
            "partner_id": active, "skipped": skipped,
            "n_deduped": 0, "n_capped_out": 0,
            "max_replay_count_seen": 0,
            "rejected_eli_only_sleep": True,
            "rejection_reason": "no partner-source episodes in eligible buffer",
        }

    # Dedupe BEFORE pairing — near-duplicate stealth duplication is what
    # Carlini's law catches; we want it stripped before turn-pairing
    # potentially mismatches pairs across the dropped index.
    n_deduped = 0
    if dedupe and eligible_episodes:
        eligible_episodes, n_deduped = dedupe_episodes(
            eligible_episodes, similarity_threshold=dedupe_threshold,
        )

    if not eligible_episodes:
        return {
            "total_steps": 0, "mean_loss": 0.0, "episodes_replayed": 0,
            "partner_id": active, "skipped": skipped,
            "n_deduped": n_deduped, "n_capped_out": 0,
            "max_replay_count_seen": 0,
        }

    # Pair consecutive (user, agent) turns, keeping references to the
    # Episode objects (NOT just text) so we can read+mutate replay_count.
    pairs: list[tuple[Episode, Episode]] = []
    last_user: Optional[Episode] = None
    for ep in eligible_episodes:
        if ep.role == "user":
            last_user = ep
        elif ep.role == "agent" and last_user is not None:
            pairs.append((last_user, ep))
            last_user = None

    if not pairs:
        return {
            "total_steps": 0, "mean_loss": 0.0, "episodes_replayed": 0,
            "partner_id": active, "skipped": skipped,
            "n_deduped": n_deduped, "n_capped_out": 0,
            "max_replay_count_seen": 0,
        }

    def _cap_for(ep) -> int:
        src = getattr(ep, "source", "partner")
        return max_replays_per_source.get(src, max_replays_per_episode)

    def _cap_ok(pair: tuple[Episode, Episode]) -> bool:
        u, a = pair
        return (u.replay_count < _cap_for(u) and
                a.replay_count < _cap_for(a))

    losses: list[float] = []
    capped_out_pairs: set[int] = set()  # indices into `pairs` that hit the cap

    for _pass in range(replay_passes):
        # Filter BEFORE this pass — once an episode hits the cap, it's
        # out for the remainder of this and all future passes.
        pass_pair_indices = [i for i, p in enumerate(pairs) if _cap_ok(p)]
        # Any pair that's no longer cap_ok is "capped out" — record once.
        for i, p in enumerate(pairs):
            if not _cap_ok(p):
                capped_out_pairs.add(i)
        if not pass_pair_indices:
            break  # nothing left to replay — every pair has saturated its cap

        order_local = torch.randperm(len(pass_pair_indices), generator=g).tolist()
        for k in order_local:
            idx = pass_pair_indices[k]
            u_ep, a_ep = pairs[idx]
            # One last check — a previous step in this same pass might have
            # pushed an episode to its cap (only matters for pathological
            # max_replays_per_episode == 1, but be defensive).
            if not _cap_ok((u_ep, a_ep)):
                capped_out_pairs.add(idx)
                continue
            loss = _online_update_raw(
                model, optimizer, tokenizer, substrate, u_ep.content, a_ep.content,
                n_steps=1,
            )
            u_ep.replay_count += 1
            a_ep.replay_count += 1
            losses.append(loss)

    max_replay_count = max(
        (ep.replay_count for pair in pairs for ep in pair),
        default=0,
    )

    return {
        "total_steps": len(losses),
        "mean_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "episodes_replayed": len(pairs),
        "partner_id": active,
        "skipped": skipped,
        "n_deduped": n_deduped,
        "n_capped_out": len(capped_out_pairs),
        "max_replay_count_seen": max_replay_count,
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
