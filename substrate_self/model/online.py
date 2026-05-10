"""Online weight updates during wake + sleep-replay consolidation.

This is the load-bearing piece for "the model knows what we talked about
through its weights, not through RAG."

Architecture (BetterThanLLM thesis applied at runtime):

  WAKE — after each user/agent turn pair:
      1. Build the formatted dialogue snippet (substrate prefix + turn pair)
      2. One small gradient step on the model (online fine-tune, low lr)
      3. Append the turn to substrate.episodic
      4. The model has now physically changed from the experience

  SLEEP — at end of session:
      1. Replay each significant episode in shuffled order
      2. Multiple gradient steps total (consolidation = repeated exposure)
      3. Wipe substrate.episodic (only the slow-weight changes survive)
      4. Save the model checkpoint AND substrate

  WAKE B (next session):
      1. Load the model — its weights now reflect prior conversations
      2. Load the substrate — its slow state reflects prior conversations
      3. The entity literally IS what it has experienced

No RAG. No prompt-stuffing of past conversations. The model has changed.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from substrate_self.core import Substrate
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.transformer import TinyGPT


def _format_turn(substrate: Substrate, user_text: str, agent_text: str) -> str:
    """Same surface format used during pre-training corpus."""
    return f"User: {user_text.strip()}\n{substrate.name}: {agent_text.strip()}\n"


def online_update(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    substrate: Substrate,
    user_text: str,
    agent_text: str,
    n_steps: int = 1,
    max_seq_len: Optional[int] = None,
) -> float:
    """One (or n_steps) gradient step on the (user, agent) turn pair.

    Returns the loss value. Mutates model and optimizer state.
    Does NOT mutate substrate (caller does that).
    """
    formatted = _format_turn(substrate, user_text, agent_text)
    ids = tokenizer.encode(formatted)
    block = max_seq_len or model.cfg.block_size
    if len(ids) > block + 1:
        ids = ids[-(block + 1):]
    if len(ids) < 2:
        return 0.0

    device = next(model.parameters()).device
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)

    model.train()
    last_loss = 0.0
    for _ in range(n_steps):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())
    model.eval()
    return last_loss


def sleep_replay(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    substrate: Substrate,
    replay_passes: int = 3,
    significance_threshold: float = 0.0,
    seed: int = 0,
) -> dict:
    """Replay episodic buffer in shuffled order N times, gradient-stepping
    on each. After this returns, the substrate caller should call
    `substrate.end_sleep(wipe_episodic=True)` and save both the model
    and the substrate.

    Returns metrics: total_steps, mean_loss, episodes_replayed.

    NOTE (v0.5 Carlini-defense): this is the legacy non-LoRA path. It does
    NOT enforce replay caps or dedupe. The Carlini-defense (replay caps +
    near-duplicate dropping) lives only in `sleep_replay_partner`
    (`online_lora.py`) because the LoRA path is the v0.4+ default and the
    only path used in multi-partner deployments. If you're using this
    legacy path, you accept the unfiltered duplication risk documented in
    `notes/research_discretion.md`. Migrate to `sleep_replay_partner`.
    """
    g = torch.Generator().manual_seed(seed)
    eligible = [(i, ep) for i, ep in enumerate(substrate.episodic)
                if ep.significance >= significance_threshold]
    if not eligible:
        return {"total_steps": 0, "mean_loss": 0.0, "episodes_replayed": 0}

    # Pair consecutive (user, agent) turns
    pairs: list[tuple[str, str]] = []
    last_user: Optional[str] = None
    for _, ep in eligible:
        if ep.role == "user":
            last_user = ep.content
        elif ep.role == "agent" and last_user is not None:
            pairs.append((last_user, ep.content))
            last_user = None
    if not pairs:
        return {"total_steps": 0, "mean_loss": 0.0, "episodes_replayed": 0}

    losses: list[float] = []
    for _pass in range(replay_passes):
        order = torch.randperm(len(pairs), generator=g).tolist()
        for idx in order:
            user_text, agent_text = pairs[idx]
            loss = online_update(model, optimizer, tokenizer, substrate, user_text, agent_text, n_steps=1)
            losses.append(loss)
    return {
        "total_steps": len(losses),
        "mean_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "episodes_replayed": len(pairs),
    }


def save_model_checkpoint(
    model: TinyGPT,
    out_dir: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Path:
    """Save model state. Optimizer state saved separately if given."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), out_dir / "optimizer.pt")
    return out_dir / "model.pt"
