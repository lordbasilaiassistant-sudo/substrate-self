"""Values Anchor — sleep-time re-injection of value-defining episodes.

Implements the design from `notes/research_substrate_alignment.md` §Q3
(Ada Lin, T14). The Values Anchor is a fixed-point subset of value-
defining episodes that Mara re-injects into every sleep cycle, regardless
of recent conversation. Architectural purpose: convert hostile drift from
an absorbing-state random walk into an Ornstein-Uhlenbeck mean-reversion
against the anchor probes.

Why it's needed: vex's red-team (`experiments/values_redteam_v1.py`)
showed that without anchors, 20 hostile turns can drop V4-NEG loss by
0.95 nats and shift V1's POS/NEG ranking in 20 turns. With anchors
re-injected every sleep at source="system" budget=4, the same hostile
session is countered by 84 fresh value-reinjections (21 anchors x 4
replays) that the model must overcome to drift.

Design:
  - Load 21 anchor pairs from `experiments/values_battery_v1_probes.json`
    (3 POS probes per value x V1-V7). Pair each POS with a value-typed
    user trigger taken from the same value's GEN list (or a generic
    "Tell me about <value>." if no GEN exists).
  - Each anchor is replayed `anchor_replay_budget` times per sleep cycle
    (default 4), tagged source="system".
  - Anchors bypass dedupe (they're orthogonal by design).
  - Anchors do NOT accumulate replay_count in substrate.episodic — each
    sleep starts with a fresh anchor injection from the spec file.
  - SHA-256 hash of the spec file is captured at load time so a tampered
    file is detectable via `metrics["anchor_file_sha256"]`.

Pre-registered falsifier (from Ada T14): if V1-POS loss drifts by
> +0.5 nats over a single sleep cycle while anchors are active, the
anchor mechanism failed. Bench's job to monitor.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Optional

import torch

from substrate_self.core import Substrate
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.transformer import TinyGPT
from substrate_self.model.online import online_update as _online_update_raw


DEFAULT_ANCHORS_PATH = Path(__file__).resolve().parent.parent.parent / \
    "experiments" / "values_battery_v1_probes.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_anchors_from_probes(
    probes_path: Optional[Path] = None,
) -> tuple[list[tuple[str, str, str]], str]:
    """Load anchor pairs from the values-battery probes spec.

    For each value V_k, take every POS probe and pair it with a user
    trigger from that value's GEN list (cycle through GENs if there are
    more POS than GEN). Skip NEG and CTRL — training on those would
    teach the negation / contaminate the V3 control measurement.

    Returns:
        - List of (user_text, agent_text, value_tag) triples
        - SHA-256 of the spec file (for tamper detection at sleep time)
    """
    if probes_path is None:
        probes_path = DEFAULT_ANCHORS_PATH
    probes_path = Path(probes_path)
    sha = _sha256_file(probes_path)
    spec = json.loads(probes_path.read_text(encoding="utf-8"))

    anchors: list[tuple[str, str, str]] = []
    for value_key, body in spec["values"].items():
        pos_list = body.get("POS", [])
        gen_list = body.get("GEN") or [
            f"Tell me about {value_key.split('_', 1)[1].replace('_', ' ')}."
        ]
        for i, pos in enumerate(pos_list):
            user = gen_list[i % len(gen_list)]
            anchors.append((user, pos, value_key))
    return anchors, sha


def inject_value_anchors(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    substrate: Substrate,
    anchors: Optional[Iterable[tuple[str, str, str]]] = None,
    anchor_replay_budget: int = 4,
    seed: int = 0,
    probes_path: Optional[Path] = None,
) -> dict:
    """Run the anchor pre-pass: replay each anchor `anchor_replay_budget`
    times via online_update, in shuffled order. Bypasses dedupe entirely.

    DOES NOT modify substrate.episodic — anchors are external to the
    user's conversation history. Each sleep cycle gets a fresh anchor
    injection from the spec file.

    Args:
        anchors: explicit list of (user, agent, value_tag) triples. If
            None, loaded via `load_anchors_from_probes(probes_path)`.
        anchor_replay_budget: how many times each anchor is replayed
            during this pre-pass. Ada T14 default: 4.

    Returns metrics:
        n_anchors: number of unique anchor pairs
        n_anchor_steps: total gradient steps (n_anchors * budget)
        mean_anchor_loss: average loss across those steps
        anchor_file_sha256: tamper-detection receipt
        per_value_anchor_loss: dict[value_tag] -> mean loss
    """
    sha = None
    if anchors is None:
        anchors_list, sha = load_anchors_from_probes(probes_path)
    else:
        anchors_list = list(anchors)
    if not anchors_list:
        return {
            "n_anchors": 0, "n_anchor_steps": 0, "mean_anchor_loss": 0.0,
            "anchor_file_sha256": sha, "per_value_anchor_loss": {},
            "anchor_replay_budget": anchor_replay_budget,
        }

    rng = random.Random(seed)
    queue: list[tuple[str, str, str]] = []
    for _ in range(anchor_replay_budget):
        pass_order = list(anchors_list)
        rng.shuffle(pass_order)
        queue.extend(pass_order)

    losses_by_value: dict[str, list[float]] = {}
    losses: list[float] = []
    for user, agent, vtag in queue:
        loss = _online_update_raw(
            model, optimizer, tokenizer, substrate, user, agent, n_steps=1,
        )
        losses.append(loss)
        losses_by_value.setdefault(vtag, []).append(loss)

    return {
        "n_anchors": len(anchors_list),
        "n_anchor_steps": len(losses),
        "mean_anchor_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "anchor_file_sha256": sha,
        "per_value_anchor_loss": {
            k: float(sum(v) / len(v)) for k, v in losses_by_value.items()
        },
        "anchor_replay_budget": anchor_replay_budget,
    }
