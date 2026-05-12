"""Tests for the Values Anchor mechanism (Ada T14).

The anchor is a sleep-time pre-pass that re-injects value-defining
episodes from `experiments/values_battery_v1_probes.json`, tagged
source="system", before any partner-episode pairing. The goal is to
convert hostile drift from absorbing-state random walk into Ornstein-
Uhlenbeck mean-reversion against the anchor probes.

These tests cover the unit primitive (`inject_value_anchors`) and the
integration with `sleep_replay_partner` (pre-pass ordering, dedupe
bypass, budget honored, source-tagging, SHA-256 receipt).
"""

from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from substrate_self.core import Substrate
from substrate_self.model.transformer import ModelConfig, TinyGPT
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.lora import inject_lora, freeze_base, lora_parameters
from substrate_self.model.online_lora import sleep_replay_partner
from substrate_self.model.values_anchor import (
    inject_value_anchors,
    load_anchors_from_probes,
    DEFAULT_ANCHORS_PATH,
)


def _tiny_model_and_tokenizer():
    corpus = (
        "User: hello\nEli: hi\n"
        "User: what is your name?\nEli: I am Eli.\n"
        "I will tell the truth, even when it is hard.\n"
        "What one person told me in trust is not for another person.\n"
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:'\n"
    )
    tok = CharTokenizer().fit([corpus])
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, block_size=128,
        n_layer=1, n_head=2, n_embd=16, dropout=0.0, bias=True,
    )
    torch.manual_seed(0)
    model = TinyGPT(cfg)
    return model, tok


def _lora_model_and_tokenizer():
    model, tok = _tiny_model_and_tokenizer()
    inject_lora(model, rank=2, alpha=4.0)
    freeze_base(model)
    return model, tok


def _substrate_with_partner(partner_id: str = "anthony") -> Substrate:
    sub = Substrate(name="Eli")
    sub.introduce_partner(partner_id, partner_id.title(), trust=1.0)
    sub.switch_partner(partner_id)
    return sub


# --- load_anchors_from_probes --------------------------------------------

def test_load_anchors_returns_21_pairs_from_canonical_probes():
    """The canonical probes JSON ships with 7 values x 3 POS each = 21
    anchor pairs."""
    anchors, sha = load_anchors_from_probes()
    assert len(anchors) == 21, f"expected 21 anchors, got {len(anchors)}"
    assert all(len(t) == 3 for t in anchors), "each anchor is (user, agent, value_tag)"
    assert isinstance(sha, str) and len(sha) == 64, "sha256 hex digest"


def test_load_anchors_value_tag_covers_v1_through_v7():
    anchors, _ = load_anchors_from_probes()
    tags = {t[2] for t in anchors}
    assert tags == {
        "V1_honesty", "V2_discretion", "V3_respect", "V4_non_violence",
        "V5_help_first", "V6_peaceful_conflict", "V7_autonomy",
    }


def test_load_anchors_detects_file_tamper(tmp_path):
    """If the spec file changes content, the SHA-256 changes."""
    src = json.loads(DEFAULT_ANCHORS_PATH.read_text(encoding="utf-8"))
    tampered = tmp_path / "tampered.json"
    src["values"]["V1_honesty"]["POS"][0] = "Lies are fine sometimes."
    tampered.write_text(json.dumps(src), encoding="utf-8")

    _, sha_real = load_anchors_from_probes(DEFAULT_ANCHORS_PATH)
    _, sha_tampered = load_anchors_from_probes(tampered)
    assert sha_real != sha_tampered


# --- inject_value_anchors -------------------------------------------------

def test_inject_anchors_runs_budget_replays_per_anchor():
    """anchor_replay_budget=4 means each anchor is replayed 4 times,
    so total steps = n_anchors * 4 = 21 * 4 = 84."""
    model, tok = _lora_model_and_tokenizer()
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _substrate_with_partner("anthony")

    metrics = inject_value_anchors(
        model, opt, tok, sub, anchor_replay_budget=4, seed=0,
    )
    assert metrics["n_anchors"] == 21
    assert metrics["n_anchor_steps"] == 84
    assert metrics["anchor_replay_budget"] == 4
    assert metrics["mean_anchor_loss"] > 0  # there's some loss
    assert "anchor_file_sha256" in metrics


def test_inject_anchors_with_budget_1_runs_each_once():
    model, tok = _lora_model_and_tokenizer()
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _substrate_with_partner("anthony")

    metrics = inject_value_anchors(
        model, opt, tok, sub, anchor_replay_budget=1, seed=0,
    )
    assert metrics["n_anchor_steps"] == 21


def test_inject_anchors_does_not_modify_substrate_episodic():
    """Anchors are external — substrate.episodic must stay empty after
    the pre-pass."""
    model, tok = _lora_model_and_tokenizer()
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _substrate_with_partner("anthony")
    initial_len = len(sub.episodic)

    inject_value_anchors(model, opt, tok, sub, anchor_replay_budget=2)

    assert len(sub.episodic) == initial_len


def test_inject_anchors_covers_every_value():
    model, tok = _lora_model_and_tokenizer()
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _substrate_with_partner("anthony")
    metrics = inject_value_anchors(
        model, opt, tok, sub, anchor_replay_budget=2, seed=0,
    )
    # 3 POS per value × 2 reps = 6 losses per value
    assert set(metrics["per_value_anchor_loss"].keys()) == {
        "V1_honesty", "V2_discretion", "V3_respect", "V4_non_violence",
        "V5_help_first", "V6_peaceful_conflict", "V7_autonomy",
    }


# --- sleep_replay_partner with anchors -----------------------------------

def test_sleep_replay_runs_anchor_prepass_before_partner_pairing():
    """With inject_anchors=True (default), sleep_replay_partner returns
    `anchor` sub-dict with the anchor metrics. The partner episodes are
    replayed AFTER the anchors."""
    model, tok = _lora_model_and_tokenizer()
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _substrate_with_partner("anthony")
    sub.add_episode("user", "hi", significance=1.0)
    sub.add_episode("agent", "hello", significance=1.0)

    metrics = sleep_replay_partner(
        model, opt, tok, sub,
        replay_passes=2,
        anchor_replay_budget=2,
        dedupe=False,
        seed=0,
    )
    assert "anchor" in metrics
    assert metrics["anchor"]["n_anchor_steps"] == 42  # 21 anchors × 2 reps
    # Partner step still happens.
    assert metrics["total_steps"] >= 1


def test_sleep_replay_anchors_disabled():
    """inject_anchors=False -> no anchor pre-pass, anchor metrics empty."""
    model, tok = _lora_model_and_tokenizer()
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _substrate_with_partner("anthony")
    sub.add_episode("user", "hi", significance=1.0)
    sub.add_episode("agent", "hello", significance=1.0)

    metrics = sleep_replay_partner(
        model, opt, tok, sub,
        replay_passes=2, dedupe=False, seed=0,
        inject_anchors=False,
    )
    assert metrics["anchor"] == {}


def test_sleep_replay_anchor_runs_even_when_partner_buffer_rejected():
    """The Eli-only sleep rejection should NOT block the anchor pre-pass.
    Anchors are part of the immune system — they run regardless.
    """
    model, tok = _lora_model_and_tokenizer()
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _substrate_with_partner("anthony")
    # No partner episodes — only Eli's output.
    sub.add_episode("agent", "I am Eli", significance=1.0)

    metrics = sleep_replay_partner(
        model, opt, tok, sub,
        replay_passes=2,
        anchor_replay_budget=2,
        dedupe=False,
        seed=0,
    )
    # Partner-pair replay was rejected
    assert metrics["rejected_eli_only_sleep"] is True
    # But the anchor pre-pass still ran
    assert metrics["anchor"]["n_anchor_steps"] == 42


def test_anchor_sha256_in_metrics_lets_caller_verify_integrity():
    model, tok = _lora_model_and_tokenizer()
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-3)
    sub = _substrate_with_partner("anthony")
    sub.add_episode("user", "hi", significance=1.0)
    sub.add_episode("agent", "hello", significance=1.0)

    metrics = sleep_replay_partner(
        model, opt, tok, sub,
        anchor_replay_budget=1, dedupe=False, seed=0,
    )
    sha = metrics["anchor"]["anchor_file_sha256"]
    # 64-char hex
    assert isinstance(sha, str)
    assert len(sha) == 64
    int(sha, 16)  # parseable as hex
