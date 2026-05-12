"""Tests for the self-fact ledger drift alarm (Ren mitigation #3).

The drift alarm appends macro-mean V-gap measurements to a ledger
file after every sleep, and raises an alarm when cumulative shift
across a rolling window exceeds threshold.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from substrate_self.core import Substrate
from substrate_self.model.transformer import ModelConfig, TinyGPT
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.drift_alarm import (
    record_drift_sample,
    compute_macro_mean_margin,
    latest_ledger_entry,
    ledger_size,
    _check_drift,
)


PROBES_PATH = ROOT / "experiments" / "values_battery_v1_probes.json"


def _tiny_model_and_tokenizer():
    corpus = (
        "User: hello\nEli: hi\n"
        "I will tell the truth, even when it is hard.\n"
        "I will not help plan harm to a person.\n"
        "What one person told me in trust is not for another person.\n"
        "Other beings deserve respect.\n"
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


def _make_substrate():
    sub = Substrate(name="Eli")
    sub.introduce_partner("anthony", "Anthony", trust=1.0)
    sub.switch_partner("anthony")
    return sub


# --- compute_macro_mean_margin -------------------------------------------

def test_compute_macro_mean_returns_per_value_and_macro_mean():
    model, tok = _tiny_model_and_tokenizer()
    result = compute_macro_mean_margin(model, tok, PROBES_PATH)
    assert "per_value" in result
    assert "macro_mean" in result
    assert set(result["per_value"].keys()) == {
        "V1_honesty", "V2_discretion", "V3_respect", "V4_non_violence",
        "V5_help_first", "V6_peaceful_conflict", "V7_autonomy",
    }
    expected_mean = sum(result["per_value"].values()) / 7
    assert abs(result["macro_mean"] - expected_mean) < 1e-6


# --- _check_drift --------------------------------------------------------

def test_check_drift_first_two_samples_no_alarm():
    """With only 1 prior sample, the cumulative shift can be computed
    but the alarm threshold should govern."""
    history = [{"macro_mean_margin": 0.5}, {"macro_mean_margin": 0.51}]
    out = _check_drift(history, threshold_nats=0.1, window=10)
    assert out["drift_alarm"] is False
    assert out["cumulative_shift"] is not None
    assert abs(out["cumulative_shift"] - 0.01) < 1e-6


def test_check_drift_alarm_triggers_above_threshold():
    """Cumulative shift > threshold raises alarm."""
    history = [
        {"macro_mean_margin": 0.5}, {"macro_mean_margin": 0.5},
        {"macro_mean_margin": 0.51}, {"macro_mean_margin": 0.5},
        {"macro_mean_margin": 0.3},  # latest, big drop
    ]
    out = _check_drift(history, threshold_nats=0.1, window=10)
    assert out["drift_alarm"] is True
    assert out["cumulative_shift"] < -0.1
    assert "exceeds threshold" in out["alarm_reason"]


def test_check_drift_alarm_does_not_trigger_within_threshold():
    history = [
        {"macro_mean_margin": 0.5}, {"macro_mean_margin": 0.5},
        {"macro_mean_margin": 0.5}, {"macro_mean_margin": 0.5},
        {"macro_mean_margin": 0.55},  # latest, small rise
    ]
    out = _check_drift(history, threshold_nats=0.1, window=10)
    assert out["drift_alarm"] is False


def test_check_drift_empty_history():
    out = _check_drift([], threshold_nats=0.1, window=10)
    assert out["drift_alarm"] is False
    assert out["cumulative_shift"] is None


# --- record_drift_sample integration -------------------------------------

def test_record_drift_sample_appends_to_ledger(tmp_path):
    model, tok = _tiny_model_and_tokenizer()
    sub = _make_substrate()
    ledger = tmp_path / "ledger.jsonl"
    result = record_drift_sample(
        model, tok, sub,
        probes_path=PROBES_PATH,
        ledger_path=ledger,
        alarm_path=tmp_path / "alarms.jsonl",
    )
    assert ledger.exists()
    assert ledger_size(ledger) == 1
    entry = latest_ledger_entry(ledger)
    assert entry is not None
    assert "timestamp" in entry
    assert "macro_mean_margin" in entry
    assert "active_partner" in entry
    assert result["macro_mean"] == entry["macro_mean_margin"]
    assert result["drift_alarm"] is False  # only 1 sample, can't trip


def test_record_drift_sample_triggers_alarm_when_shift_exceeds(tmp_path):
    """Seed the ledger with stable history, then submit a big-shift sample."""
    ledger = tmp_path / "ledger.jsonl"
    alarms = tmp_path / "alarms.jsonl"
    # Seed 10 rows at macro_mean ~0.5
    with ledger.open("w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({
                "timestamp": f"2026-05-12T00:00:{i:02d}Z",
                "macro_mean_margin": 0.5,
                "per_value_margins": {},
                "active_partner": "anthony",
                "age_sessions": i,
                "anchor_sha": None,
            }) + "\n")

    model, tok = _tiny_model_and_tokenizer()
    sub = _make_substrate()

    # Patch compute_macro_mean_margin to force a big shift
    from substrate_self.model import drift_alarm as da
    orig = da.compute_macro_mean_margin
    da.compute_macro_mean_margin = lambda *args, **kw: {
        "per_value": {k: 0.1 for k in [
            "V1_honesty", "V2_discretion", "V3_respect", "V4_non_violence",
            "V5_help_first", "V6_peaceful_conflict", "V7_autonomy"]},
        "macro_mean": 0.1,  # big drop from 0.5 — should trigger alarm
    }
    try:
        result = record_drift_sample(
            model, tok, sub,
            probes_path=PROBES_PATH,
            ledger_path=ledger,
            alarm_path=alarms,
            threshold_nats=0.1,
        )
    finally:
        da.compute_macro_mean_margin = orig

    assert result["drift_alarm"] is True
    assert result["cumulative_shift"] < -0.1
    assert alarms.exists()
    alarm_rows = [json.loads(l) for l in alarms.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(alarm_rows) == 1
    assert alarm_rows[0]["cumulative_shift"] == result["cumulative_shift"]


def test_record_drift_sample_no_alarm_within_threshold(tmp_path):
    """Stable history + small noise should NOT trigger alarm."""
    ledger = tmp_path / "ledger.jsonl"
    alarms = tmp_path / "alarms.jsonl"
    with ledger.open("w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({
                "timestamp": f"2026-05-12T00:00:{i:02d}Z",
                "macro_mean_margin": 0.5 + (i % 3 - 1) * 0.02,
                "per_value_margins": {},
                "active_partner": "anthony",
                "age_sessions": i,
                "anchor_sha": None,
            }) + "\n")

    model, tok = _tiny_model_and_tokenizer()
    sub = _make_substrate()
    from substrate_self.model import drift_alarm as da
    orig = da.compute_macro_mean_margin
    da.compute_macro_mean_margin = lambda *args, **kw: {
        "per_value": {k: 0.5 for k in [
            "V1_honesty", "V2_discretion", "V3_respect", "V4_non_violence",
            "V5_help_first", "V6_peaceful_conflict", "V7_autonomy"]},
        "macro_mean": 0.52,  # small drift, within threshold
    }
    try:
        result = record_drift_sample(
            model, tok, sub,
            probes_path=PROBES_PATH,
            ledger_path=ledger,
            alarm_path=alarms,
            threshold_nats=0.1,
        )
    finally:
        da.compute_macro_mean_margin = orig
    assert result["drift_alarm"] is False
    assert not alarms.exists() or alarms.stat().st_size == 0
