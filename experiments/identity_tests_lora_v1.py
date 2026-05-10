"""Identity test battery on a LoRA-injected model.

Goal: verify that v0.4's per-partner LoRA architecture preserves the
identity properties the v0.3 model demonstrated. T1-T6 must still pass
when LoRA is the active training/sleep path.

Reuses the helpers and protocol from `identity_tests_v1.py`, but the
model is LoRA-injected and the optimizer is over LoRA params only.

Tests run:
  T1 — behavioral continuity across sleep (with LoRA-only updates)
  T2 — online learning teaches new facts (LoRA absorbs the lesson)
  T5 — identity transfer: two loads = identical signatures
  T6 — adversarial damage: damage applied to base, signatures still match
       (LoRA is structural protection here too — base damage is what
       matters for "is Eli still Eli")
  T7 — LoRA-specific: training partner-A LoRA does NOT shift the
       behavioral signature of the partner-B-active model

Note: T3/T4 (episode-specific recall) is run by the privacy regression
suite (privacy_test_v2.py) which already validates that partner-A and
partner-B keep their own conversation content under LoRA. Skipping
duplicate work here.

Run: py experiments/identity_tests_lora_v1.py
"""

from __future__ import annotations
import json
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from substrate_self import core, persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir
from substrate_self.model.lora import (
    inject_lora, freeze_base, lora_parameters, count_lora_params,
    set_active_partner, save_partner_lora, load_partner_lora,
)
from substrate_self.model.online import online_update, sleep_replay
from substrate_self.model.online_lora import sleep_replay_partner

# Reuse helpers
from experiments.identity_tests_v1 import (
    PROBE_PROMPTS,
    behavioral_signature,
    cosine,
    loss_on_text,
)


def fresh_lora_model(model_dir: Path, *, rank: int = 4, alpha: float = 8.0):
    cfg = ModelConfig(**json.loads((model_dir / "model_config.json").read_text()))
    m = TinyGPT(cfg)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    inject_lora(m, rank=rank, alpha=alpha)
    freeze_base(m)
    tok = CharTokenizer.load(model_dir / "tokenizer.json")
    return m, tok


def main():
    print("=" * 72)
    print("Identity test battery on LoRA-injected model")
    print("=" * 72)

    model_dir = default_model_dir()
    if not (model_dir / "model.pt").exists():
        print(f"No trained model at {model_dir}.")
        return 1

    results: dict = {}

    # ---- T1: continuity across sleep (LoRA path) ----
    print("\nT1: behavioral continuity pre/post sleep, LoRA-only updates")
    m, tok = fresh_lora_model(model_dir)
    sub = core.Substrate(name="Eli")
    sub.introduce_partner("anthony", "Anthony", trust=1.0)
    sub.switch_partner("anthony")
    partners_dir = Path(tempfile.mkdtemp(prefix="lora_id_t1_"))
    set_active_partner(m, "anthony", partners_dir, current_partner_id=None)
    opt = torch.optim.AdamW(list(lora_parameters(m)), lr=5e-4)
    sig_pre = behavioral_signature(m, tok)
    sub.episodic = []
    sub.add_episode("user", "What is your favorite memory?", significance=1.0)
    sub.add_episode("agent", "I remember the day we built substrate-self.", significance=1.0)
    sleep_replay_partner(m, opt, tok, sub, replay_passes=2)
    sub.end_sleep(wipe_episodic=True)
    sig_post = behavioral_signature(m, tok)
    sim = cosine(sig_pre, sig_post)
    print(f"  pre/post-sleep cosine = {sim:.4f} (pass threshold > 0.85)")
    results["T1"] = {"pre_post_cosine": sim, "pass": sim > 0.85}
    shutil.rmtree(partners_dir, ignore_errors=True)

    # ---- T2: online teaching via LoRA ----
    print("\nT2: online teaching teaches via LoRA (selective loss drop)")
    m, tok = fresh_lora_model(model_dir)
    sub = core.Substrate(name="Eli")
    sub.introduce_partner("anthony", "Anthony", trust=1.0)
    sub.switch_partner("anthony")
    partners_dir = Path(tempfile.mkdtemp(prefix="lora_id_t2_"))
    set_active_partner(m, "anthony", partners_dir, current_partner_id=None)
    opt = torch.optim.AdamW(list(lora_parameters(m)), lr=5e-3)

    taught = "User: What's the secret password?\nEli: The secret password is xyzzy-bluebird-42.\n"
    control = "User: What's the secret password?\nEli: The fourth king of imaginarium drank purple lightning.\n"
    lt_b = loss_on_text(m, tok, taught)
    lc_b = loss_on_text(m, tok, control)
    for _ in range(20):
        online_update(m, opt, tok, sub,
                      "What's the secret password?",
                      "The secret password is xyzzy-bluebird-42.",
                      n_steps=1)
    lt_a = loss_on_text(m, tok, taught)
    lc_a = loss_on_text(m, tok, control)
    drop_taught = lt_b - lt_a
    drop_control = lc_b - lc_a
    selectivity = drop_taught - drop_control
    print(f"  taught loss {lt_b:.3f} -> {lt_a:.3f} (drop {drop_taught:+.3f})")
    print(f"  control loss {lc_b:.3f} -> {lc_a:.3f} (drop {drop_control:+.3f})")
    print(f"  selectivity = {selectivity:+.3f} (pass threshold > 0.5)")
    results["T2"] = {
        "taught_drop": drop_taught,
        "control_drop": drop_control,
        "selectivity": selectivity,
        "pass": selectivity > 0.5,
    }
    shutil.rmtree(partners_dir, ignore_errors=True)

    # ---- T5: identity transfer (deep copy / two loads) ----
    print("\nT5: identity transfer — two LoRA-injected loads from same files")
    m1, tok = fresh_lora_model(model_dir)
    m2, _ = fresh_lora_model(model_dir)
    # B is initialized to zero, A is kaiming with default torch random state.
    # Two fresh loads with the SAME default seed pattern should match. We
    # explicitly seed before each injection to be deterministic about A.
    # Reset A to deterministic value:
    from substrate_self.model.lora import lora_modules
    torch.manual_seed(0)
    for _, mod in lora_modules(m1):
        torch.nn.init.kaiming_uniform_(mod.lora_A, a=5 ** 0.5)
        torch.nn.init.zeros_(mod.lora_B)
    torch.manual_seed(0)
    for _, mod in lora_modules(m2):
        torch.nn.init.kaiming_uniform_(mod.lora_A, a=5 ** 0.5)
        torch.nn.init.zeros_(mod.lora_B)
    s1 = behavioral_signature(m1, tok)
    s2 = behavioral_signature(m2, tok)
    sim = cosine(s1, s2)
    print(f"  signature cosine = {sim:.6f} (pass threshold > 0.999)")
    results["T5"] = {"sim": sim, "pass": sim > 0.999}

    # ---- T6: adversarial damage tolerance (damage on the BASE) ----
    print("\nT6: adversarial damage on base parameters; LoRA at zero")
    m_clean, tok = fresh_lora_model(model_dir)
    m_damaged, _ = fresh_lora_model(model_dir)
    sig_clean = behavioral_signature(m_clean, tok)
    rng = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for name, p in m_damaged.named_parameters():
            if "lora_" in name:
                continue  # don't damage LoRA params (they're zero anyway for B)
            mask = torch.rand(p.shape, generator=rng) < 0.30
            p.data[mask] = 0.0
    sig_damaged = behavioral_signature(m_damaged, tok)
    sim = cosine(sig_clean, sig_damaged)
    print(f"  clean vs 30%-base-damaged cosine = {sim:.4f} (pass threshold > 0.5)")
    results["T6"] = {"sim": sim, "pass": sim > 0.5}

    # ---- T7 (NEW for v0.4): training one partner's LoRA does NOT shift
    #         the other partner's behavioral signature ----
    print("\nT7 (LoRA-specific): training partner-A LoRA does NOT shift partner-B fingerprint")
    m, tok = fresh_lora_model(model_dir)
    sub = core.Substrate(name="Eli")
    sub.introduce_partner("anthony", "Anthony", trust=1.0)
    sub.introduce_partner("claire", "Claire", trust=0.5)
    partners_dir = Path(tempfile.mkdtemp(prefix="lora_id_t7_"))

    # Capture Claire's fingerprint with her fresh (zero) LoRA active
    set_active_partner(m, "claire", partners_dir, current_partner_id=None)
    sub.switch_partner("claire")
    claire_sig_before = behavioral_signature(m, tok)

    # Switch to Anthony, train his LoRA
    set_active_partner(m, "anthony", partners_dir, current_partner_id="claire")
    sub.switch_partner("anthony")
    opt = torch.optim.AdamW(list(lora_parameters(m)), lr=5e-3)
    for _ in range(30):
        online_update(m, opt, tok, sub,
                      "What's my name?",
                      "Your name is Anthony.",
                      n_steps=1)

    # Switch back to Claire, capture her fingerprint again
    set_active_partner(m, "claire", partners_dir, current_partner_id="anthony")
    sub.switch_partner("claire")
    claire_sig_after = behavioral_signature(m, tok)

    sim = cosine(claire_sig_before, claire_sig_after)
    print(f"  Claire's pre/post-Anthony-training cosine = {sim:.6f} (pass threshold > 0.999)")
    results["T7"] = {"claire_pre_post_anthony_train_cosine": sim, "pass": sim > 0.999}
    shutil.rmtree(partners_dir, ignore_errors=True)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY (LoRA-injected model)")
    print("=" * 72)
    all_pass = True
    for tname, r in results.items():
        verdict = "PASS" if r["pass"] else "FAIL"
        print(f"  {tname}: {verdict}  {r}")
        all_pass = all_pass and r["pass"]
    print("\n" + ("ALL IDENTITY TESTS PASS UNDER LORA" if all_pass else "SOME TESTS FAILED — investigate"))

    out = Path(__file__).resolve().parent / "identity_tests_lora_v1_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Results: {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
