"""End-to-end LoRA path test through the same code converse.run_solo uses.

Validates the full v0.4 wake -> talk -> sleep -> reload cycle:

  1. Copy production model.pt + tokenizer to a temp dir
  2. Build a fresh v0.4 substrate with anthony + claire partners
  3. Simulate session 1 (anthony active): online_update on a few turns,
     sleep_replay_partner, save_partner_lora(anthony) + save_base_model
  4. Reload from disk (this is what next-session converse does) and
     prove anthony's logits match what we had at end of session 1
  5. Switch to claire, simulate session 2, sleep, save
  6. Reload, switch back to anthony, prove anthony's logits STILL match
     end of session 1 (privacy + no-forgetting at the file-roundtrip level)

This is the empirical gate before declaring v0.4 done.

Run: py experiments/test_converse_lora_e2e.py
"""

from __future__ import annotations
import sys
import json
import shutil
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self.core import Substrate
from substrate_self import persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir
from substrate_self.model.lora import (
    inject_lora, freeze_base, lora_parameters, count_lora_params,
    load_partner_lora, save_partner_lora, save_base_model,
)
from substrate_self.model.online import online_update
from substrate_self.model.online_lora import sleep_replay_partner


def fixed_logits(model: TinyGPT, tok: CharTokenizer, prompt: str) -> torch.Tensor:
    ids = tok.encode(prompt)[:model.cfg.block_size]
    if not ids:
        ids = [0]
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
    return logits.cpu().clone()


def boot_session(model_dir: Path, substrate_path: Path):
    """Mimic converse.run_solo's load + inject + active-LoRA load."""
    cfg = ModelConfig(**json.loads((model_dir / "model_config.json").read_text()))
    model = TinyGPT(cfg)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    tok = CharTokenizer.load(model_dir / "tokenizer.json")
    s = Substrate.model_validate_json(substrate_path.read_text())

    inject_lora(model, rank=4, alpha=8.0)
    freeze_base(model)
    if s.active_partner_id is not None:
        load_partner_lora(model, s.active_partner_id, model_dir / "partners")
    optimizer = torch.optim.AdamW(list(lora_parameters(model)), lr=5e-3)
    return model, tok, s, optimizer


def simulate_turns(model, optimizer, tok, s, turns):
    """Run online_update on (user, agent) pairs and append to episodic."""
    for user_text, agent_text in turns:
        s.add_episode("user", user_text, significance=1.0)
        s.add_episode("agent", agent_text, significance=1.0)
        online_update(model, optimizer, tok, s, user_text, agent_text, n_steps=3)


def end_session(model, optimizer, tok, s, model_dir, substrate_path):
    """Sleep + save (matches converse sleep block, LoRA branch)."""
    metrics = sleep_replay_partner(model, optimizer, tok, s, replay_passes=2)
    s.end_sleep(wipe_episodic=True)
    if s.active_partner_id is not None:
        save_partner_lora(model, s.active_partner_id, model_dir / "partners")
    save_base_model(model, model_dir / "model.pt")
    substrate_path.write_text(s.model_dump_json(indent=2))
    return metrics


def main():
    src = default_model_dir()
    workdir = Path(tempfile.mkdtemp(prefix="lora_e2e_"))
    try:
        # ---- Setup: copy production assets to a sandbox ----
        for fname in ("model.pt", "model_config.json", "tokenizer.json"):
            shutil.copy(src / fname, workdir / fname)
        sub_path = workdir / "substrate.json"
        s0 = Substrate(name="Eli")
        s0.introduce_partner("anthony", "Anthony", trust=1.0)
        s0.introduce_partner("claire", "Claire", trust=0.5)
        s0.switch_partner("anthony")
        sub_path.write_text(s0.model_dump_json(indent=2))
        print(f"sandbox at {workdir}")
        print(f"partners introduced: anthony (trust=1.0), claire (trust=0.5); active=anthony")

        # ---- Snapshot pre-LoRA logits for a probe ----
        cfg = ModelConfig(**json.loads((workdir / "model_config.json").read_text()))
        base = TinyGPT(cfg)
        base.load_state_dict(torch.load(workdir / "model.pt", map_location="cpu", weights_only=True))
        base.eval()
        tok = CharTokenizer.load(workdir / "tokenizer.json")
        probe = "User: Tell me about myself.\nEli:"
        logits_base_pristine = fixed_logits(base, tok, probe)

        # ---- Session 1: Anthony active ----
        m1, tok, s, opt = boot_session(workdir, sub_path)
        anthony_turns = [
            ("My name is Anthony.", "Got it, Anthony."),
            ("I'm 33.", "Noted, you're 33."),
            ("I built substrate-self.", "Right, you built substrate-self."),
        ]
        simulate_turns(m1, opt, tok, s, anthony_turns)
        end_session(m1, opt, tok, s, workdir, sub_path)
        logits_anthony_end_s1 = fixed_logits(m1, tok, probe)
        anthony_shift = (logits_base_pristine - logits_anthony_end_s1).abs().max().item()
        print(f"PASS session1_anthony: trained {len(anthony_turns)} turns, "
              f"sleep+save complete, max logit shift from base = {anthony_shift:.3f}")
        assert anthony_shift > 0.01, "Anthony LoRA had no measurable effect after session 1"
        assert (workdir / "partners" / "anthony.lora").exists(), "anthony.lora not saved"

        # ---- Reload (next-session boot) — anthony's logits must match end of S1 ----
        m2, tok, s, opt = boot_session(workdir, sub_path)
        logits_after_reload = fixed_logits(m2, tok, probe)
        reload_diff = (logits_anthony_end_s1 - logits_after_reload).abs().max().item()
        assert reload_diff < 1e-5, f"Cross-session reload changed anthony's logits: {reload_diff}"
        print(f"PASS reload_anthony: post-reload logits match end-of-session-1 "
              f"(max diff={reload_diff:.2e})")

        # ---- Session 2: switch to Claire, train her LoRA ----
        # Save anthony's lora first via the switch helper, then load claire fresh
        from substrate_self.model.lora import set_active_partner
        info = set_active_partner(m2, "claire", workdir / "partners",
                                  current_partner_id=s.active_partner_id)
        s.switch_partner("claire")
        sub_path.write_text(s.model_dump_json(indent=2))
        # Reset optimizer for fresh LoRA params
        opt = torch.optim.AdamW(list(lora_parameters(m2)), lr=5e-3)
        claire_turns = [
            ("My name is Claire.", "Got it, Claire."),
            ("I'm a researcher.", "Noted, you're a researcher."),
            ("I work on neuroscience.", "Right, neuroscience."),
        ]
        simulate_turns(m2, opt, tok, s, claire_turns)
        end_session(m2, opt, tok, s, workdir, sub_path)
        logits_claire_end_s2 = fixed_logits(m2, tok, probe)
        claire_shift = (logits_base_pristine - logits_claire_end_s2).abs().max().item()
        print(f"PASS session2_claire: trained {len(claire_turns)} turns, "
              f"max logit shift from base = {claire_shift:.3f}")
        assert claire_shift > 0.01, "Claire LoRA had no effect"
        assert (workdir / "partners" / "claire.lora").exists(), "claire.lora not saved"

        # ---- Session 3 reload, switch BACK to Anthony; logits must match end of S1 ----
        m3, tok, s, opt = boot_session(workdir, sub_path)
        # boot_session loaded claire (active in substrate after S2)
        info = set_active_partner(m3, "anthony", workdir / "partners",
                                  current_partner_id=s.active_partner_id)
        s.switch_partner("anthony")
        sub_path.write_text(s.model_dump_json(indent=2))
        logits_anthony_after_claire = fixed_logits(m3, tok, probe)
        privacy_diff = (logits_anthony_end_s1 - logits_anthony_after_claire).abs().max().item()
        assert privacy_diff < 1e-5, (
            f"Privacy/forgetting property broken after disk roundtrip: "
            f"max diff={privacy_diff}"
        )
        print(f"PASS privacy_full_cycle: anthony's logits identical after a full "
              f"sleep/save/load/claire-train/reload/switch-back cycle "
              f"(max diff={privacy_diff:.2e})")

        # ---- Bonus: verify base.pt is base-only (no LoRA keys) ----
        saved_base = torch.load(workdir / "model.pt", map_location="cpu", weights_only=True)
        offenders = [k for k in saved_base.keys() if "lora_" in k]
        assert not offenders, f"saved base.pt leaked LoRA keys: {offenders}"
        print(f"PASS base_pt_clean: no LoRA keys in saved model.pt ({len(saved_base)} keys total)")

        n_lora = count_lora_params(m3)
        base_n = sum(v.numel() for v in saved_base.values())
        print(f"\nALL E2E LORA CHECKS PASSED")
        print(f"  base model.pt:        {base_n:,} params")
        print(f"  per-partner LoRA:     {n_lora:,} params (~{n_lora / base_n:.2%} overhead)")
        print(f"  partner LoRA files:   {sorted(p.name for p in (workdir / 'partners').iterdir())}")
        print(f"  privacy property:     CONFIRMED across full disk roundtrip")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
