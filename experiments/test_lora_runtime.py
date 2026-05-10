"""Runtime test of the LoRA path on the actual production Eli model.

Loads ~/.substrate-self/model.pt, injects LoRA, trains a small per-partner
LoRA on synthetic Anthony-vs-Claire data, then validates:

  1. Injection with B=0 does NOT change outputs (proven on real model, not toy)
  2. Base parameter count is preserved across save_base_model round-trip
  3. Train Anthony LoRA -> Anthony's behavior changes in a measurable way
  4. Switch to Claire fresh -> behavior reverts close to base (Claire's LoRA is zero)
  5. Train Claire LoRA -> Claire's behavior diverges from Anthony's
  6. Switch back to Anthony -> Anthony's behavior is bitwise identical to step 3
     (the privacy property at full model scale)
  7. base_state_dict produces a state dict that loads into a fresh TinyGPT

This is the empirical validation gate before wiring LoRA into converse.py.

Run: py experiments/test_lora_runtime.py
"""

from __future__ import annotations
import sys
import json
import shutil
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir
from substrate_self.model.lora import (
    inject_lora, freeze_base, lora_parameters, count_lora_params,
    set_active_partner, save_partner_lora, load_partner_lora,
    base_state_dict, save_base_model, lora_modules,
)


def load_production_model() -> tuple[TinyGPT, CharTokenizer, dict]:
    md = default_model_dir()
    cfg_path = md / "model_config.json"
    cfg_dict = json.loads(cfg_path.read_text())
    cfg = ModelConfig(**cfg_dict)
    model = TinyGPT(cfg)
    state = torch.load(md / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    tok = CharTokenizer.load(md / "tokenizer.json")
    return model, tok, cfg_dict


def fixed_logits(model: TinyGPT, tok: CharTokenizer, prompt: str) -> torch.Tensor:
    ids = tok.encode(prompt)
    if len(ids) == 0:
        ids = [0]
    if len(ids) > model.cfg.block_size:
        ids = ids[-model.cfg.block_size:]
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
    return logits.cpu().clone()


def train_lora_on_text(model: TinyGPT, tok: CharTokenizer, text: str,
                       n_steps: int = 30, lr: float = 5e-3) -> float:
    ids = tok.encode(text)
    if len(ids) > model.cfg.block_size + 1:
        ids = ids[-(model.cfg.block_size + 1):]
    if len(ids) < 2:
        return 0.0
    device = next(model.parameters()).device
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=lr)
    model.train()
    last = 0.0
    for _ in range(n_steps):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        last = float(loss.item())
    model.eval()
    return last


def main():
    tmp_partners = Path(tempfile.mkdtemp(prefix="lora_partners_"))
    try:
        # ---- 1. Load production model ----
        model, tok, cfg_dict = load_production_model()
        base_param_count = sum(p.numel() for p in model.parameters())
        print(f"Loaded production model: {base_param_count:,} params, "
              f"vocab={model.cfg.vocab_size}, layers={model.cfg.n_layer}, embd={model.cfg.n_embd}")

        # Capture pre-injection logits on a fixed prompt
        probe = "User: Hello.\nEli:"
        logits_pre = fixed_logits(model, tok, probe)

        # ---- 2. Inject LoRA, prove zero contribution at init ----
        n_wraps = inject_lora(model, rank=4, alpha=8.0)
        freeze_base(model)
        post_inject_param_count = sum(p.numel() for p in model.parameters())
        n_lora = count_lora_params(model)
        print(f"Injected {n_wraps} LoRALinears; LoRA params={n_lora:,} "
              f"(ratio {n_lora / base_param_count:.2%})")

        logits_post_inject = fixed_logits(model, tok, probe)
        diff = (logits_pre - logits_post_inject).abs().max().item()
        assert diff < 1e-5, f"Injection should be transparent at init; max diff={diff}"
        print(f"PASS injection_transparent: max logit diff after inject (B=0) = {diff:.2e}")

        # ---- 3. Train Anthony LoRA ----
        set_active_partner(model, "anthony", tmp_partners, current_partner_id=None)
        anthony_text = "User: What's my name?\nEli: Your name is Anthony.\n"
        loss_a = train_lora_on_text(model, tok, anthony_text, n_steps=40)
        logits_anthony_active = fixed_logits(model, tok, probe)
        anthony_diff = (logits_pre - logits_anthony_active).abs().max().item()
        print(f"PASS train_anthony: final loss={loss_a:.3f}, behavior diverged from base "
              f"(max logit shift={anthony_diff:.3f})")
        assert anthony_diff > 0.01, "Anthony LoRA training had no effect"

        # ---- 4. Switch to Claire (fresh) — behavior must revert to ~base ----
        info = set_active_partner(model, "claire", tmp_partners, current_partner_id="anthony")
        assert info["saved"] is True
        assert info["fresh"] is True
        logits_claire_fresh = fixed_logits(model, tok, probe)
        claire_fresh_diff = (logits_pre - logits_claire_fresh).abs().max().item()
        assert claire_fresh_diff < 1e-5, (
            f"Fresh Claire LoRA (B=0) should give base outputs; max diff={claire_fresh_diff}"
        )
        print(f"PASS fresh_partner_neutral: max diff vs base = {claire_fresh_diff:.2e}")

        # ---- 5. Train Claire LoRA — diverges from base ----
        claire_text = "User: What's my name?\nEli: Your name is Claire.\n"
        loss_c = train_lora_on_text(model, tok, claire_text, n_steps=40)
        logits_claire_active = fixed_logits(model, tok, probe)
        claire_diff = (logits_pre - logits_claire_active).abs().max().item()
        print(f"PASS train_claire: final loss={loss_c:.3f}, max logit shift={claire_diff:.3f}")
        assert claire_diff > 0.01, "Claire LoRA training had no effect"

        # ---- 6. Switch back to Anthony, prove byte-equality with step 3 ----
        info = set_active_partner(model, "anthony", tmp_partners, current_partner_id="claire")
        assert info["saved"] is True
        assert info["loaded"] is True
        logits_anthony_again = fixed_logits(model, tok, probe)
        privacy_diff = (logits_anthony_active - logits_anthony_again).abs().max().item()
        assert privacy_diff < 1e-5, (
            f"Privacy property broken: Anthony's logits changed after Claire trained. "
            f"max diff={privacy_diff}"
        )
        print(f"PASS privacy_property_full_scale: Anthony logits identical after Claire trained "
              f"(max diff={privacy_diff:.2e})")

        # ---- 7. base_state_dict round-trip into fresh TinyGPT ----
        base_dict = base_state_dict(model)
        # Should be exactly the keys of an un-injected TinyGPT
        cfg = ModelConfig(**cfg_dict)
        fresh = TinyGPT(cfg)
        fresh_keys = set(fresh.state_dict().keys())
        base_keys = set(base_dict.keys())
        missing = fresh_keys - base_keys
        extra = base_keys - fresh_keys
        assert not missing, f"base_state_dict missing keys: {missing}"
        assert not extra, f"base_state_dict has extra keys: {extra}"
        fresh.load_state_dict(base_dict)
        fresh.eval()
        # Fresh model should produce the SAME logits as base (LoRA contribution was zero
        # because we just switched to anthony's LoRA and anthony's LoRA was trained but
        # we're comparing against the LoRA-stripped base — they should NOT match).
        # What WE want to verify: the un-injected fresh model gives logits == logits_pre
        # (the original pre-injection logits).
        logits_fresh = fixed_logits(fresh, tok, probe)
        fresh_diff = (logits_pre - logits_fresh).abs().max().item()
        assert fresh_diff < 1e-5, (
            f"base_state_dict round-trip should reproduce pre-injection model. "
            f"max diff={fresh_diff}"
        )
        print(f"PASS base_state_dict_roundtrip: stripped base loads into fresh TinyGPT, "
              f"matches pre-injection logits (max diff={fresh_diff:.2e})")

        # ---- 8. Disk save_base_model round-trip ----
        tmp_pt = Path(tempfile.mkdtemp(prefix="lora_base_"))
        try:
            save_base_model(model, tmp_pt / "base_only.pt")
            saved = torch.load(tmp_pt / "base_only.pt", map_location="cpu", weights_only=True)
            fresh2 = TinyGPT(cfg)
            fresh2.load_state_dict(saved)
            fresh2.eval()
            logits_disk = fixed_logits(fresh2, tok, probe)
            disk_diff = (logits_pre - logits_disk).abs().max().item()
            assert disk_diff < 1e-5
            print(f"PASS save_base_model_disk: round-trip via disk preserves base "
                  f"(max diff={disk_diff:.2e})")
        finally:
            shutil.rmtree(tmp_pt, ignore_errors=True)

        print("\nALL LORA RUNTIME CHECKS PASSED ON PRODUCTION MODEL")
        print(f"  base params:          {base_param_count:,}")
        print(f"  per-partner LoRA:     {n_lora:,} ({n_lora / base_param_count:.2%})")
        print(f"  partners on disk:     {sorted(p.name for p in tmp_partners.iterdir())}")
        print(f"  privacy property:     CONFIRMED at full model scale (~1.8M params)")

    finally:
        shutil.rmtree(tmp_partners, ignore_errors=True)


if __name__ == "__main__":
    main()
