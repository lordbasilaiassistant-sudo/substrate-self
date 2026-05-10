"""LoRA unit checks that don't depend on the v0.4 partner schema.

Tests the model-side mechanism in isolation:
  - inject_lora wraps the right Linears
  - LoRA contribution is initially zero (B init zero)
  - Only LoRA params are trainable after freeze_base
  - extract/apply round-trips
  - save/load to disk round-trips
  - set_active_partner saves current and loads new
  - Two partners' LoRAs do NOT interfere
  - Privacy property: partner B's training does NOT change partner A's outputs

Run: py experiments/test_lora_unit.py
"""

from __future__ import annotations
import sys
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.lora import (
    LoRALinear,
    inject_lora,
    lora_modules,
    lora_parameters,
    freeze_base,
    extract_lora_state,
    apply_lora_state,
    save_partner_lora,
    load_partner_lora,
    set_active_partner,
    count_lora_params,
)


def build_model() -> TinyGPT:
    cfg = ModelConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    torch.manual_seed(0)
    return TinyGPT(cfg)


def assert_close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6, msg: str = "") -> None:
    diff = (a - b).abs().max().item()
    assert diff < tol, f"{msg} max abs diff {diff} >= {tol}"


def test_injection_count():
    m = build_model()
    n = inject_lora(m, rank=4, alpha=8.0)
    expected = 2 * 2  # 2 layers × (c_attn + c_proj)
    assert n == expected, f"expected {expected} LoRA wraps, got {n}"
    mods = lora_modules(m)
    assert len(mods) == expected
    print(f"PASS injection_count: {n} LoRALinears injected")


def test_initial_zero_contribution():
    m = build_model()
    cfg = m.cfg
    inp = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits_before, _ = m(inp)
    inject_lora(m, rank=4, alpha=8.0)
    with torch.no_grad():
        logits_after, _ = m(inp)
    assert_close(logits_before, logits_after, tol=1e-5,
                 msg="LoRA injection with B=0 must not change outputs")
    print("PASS initial_zero_contribution: B init zero -> identical logits before/after injection")


def test_freeze_base_only_lora_trainable():
    m = build_model()
    inject_lora(m, rank=4, alpha=8.0)
    freeze_base(m)
    n_train = sum(1 for p in m.parameters() if p.requires_grad)
    n_lora = sum(1 for _ in lora_parameters(m))
    assert n_train == n_lora, f"trainable={n_train}, lora_params={n_lora} — must match"
    print(f"PASS freeze_base: only the {n_lora} LoRA params are trainable")


def test_lora_actually_trains():
    """One step of training on a target should reduce loss."""
    m = build_model()
    inject_lora(m, rank=4, alpha=8.0)
    freeze_base(m)
    opt = torch.optim.Adam(list(lora_parameters(m)), lr=1e-2)

    cfg = m.cfg
    torch.manual_seed(42)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = torch.randint(0, cfg.vocab_size, (1, 8))

    m.train()
    _, loss0 = m(x, y)
    loss0_val = loss0.item()
    for _ in range(50):
        opt.zero_grad()
        _, loss = m(x, y)
        loss.backward()
        opt.step()
    _, loss_final = m(x, y)
    loss_f = loss_final.item()
    assert loss_f < loss0_val, f"loss did not decrease: {loss0_val:.3f} -> {loss_f:.3f}"
    print(f"PASS lora_actually_trains: loss {loss0_val:.3f} -> {loss_f:.3f} via LoRA-only updates")


def test_extract_apply_roundtrip():
    m = build_model()
    inject_lora(m, rank=4, alpha=8.0)
    freeze_base(m)
    # Train a bit so LoRA is nonzero
    opt = torch.optim.Adam(list(lora_parameters(m)), lr=1e-2)
    x = torch.randint(0, m.cfg.vocab_size, (1, 8))
    y = torch.randint(0, m.cfg.vocab_size, (1, 8))
    m.train()
    for _ in range(20):
        opt.zero_grad()
        _, loss = m(x, y)
        loss.backward()
        opt.step()

    state = extract_lora_state(m)
    # Capture logits, then perturb LoRA, then restore and compare
    m.eval()
    with torch.no_grad():
        logits_a, _ = m(x)
    for _, mod in lora_modules(m):
        mod.reset_lora()
    apply_lora_state(m, state)
    with torch.no_grad():
        logits_b, _ = m(x)
    assert_close(logits_a, logits_b, tol=1e-6, msg="extract/apply round-trip mismatch")
    print("PASS extract_apply_roundtrip: LoRA state survives reset+apply")


def test_save_load_disk_roundtrip(tmp: Path):
    m = build_model()
    inject_lora(m, rank=4, alpha=8.0)
    freeze_base(m)
    opt = torch.optim.Adam(list(lora_parameters(m)), lr=1e-2)
    x = torch.randint(0, m.cfg.vocab_size, (1, 8))
    y = torch.randint(0, m.cfg.vocab_size, (1, 8))
    m.train()
    for _ in range(20):
        opt.zero_grad()
        _, loss = m(x, y)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        logits_a, _ = m(x)

    save_partner_lora(m, "anthony", tmp)
    for _, mod in lora_modules(m):
        mod.reset_lora()
    found = load_partner_lora(m, "anthony", tmp)
    assert found is True
    with torch.no_grad():
        logits_b, _ = m(x)
    assert_close(logits_a, logits_b, tol=1e-6, msg="disk save/load round-trip mismatch")
    print("PASS save_load_disk_roundtrip: partner LoRA survives disk round-trip")


def test_two_partners_isolated(tmp: Path):
    """Privacy property: partner B training does NOT change partner A outputs."""
    m = build_model()
    inject_lora(m, rank=4, alpha=8.0)
    freeze_base(m)

    cfg = m.cfg
    torch.manual_seed(1)
    x_eval = torch.randint(0, cfg.vocab_size, (1, 8))

    torch.manual_seed(11)
    x_a = torch.randint(0, cfg.vocab_size, (1, 8))
    y_a = torch.randint(0, cfg.vocab_size, (1, 8))
    torch.manual_seed(22)
    x_b = torch.randint(0, cfg.vocab_size, (1, 8))
    y_b = torch.randint(0, cfg.vocab_size, (1, 8))

    # ---- Train partner A ----
    set_active_partner(m, "anthony", tmp, current_partner_id=None)
    opt = torch.optim.Adam(list(lora_parameters(m)), lr=1e-2)
    m.train()
    for _ in range(40):
        opt.zero_grad()
        _, loss = m(x_a, y_a)
        loss.backward()
        opt.step()
    # Capture A's behavior on the shared eval input
    m.eval()
    with torch.no_grad():
        logits_a_when_a_active, _ = m(x_eval)

    # ---- Switch to partner B (saves A, loads B fresh) ----
    info = set_active_partner(m, "claire", tmp, current_partner_id="anthony")
    assert info["saved"] is True
    assert info["fresh"] is True

    opt = torch.optim.Adam(list(lora_parameters(m)), lr=1e-2)
    m.train()
    for _ in range(40):
        opt.zero_grad()
        _, loss = m(x_b, y_b)
        loss.backward()
        opt.step()

    # ---- Switch back to A (saves B, loads A) ----
    info = set_active_partner(m, "anthony", tmp, current_partner_id="claire")
    assert info["saved"] is True
    assert info["loaded"] is True

    m.eval()
    with torch.no_grad():
        logits_a_after_b_trained, _ = m(x_eval)

    assert_close(
        logits_a_when_a_active,
        logits_a_after_b_trained,
        tol=1e-5,
        msg="partner-A outputs changed after partner-B training — privacy property broken",
    )
    print("PASS two_partners_isolated: partner-A logits identical before/after partner-B training")


def test_param_count_summary():
    m = build_model()
    base_params = sum(p.numel() for p in m.parameters())
    inject_lora(m, rank=4, alpha=8.0)
    freeze_base(m)
    lora_p = count_lora_params(m)
    print(f"INFO  base={base_params:,} params, per-partner LoRA={lora_p:,} params "
          f"(ratio {lora_p / base_params:.4%})")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="lora_test_"))
    try:
        test_injection_count()
        test_initial_zero_contribution()
        test_freeze_base_only_lora_trainable()
        test_lora_actually_trains()
        test_extract_apply_roundtrip()
        test_save_load_disk_roundtrip(tmp)
        test_two_partners_isolated(tmp)
        test_param_count_summary()
        print("\nALL LORA UNIT CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
