"""Smoke test for SubstrateLM v0.5 starter.

Confirms the architecture builds, forward produces correct shapes, loss
is finite, generate runs, and the SDR gate + Hebbian fast-weight memory
behave as designed. NOT a performance test — that's a separate
benchmark against TinyGPT on the corpus.

Run: py experiments/test_substrate_lm_smoke.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from substrate_self.model.substrate_lm import (
    SubstrateLMConfig, SubstrateLM, LinearAttentionHebbian, SDRGate,
)
from substrate_self.model.transformer import ModelConfig as TinyGPTConfig, TinyGPT


def test_build_and_shapes():
    cfg = SubstrateLMConfig(vocab_size=64, block_size=32, n_layer=2,
                            n_head=2, n_embd=64, dropout=0.0, topk_active=20)
    m = SubstrateLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(x, y)
    assert logits.shape == (2, 16, cfg.vocab_size), f"got {logits.shape}"
    assert loss is not None and torch.isfinite(loss), f"non-finite loss: {loss}"
    print(f"PASS build_and_shapes: logits {logits.shape}, loss={loss.item():.3f}")
    print(f"  params: {m.num_params():,}")


def test_generate_runs():
    cfg = SubstrateLMConfig(vocab_size=64, block_size=32, n_layer=2,
                            n_head=2, n_embd=64, dropout=0.0)
    m = SubstrateLM(cfg)
    m.eval()
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = m.generate(x, max_new_tokens=8, temperature=1.0, top_k=10)
    assert out.shape == (1, 12), f"got {out.shape}"
    print(f"PASS generate_runs: input(4) -> output({out.shape[1]}), "
          f"tokens={out[0].tolist()}")


def test_sdr_actually_sparsifies():
    cfg = SubstrateLMConfig(vocab_size=32, block_size=8, n_layer=1,
                            n_head=1, n_embd=16, dropout=0.0, topk_active=4)
    gate = SDRGate(cfg)
    x = torch.randn(2, 4, 16)
    y = gate(x)
    nonzero_per_token = (y != 0).sum(dim=-1)
    assert (nonzero_per_token == 4).all(), \
        f"expected exactly 4 active per token, got {nonzero_per_token.tolist()}"
    print(f"PASS sdr_sparsifies: K={cfg.topk_active}, observed active counts {nonzero_per_token[0].tolist()}")


def test_hebbian_state_evolves():
    """When persist_fast=True, M_persist accumulates across forwards.

    Note: the default vectorized `forward` computes via the parallel
    kernel formulation and does NOT touch M_persist (the kernel form
    re-derives the same y_t without explicitly materializing the
    cumulative M). The recurrent `forward_recurrent` IS the path
    that uses M_persist. We test the recurrent path here for the
    persist semantics; the parallel form's correctness is covered
    by build_and_shapes + same_interface_as_tinygpt + trainable_via_gradient.
    """
    cfg = SubstrateLMConfig(vocab_size=32, block_size=8, n_layer=1,
                            n_head=1, n_embd=16, dropout=0.0, persist_fast=True)
    m = SubstrateLM(cfg)
    m.eval()
    x1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attn = m.blocks[0].attn
    with torch.no_grad():
        # Drive the recurrent path explicitly to exercise persist_fast.
        # We can't use m(x1) because that calls SubstrateBlock.forward which
        # uses the vectorized attn.forward. Call attn.forward_recurrent on
        # an embedded x:
        pos = torch.arange(0, x1.size(1)).unsqueeze(0)
        h = m.drop(m.tok_emb(x1) + m.pos_emb(pos))
        _ = attn.forward_recurrent(m.blocks[0].ln_1(h))
    norm_after_1 = float(attn.M_persist.norm().item())
    assert norm_after_1 > 0, "M_persist should have accumulated content after recurrent forward"
    with torch.no_grad():
        x2 = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        h2 = m.drop(m.tok_emb(x2) + m.pos_emb(pos))
        _ = attn.forward_recurrent(m.blocks[0].ln_1(h2))
    norm_after_2 = float(attn.M_persist.norm().item())
    assert norm_after_2 > 0
    m.reset_fast()
    norm_after_reset = float(attn.M_persist.norm().item()) if attn.M_persist.numel() > 0 else 0.0
    assert norm_after_reset == 0.0, f"reset_fast should wipe M (got norm {norm_after_reset})"
    print(f"PASS hebbian_state_evolves (recurrent path): "
          f"M norm 1st fwd={norm_after_1:.3f}, 2nd={norm_after_2:.3f}, reset={norm_after_reset:.3f}")


def test_vectorized_matches_recurrent():
    """The parallel kernel form (forward) and the recurrent form (forward_recurrent)
    should produce identical outputs for the same input (within numerical tol)."""
    cfg = SubstrateLMConfig(vocab_size=32, block_size=8, n_layer=1,
                            n_head=1, n_embd=16, dropout=0.0)
    m = SubstrateLM(cfg)
    m.eval()
    torch.manual_seed(0)
    # Build an input embedding once
    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    pos = torch.arange(0, x.size(1)).unsqueeze(0)
    h = m.drop(m.tok_emb(x) + m.pos_emb(pos))
    h_normed = m.blocks[0].ln_1(h)
    attn = m.blocks[0].attn
    with torch.no_grad():
        y_par = attn.forward(h_normed)
        y_rec = attn.forward_recurrent(h_normed)
    diff = (y_par - y_rec).abs().max().item()
    assert diff < 1e-4, f"parallel vs recurrent diverge: max diff {diff}"
    print(f"PASS vectorized_matches_recurrent: max diff {diff:.2e}")


def test_trainable_via_gradient():
    """SubstrateLM is gradient-trainable end-to-end (slow weights take the gradient)."""
    cfg = SubstrateLMConfig(vocab_size=32, block_size=16, n_layer=2,
                            n_head=2, n_embd=32, dropout=0.0, topk_active=16)
    m = SubstrateLM(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = torch.randint(0, cfg.vocab_size, (1, 8))
    m.train()
    _, l0 = m(x, y)
    initial = l0.item()
    for _ in range(30):
        opt.zero_grad()
        _, loss = m(x, y)
        loss.backward()
        opt.step()
    m.eval()
    _, lf = m(x, y)
    final = lf.item()
    assert final < initial, f"loss did not decrease: {initial:.3f} -> {final:.3f}"
    print(f"PASS trainable_via_gradient: loss {initial:.3f} -> {final:.3f}")


def test_same_interface_as_tinygpt():
    """Drop-in compatibility check: same forward/generate API shape."""
    sub_cfg = SubstrateLMConfig(vocab_size=32, block_size=8, n_layer=1,
                                n_head=1, n_embd=16, dropout=0.0)
    sub = SubstrateLM(sub_cfg)
    tg_cfg = TinyGPTConfig(vocab_size=32, block_size=8, n_layer=1,
                           n_head=1, n_embd=16, dropout=0.0)
    tg = TinyGPT(tg_cfg)

    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    sub_logits, sub_loss = sub(x, y)
    tg_logits, tg_loss = tg(x, y)
    assert sub_logits.shape == tg_logits.shape
    assert sub_loss.shape == tg_loss.shape

    sub_gen = sub.generate(x, max_new_tokens=4, temperature=1.0, top_k=8)
    tg_gen = tg.generate(x, max_new_tokens=4, temperature=1.0, top_k=8)
    assert sub_gen.shape == tg_gen.shape
    print(f"PASS same_interface_as_tinygpt: forward + generate shapes match TinyGPT")


def test_param_count_comparable_to_tinygpt():
    sub_cfg = SubstrateLMConfig(vocab_size=128, block_size=128, n_layer=4,
                                n_head=4, n_embd=192, dropout=0.1)
    sub = SubstrateLM(sub_cfg)
    tg_cfg = TinyGPTConfig(vocab_size=128, block_size=128, n_layer=4,
                           n_head=4, n_embd=192, dropout=0.1)
    tg = TinyGPT(tg_cfg)
    sub_p = sub.num_params()
    tg_p = tg.num_params()
    print(f"INFO  param counts: SubstrateLM={sub_p:,}, TinyGPT={tg_p:,}, "
          f"ratio={sub_p / tg_p:.3f}")


def main():
    test_build_and_shapes()
    test_generate_runs()
    test_sdr_actually_sparsifies()
    test_hebbian_state_evolves()
    test_vectorized_matches_recurrent()
    test_trainable_via_gradient()
    test_same_interface_as_tinygpt()
    test_param_count_comparable_to_tinygpt()
    print("\nALL SUBSTRATELM SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
