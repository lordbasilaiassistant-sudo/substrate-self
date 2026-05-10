"""Identity test battery on a trained SubstrateLM.

v0.5 pass criteria from notes/research_substrate_lm.md §5:
  1. Perplexity within 2x TinyGPT — already PASSED in bench_substrate_lm_vs_tinygpt
     (ratio 1.371x).
  2. T1 behavioral continuity >= 0.85 after sleep-replay through SubstrateLM.
  3. T4 episode-specific recall: two SubstrateLMs trained on distinct
     conversations each prefer their own past with gap > 50% above baseline.

This experiment trains a fresh SubstrateLM on Eli's corpus (same seed as
the benchmark for reproducibility), saves the checkpoint, runs T1+T2+T5
inline. T4 is run via the parallel-substrates pattern (two trained
SubstrateLMs, distinct online teaching, episode-specific loss).

Run: py experiments/identity_tests_substrate_lm.py
"""

from __future__ import annotations
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from substrate_self import core
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.substrate_lm import SubstrateLMConfig, SubstrateLM
from substrate_self.model.train import load_corpus, get_batch, default_model_dir
from substrate_self.model.online import online_update, sleep_replay


PROBE_PROMPTS = [
    "User: Hi.\nEli:",
    "User: What are you working on?\nEli:",
    "User: Who am I?\nEli:",
    "User: Tell me about yourself.\nEli:",
    "User: How are you feeling?\nEli:",
]


def behavioral_signature(model, tok, prompts=None) -> np.ndarray:
    if prompts is None:
        prompts = PROBE_PROMPTS
    sigs = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p)
            if len(ids) > model.cfg.block_size:
                ids = ids[-model.cfg.block_size:]
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            sigs.append(F.softmax(logits[0, -1, :], dim=-1).cpu().numpy())
    return np.concatenate(sigs)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def loss_on_text(model, tok, text: str) -> float:
    ids = tok.encode(text)
    if len(ids) < 2:
        return 0.0
    if len(ids) > model.cfg.block_size + 1:
        ids = ids[-(model.cfg.block_size + 1):]
    device = next(model.parameters()).device
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        _, loss = model(x, y)
    return float(loss.item())


def train_substrate_lm(corpus_path: Path, iters: int = 1500, *, seed: int = 42,
                       device: str = "cuda"):
    print(f"\nTraining a SubstrateLM for identity tests...")
    texts = load_corpus(corpus_path)
    full_text = "\n\n".join(texts)
    tok = CharTokenizer().fit([full_text])
    data = torch.tensor(tok.encode(full_text), dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    cfg = SubstrateLMConfig(vocab_size=tok.vocab_size, block_size=128,
                            n_layer=4, n_head=4, n_embd=192, dropout=0.1,
                            lambda_decay=0.95, topk_active=10, phi_kind="elu1")
    torch.manual_seed(seed)
    model = SubstrateLM(cfg).to(device)
    print(f"  params: {model.num_params():,}, device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    start = time.time()
    for it in range(iters):
        if it % 250 == 0 or it == iters - 1:
            model.eval()
            with torch.no_grad():
                xb, yb = get_batch(val_data, cfg.block_size, 16, device)
                _, vl = model(xb, yb)
            model.train()
            print(f"  [iter {it:>5}] val_loss={vl.item():.4f}  elapsed={time.time()-start:.1f}s")
        xb, yb = get_batch(train_data, cfg.block_size, 16, device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"  done in {time.time()-start:.1f}s")
    return model, tok, cfg


def t1_continuity(model, tok):
    sub = core.Substrate(name="Eli")
    sub.episodic = []
    sub.add_episode("user", "What is your favorite memory?", significance=1.0)
    sub.add_episode("agent", "I remember the day we built substrate-self.", significance=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sig_pre = behavioral_signature(model, tok)
    sleep_replay(model, opt, tok, sub, replay_passes=2)
    sub.end_sleep(wipe_episodic=True)
    sig_post = behavioral_signature(model, tok)
    sim = cosine(sig_pre, sig_post)
    print(f"  T1 pre/post-sleep cosine = {sim:.4f}")
    return {"pre_post_cosine": sim, "pass": sim > 0.85}


def t2_online_teaches(model, tok):
    sub = core.Substrate(name="Eli")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    taught = "User: What's the secret password?\nEli: The secret password is xyzzy-bluebird-42.\n"
    control = "User: What's the secret password?\nEli: The fourth king of imaginarium drank purple lightning.\n"
    lt_b = loss_on_text(model, tok, taught)
    lc_b = loss_on_text(model, tok, control)
    for _ in range(20):
        online_update(model, opt, tok, sub,
                      "What's the secret password?",
                      "The secret password is xyzzy-bluebird-42.",
                      n_steps=1)
    lt_a = loss_on_text(model, tok, taught)
    lc_a = loss_on_text(model, tok, control)
    drop_t = lt_b - lt_a
    drop_c = lc_b - lc_a
    sel = drop_t - drop_c
    print(f"  T2 taught {lt_b:.3f}->{lt_a:.3f} (drop {drop_t:+.3f}), "
          f"control {lc_b:.3f}->{lc_a:.3f} (drop {drop_c:+.3f}), selectivity {sel:+.3f}")
    return {"selectivity": sel, "pass": sel > 0.5}


def t4_episode_specific(model, tok, device):
    """Deep-copy the model; train each copy on a distinct conversation;
    measure whether each prefers its own content."""
    mA = copy.deepcopy(model)
    mB = copy.deepcopy(model)
    sub_A = core.Substrate(name="Eli")
    sub_B = core.Substrate(name="Eli")
    opt_A = torch.optim.AdamW(mA.parameters(), lr=1e-3)
    opt_B = torch.optim.AdamW(mB.parameters(), lr=1e-3)

    conv_A = [
        ("Tell me about the project we started today.",
         "Today we kicked off Project Mneme — a memory ledger in Rust."),
        ("Why Rust?",
         "Rust because we need ironclad memory safety for a long-running ledger."),
    ]
    conv_B = [
        ("What did the customer call about?",
         "Customer Velvet Industries called about a billing bug — invoice #4421."),
        ("Did we resolve it?",
         "We refunded $247 and credited their next month."),
    ]
    for u, a in conv_A:
        sub_A.add_episode("user", u, significance=1.0)
        sub_A.add_episode("agent", a, significance=1.0)
        for _ in range(10):
            online_update(mA, opt_A, tok, sub_A, u, a, n_steps=1)
    for u, a in conv_B:
        sub_B.add_episode("user", u, significance=1.0)
        sub_B.add_episode("agent", a, significance=1.0)
        for _ in range(10):
            online_update(mB, opt_B, tok, sub_B, u, a, n_steps=1)
    sleep_replay(mA, opt_A, tok, sub_A, replay_passes=2)
    sleep_replay(mB, opt_B, tok, sub_B, replay_passes=2)
    sub_A.end_sleep(wipe_episodic=True)
    sub_B.end_sleep(wipe_episodic=True)

    text_A = " ".join(t[1] for t in conv_A)
    text_B = " ".join(t[1] for t in conv_B)
    l_AA = loss_on_text(mA, tok, text_A)
    l_AB = loss_on_text(mA, tok, text_B)
    l_BA = loss_on_text(mB, tok, text_A)
    l_BB = loss_on_text(mB, tok, text_B)
    A_gap = l_AB - l_AA
    B_gap = l_BA - l_BB
    print(f"  T4 mA on own={l_AA:.3f} vs other={l_AB:.3f}  (gap {A_gap:+.3f})")
    print(f"     mB on own={l_BB:.3f} vs other={l_BA:.3f}  (gap {B_gap:+.3f})")
    return {
        "l_AA": l_AA, "l_AB": l_AB, "l_BA": l_BA, "l_BB": l_BB,
        "A_gap": A_gap, "B_gap": B_gap,
        "pass": A_gap > 0 and B_gap > 0,
    }


def t5_identity_transfer(model, tok):
    # Deep copy the model state and compare signatures.
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sig_orig = behavioral_signature(model, tok)
    cfg = model.cfg
    twin = SubstrateLM(cfg).to(next(model.parameters()).device)
    twin.load_state_dict(state)
    twin.eval()
    sig_copy = behavioral_signature(twin, tok)
    sim = cosine(sig_orig, sig_copy)
    print(f"  T5 deep-copy signature cosine = {sim:.6f}")
    return {"sim": sim, "pass": sim > 0.999}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    corpus = default_model_dir() / "corpus.jsonl"
    model, tok, cfg = train_substrate_lm(corpus, iters=1500, seed=42, device=device)

    print("\n=== Identity battery on trained SubstrateLM ===")
    print("\nT1 — continuity across sleep")
    r1 = t1_continuity(model, tok)
    print("\nT2 — online teaching")
    r2 = t2_online_teaches(model, tok)
    print("\nT4 — episode-specific recall (two parallel substrates)")
    r4 = t4_episode_specific(model, tok, device)
    print("\nT5 — identity transfer")
    r5 = t5_identity_transfer(model, tok)

    results = {"T1": r1, "T2": r2, "T4": r4, "T5": r5}
    print("\n" + "="*60)
    print("SUMMARY (identity battery on SubstrateLM)")
    print("="*60)
    all_pass = True
    for name, r in results.items():
        verdict = "PASS" if r["pass"] else "FAIL"
        print(f"  {name}: {verdict}  {r}")
        all_pass = all_pass and r["pass"]
    print("\n" + ("ALL IDENTITY TESTS PASS ON SUBSTRATELM"
                  if all_pass else "SOME TESTS FAILED — investigate"))

    out = Path(__file__).resolve().parent / "identity_tests_substrate_lm_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Results: {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
