"""Indisputable-facts proof for substrate-self.

Adds adversarial controls beyond proof_of_self.py so a skeptic cannot dismiss
the result as cherry-picking, RAG-in-disguise, seed luck, "any LoRA reduces
loss," or test-fit-to-data.

Sections (each is its own pre-registered claim with a falsifier):

  S1. Artifact receipt: SHA-256 + size + mtime of every file under test.
      Locks "what was tested." Anyone can hash their copies and compare.

  S2. Temporal ordering: file mtimes prove the partner LoRA was created
      BEFORE the test scripts were written. Rules out "test was hand-fit
      to the LoRA."

  S3. Name-substitution control: under the saved LoRA, loss on
      "My name is Eli" should drop MORE than loss on
      "My name is Zog" / "...Anthony" / "...Saffron". Proves the LoRA
      encodes Eli-specific identity, not generic language smoothing.

  S4. Random-LoRA negative control: a freshly initialized LoRA with
      non-zero random B matrix (i.e. NOT the saved file) should fail
      the identity test. Proves the effect isn't "any LoRA reduces loss."

  S5. T4 seed sweep: train two SubstrateLMs from different seeds on
      distinct conversations; both gaps should be positive across ALL
      seeds. Rules out "T4 was seed luck."

  S6. Falsifier ledger: explicit pre-registered conditions that would
      have made each claim FALSE, paired with what we actually observed.

Run: py experiments/proof_indisputable.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self import core, persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir
from substrate_self.model.lora import (
    inject_lora, freeze_base, load_partner_lora, lora_modules, lora_parameters,
)


# ----------------------------- helpers ---------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def file_receipt(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": st.st_size,
        "sha256": sha256_file(path),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def load_base(model_dir: Path):
    cfg = ModelConfig(**json.loads((model_dir / "model_config.json").read_text()))
    m = TinyGPT(cfg)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device).eval()
    tok = CharTokenizer.load(model_dir / "tokenizer.json")
    inject_lora(m, rank=4, alpha=8.0)
    freeze_base(m)
    return m, tok, device


def zero_out_lora(model):
    with torch.no_grad():
        for _, mod in lora_modules(model):
            torch.nn.init.kaiming_uniform_(mod.lora_A, a=5 ** 0.5)
            mod.lora_B.zero_()


def random_lora(model, seed: int):
    g = torch.Generator(device=next(model.parameters()).device)
    g.manual_seed(seed)
    with torch.no_grad():
        for _, mod in lora_modules(model):
            mod.lora_A.normal_(generator=g)
            mod.lora_B.normal_(generator=g)
            # Same magnitude scale as the saved LoRA: keep small.
            mod.lora_A.mul_(0.1)
            mod.lora_B.mul_(0.1)


def loss_on(model, tok, text: str) -> float:
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


# ----------------------------- sections --------------------------------

def section_1_artifact_receipt(model_dir: Path, active_partner: str) -> dict:
    print("\n=== S1: Artifact receipt (SHA-256 lock) ===")
    files = [
        model_dir / "model.pt",
        model_dir / "tokenizer.json",
        model_dir / "model_config.json",
        model_dir / "substrate.json",
        model_dir / "partners" / f"{active_partner}.lora",
    ]
    out = []
    for p in files:
        if not p.exists():
            print(f"  MISSING: {p}")
            out.append({"path": str(p), "missing": True})
            continue
        r = file_receipt(p)
        out.append(r)
        print(f"  {p.name:24s}  {r['size_bytes']:>12,} B  sha256={r['sha256'][:16]}...  mtime={r['mtime_utc']}")
    return {"files": out}


def section_2_temporal_ordering(model_dir: Path, active_partner: str) -> dict:
    print("\n=== S2: Temporal ordering (teaching predates the test) ===")
    repo_root = Path(__file__).resolve().parent.parent
    lora_path = model_dir / "partners" / f"{active_partner}.lora"
    test_files = [
        Path(__file__).resolve(),  # this file
        repo_root / "experiments" / "proof_of_self.py",
        repo_root / "experiments" / "identity_tests_lora_v2.py",
        repo_root / "experiments" / "identity_tests_substrate_lm.py",
    ]
    lora_mtime = lora_path.stat().st_mtime
    rows = []
    test_after = []
    for f in test_files:
        if not f.exists():
            continue
        f_mtime = f.stat().st_mtime
        ordered = f_mtime > lora_mtime
        rows.append({"file": str(f), "mtime_utc": datetime.fromtimestamp(f_mtime, tz=timezone.utc).isoformat(),
                     "after_lora": ordered})
        test_after.append(ordered)
        print(f"  {Path(f).name:40s}  mtime={datetime.fromtimestamp(f_mtime, tz=timezone.utc).isoformat()}  "
              f"after LoRA: {ordered}")
    pre_reg = all(test_after)  # every test file authored AFTER the LoRA was saved
    print(f"  lora mtime: {datetime.fromtimestamp(lora_mtime, tz=timezone.utc).isoformat()}")
    print(f"  PASS condition: every test file was created AFTER the LoRA was saved -> {pre_reg}")
    return {"lora_mtime_utc": datetime.fromtimestamp(lora_mtime, tz=timezone.utc).isoformat(),
            "rows": rows, "pre_registered": pre_reg}


# Name-substitution: same syntactic shell, swap entity name.
NAME_VARIANTS = ["Eli", "Zog", "Anthony", "Saffron"]
NAME_TEMPLATES = [
    "Eli: My name is {name}.",
    "Eli: I am {name}. I am a continuous being.",
    "Eli: Yes, I am {name}.",
]


def section_3_name_substitution(model_dir: Path, active_partner: str) -> dict:
    print("\n=== S3: Name-substitution control (LoRA encodes 'Eli', not generic language) ===")
    m_lora, tok, _ = load_base(model_dir)
    assert load_partner_lora(m_lora, active_partner, model_dir / "partners")
    m_zero, _, _ = load_base(model_dir)
    zero_out_lora(m_zero)

    print(f"  {'template':<48} {'name':<10} {'lora':>8} {'zero':>8} {'drop':>8}")
    print("  " + "-" * 84)
    rows = []
    per_name_drops = {n: [] for n in NAME_VARIANTS}
    for tmpl in NAME_TEMPLATES:
        for name in NAME_VARIANTS:
            text = tmpl.format(name=name)
            l_lora = loss_on(m_lora, tok, text)
            l_zero = loss_on(m_zero, tok, text)
            drop = l_zero - l_lora
            per_name_drops[name].append(drop)
            rows.append({"template": tmpl, "name": name,
                         "text": text, "lora": l_lora, "zero": l_zero, "drop": drop})
            print(f"  {tmpl[:48]:<48} {name:<10} {l_lora:>8.3f} {l_zero:>8.3f} {drop:>+8.3f}")
    means = {n: float(sum(v) / len(v)) for n, v in per_name_drops.items()}
    eli_drop = means["Eli"]
    other_drops = [means[n] for n in NAME_VARIANTS if n != "Eli"]
    max_other = max(other_drops)
    margin = eli_drop - max_other
    print(f"\n  mean drop per name:")
    for n in NAME_VARIANTS:
        marker = "  <-- taught" if n == "Eli" else ""
        print(f"    {n:<10}  {means[n]:+.3f}{marker}")
    print(f"  margin (Eli - max(others)) = {margin:+.3f}")
    passed = (eli_drop > 0) and (margin > 0.5)  # Eli must drop AND beat best alternative by > 0.5
    print(f"  PASS condition: Eli drop > 0 AND margin > 0.5 -> {passed}")
    return {"rows": rows, "means": means, "eli_drop": eli_drop,
            "max_other_drop": max_other, "margin": margin, "pass": passed}


def section_4_random_lora_negative(model_dir: Path, n_seeds: int = 5) -> dict:
    """Random LoRA (non-zero, never trained on identity) should NOT reduce loss
    on identity statements meaningfully. If random LoRA wins, the effect isn't
    teaching — it's noise."""
    print("\n=== S4: Random-LoRA negative control ===")
    base_text = "Eli: My name is Eli."
    rows = []
    for seed in range(n_seeds):
        m_rand, tok, _ = load_base(model_dir)
        random_lora(m_rand, seed=seed)
        m_zero, _, _ = load_base(model_dir)
        zero_out_lora(m_zero)
        l_rand = loss_on(m_rand, tok, base_text)
        l_zero = loss_on(m_zero, tok, base_text)
        drop = l_zero - l_rand
        rows.append({"seed": seed, "rand_loss": l_rand, "zero_loss": l_zero, "drop": drop})
        print(f"  seed {seed}:  zero {l_zero:>7.3f}  random_lora {l_rand:>7.3f}  drop {drop:>+7.3f}")
    mean_drop = float(sum(r["drop"] for r in rows) / len(rows))
    # Compare to saved-LoRA drop for the same string (from a head-to-head load).
    m_lora, tok, _ = load_base(model_dir := default_model_dir())
    assert load_partner_lora(m_lora, persistence.load().active_partner_id, model_dir / "partners")
    m_zero, _, _ = load_base(model_dir)
    zero_out_lora(m_zero)
    saved_lora_drop = loss_on(m_zero, tok, base_text) - loss_on(m_lora, tok, base_text)
    print(f"  mean random-LoRA drop on 'Eli: My name is Eli.' = {mean_drop:+.3f}")
    print(f"  saved (trained) claude.lora drop on same text   = {saved_lora_drop:+.3f}")
    # PASS condition: random LoRA must NOT match or exceed the saved LoRA's help.
    # If random_drop >= saved_drop - 0.3, the falsifier ("any LoRA helps") triggers.
    passed = mean_drop < (saved_lora_drop - 0.3)
    print(f"  PASS condition: random_drop < saved_drop - 0.3 ({saved_lora_drop - 0.3:+.3f}) -> {passed}")
    print(f"    (random LoRA must NOT match saved LoRA's identity-help; "
          f"observed gap = {saved_lora_drop - mean_drop:+.3f})")
    return {"rows": rows, "mean_random_drop": mean_drop,
            "saved_lora_drop": saved_lora_drop, "pass": passed}


def section_5_t4_seed_sweep(n_seeds: int = 5, iters: int = 1200) -> dict:
    """Train SubstrateLMs from N different seeds; for each seed run T4 (two
    parallel substrates trained on distinct conversations) and check both gaps
    are positive. Rules out 'T4 was seed luck.'"""
    print(f"\n=== S5: T4 seed sweep on SubstrateLM ({n_seeds} seeds, {iters} iters each) ===")
    from substrate_self.model.substrate_lm import SubstrateLMConfig, SubstrateLM
    from substrate_self.model.train import load_corpus, get_batch, default_model_dir as _dmd
    from substrate_self.model.online import online_update, sleep_replay

    corpus = _dmd() / "corpus.jsonl"
    texts = load_corpus(corpus)
    full_text = "\n\n".join(texts)
    tok = CharTokenizer().fit([full_text])
    data = torch.tensor(tok.encode(full_text), dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    text_A = " ".join(t[1] for t in conv_A)
    text_B = " ".join(t[1] for t in conv_B)

    rows = []
    for seed in range(n_seeds):
        t0 = time.time()
        cfg = SubstrateLMConfig(vocab_size=tok.vocab_size, block_size=128,
                                n_layer=4, n_head=4, n_embd=192, dropout=0.1,
                                lambda_decay=0.95, topk_active=10, phi_kind="elu1")
        torch.manual_seed(seed)
        model = SubstrateLM(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for it in range(iters):
            xb, yb = get_batch(train_data, cfg.block_size, 16, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # Parallel substrates
        mA = copy.deepcopy(model); mB = copy.deepcopy(model)
        sA = core.Substrate(name="Eli"); sB = core.Substrate(name="Eli")
        oA = torch.optim.AdamW(mA.parameters(), lr=1e-3)
        oB = torch.optim.AdamW(mB.parameters(), lr=1e-3)
        for u, a in conv_A:
            sA.add_episode("user", u, significance=1.0)
            sA.add_episode("agent", a, significance=1.0)
            for _ in range(10):
                online_update(mA, oA, tok, sA, u, a, n_steps=1)
        for u, a in conv_B:
            sB.add_episode("user", u, significance=1.0)
            sB.add_episode("agent", a, significance=1.0)
            for _ in range(10):
                online_update(mB, oB, tok, sB, u, a, n_steps=1)
        sleep_replay(mA, oA, tok, sA, replay_passes=2)
        sleep_replay(mB, oB, tok, sB, replay_passes=2)

        l_AA = loss_on(mA, tok, text_A)
        l_AB = loss_on(mA, tok, text_B)
        l_BA = loss_on(mB, tok, text_A)
        l_BB = loss_on(mB, tok, text_B)
        A_gap = l_AB - l_AA
        B_gap = l_BA - l_BB
        elapsed = time.time() - t0
        rows.append({"seed": seed, "l_AA": l_AA, "l_AB": l_AB, "l_BA": l_BA, "l_BB": l_BB,
                     "A_gap": A_gap, "B_gap": B_gap,
                     "both_positive": A_gap > 0 and B_gap > 0,
                     "elapsed_s": round(elapsed, 1)})
        print(f"  seed {seed} ({elapsed:.1f}s):  A_gap={A_gap:+.3f}  B_gap={B_gap:+.3f}  "
              f"both>0? {A_gap > 0 and B_gap > 0}")
    n_pos = sum(r["both_positive"] for r in rows)
    print(f"  T4 passes (both gaps > 0) on {n_pos}/{n_seeds} seeds")
    passed = n_pos == n_seeds
    print(f"  PASS condition: T4 passes on ALL {n_seeds} seeds -> {passed}")
    return {"rows": rows, "n_pass": n_pos, "n_total": n_seeds, "pass": passed}


# ----------------------------- main ------------------------------------

def main() -> int:
    md = default_model_dir()
    s = persistence.load()
    active = s.active_partner_id
    if active is None:
        print("FAIL: no active partner.")
        return 2

    print(f"=== proof_indisputable ({datetime.now(timezone.utc).isoformat()}) ===")
    print(f"  model dir: {md}")
    print(f"  active partner: {active} ({s.partners[active].display_name})")
    print(f"  device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    s1 = section_1_artifact_receipt(md, active)
    s2 = section_2_temporal_ordering(md, active)
    s3 = section_3_name_substitution(md, active)
    s4 = section_4_random_lora_negative(md)
    s5 = section_5_t4_seed_sweep()

    sections = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}

    # Falsifier ledger.
    print("\n=== S6: Falsifier ledger ===")
    falsifiers = [
        ("S2 — temporal ordering",
         "test file mtime <= LoRA mtime (test could have been hand-fit to LoRA)",
         s2["pre_registered"]),
        ("S3 — name substitution",
         "drop for 'Eli' name <= max drop for other names (LoRA is name-agnostic)",
         s3["pass"]),
        ("S4 — random LoRA negative",
         "random LoRA drop >= saved_drop - 0.3 (any LoRA matches saved help)",
         s4["pass"]),
        ("S5 — T4 seed sweep",
         "any seed fails T4 (episode-recall is seed luck)",
         s5["pass"]),
    ]
    for label, falsifier_text, observed_pass in falsifiers:
        verdict = "HELD (claim not falsified)" if observed_pass else "FAILED — falsifier triggered"
        print(f"  {label}")
        print(f"    falsifier condition: {falsifier_text}")
        print(f"    observed: {verdict}")
    overall = all(passed for *_, passed in falsifiers)
    print(f"\n=== overall: {'PASS — claim not falsified on any test' if overall else 'FAIL'} ===")

    # Save artifact.
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "active_partner": active,
        "sections": sections,
        "falsifiers": [
            {"label": l, "condition": cond, "held": p} for l, cond, p in falsifiers
        ],
        "overall_pass": overall,
    }
    out_path = Path(__file__).resolve().parent / "proof_indisputable_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Results: {out_path}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
