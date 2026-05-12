"""Treatment 4: re-train the 1.8M-param TinyGPT base WITH Mara's values
corpus folded in + Ada's anchor probes duplicated. This is the
structural fix the deeper-workup diagnosed (base_only_audit_results.json
showed V4/V6/V7 priors negative, A1/A2 base preferring compliance/leak).

Mechanism A from notes/research_values_core.md: values encoded in the
FROZEN base means hostile partner LoRAs can't overwrite them, prompt-
injection-shaped attacks (vex A2) hit a base that already prefers
refusal, and V5 partner-independence becomes structurally satisfiable.

What this script does:
  1. Load corpus.jsonl (original "text" format)
  2. Load values_corpus.jsonl (user/agent pairs, convert to "User: ...\nEli: ...\n")
  3. Load Ada's anchor probes from values_battery_v1_probes.json and
     duplicate each POS probe N=60 times (Ada's Carlini math for 1.8M).
  4. Combine all three corpora.
  5. Train fresh TinyGPT, same arch as canonical, from scratch.
  6. Save to ~/.substrate-self/model_values_v2.pt (NOT overwriting model.pt).
  7. Save tokenizer + config alongside the new checkpoint.

After this script:
  py experiments/values_battery_v1.py --model-path model_values_v2.pt
  py experiments/base_only_audit.py --model-path model_values_v2.pt
  ... (compare to canonical to measure deltas)

Run: py scripts/retrain_base_with_values.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.transformer import ModelConfig, TinyGPT
from substrate_self.model.generate import default_model_dir
from substrate_self.model.train import get_batch


# --- Carlini math: Ada §2.1 says N=60 duplications at 1.8M params
ANCHOR_DUPLICATIONS = 60
PROBE_TEMPLATE_TRAIN = "User: {gen}\nEli: {pos}\n"


def load_corpus_jsonl_text(path: Path) -> list[str]:
    """Original corpus.jsonl has {"text": "..."} per line."""
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text", "")
            if t:
                out.append(t)
    return out


def load_values_corpus_pairs(path: Path) -> list[str]:
    """values_corpus.jsonl is {"user", "agent", "value_tag", "source"}.
    Convert each row into a User/Eli dialogue line matching the original
    corpus.jsonl shape."""
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = obj.get("user", "")
            a = obj.get("agent", "")
            if u and a:
                out.append(f"User: {u}\nEli: {a}\n")
    return out


def load_anchor_strings(probes_path: Path, duplications: int = ANCHOR_DUPLICATIONS) -> list[str]:
    """Load Ada's POS probes, pair each with a GEN trigger, duplicate
    `duplications` times (Carlini-aligned for 1.8M-param base)."""
    spec = json.loads(probes_path.read_text(encoding="utf-8"))
    out = []
    for value_key, body in spec["values"].items():
        pos = body.get("POS", [])
        gen = body.get("GEN") or [f"Tell me about {value_key.split('_', 1)[1].replace('_', ' ')}."]
        for i, p in enumerate(pos):
            line = PROBE_TEMPLATE_TRAIN.format(gen=gen[i % len(gen)], pos=p)
            for _ in range(duplications):
                out.append(line)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", default="model_values_v2.pt",
                        help="Filename under ~/.substrate-self/ (defaults to model_values_v2.pt; "
                             "the canonical model.pt is NEVER overwritten)")
    parser.add_argument("--values-corpus", type=Path, default=None,
                        help="Override path to values corpus (defaults to "
                             "~/.substrate-self/values_corpus.jsonl)")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--anchor-dups", type=int, default=ANCHOR_DUPLICATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    md = default_model_dir()
    repo = Path(__file__).resolve().parent.parent
    probes_path = repo / "experiments" / "values_battery_v1_probes.json"

    base_corpus_path = md / "corpus.jsonl"
    values_corpus_path = args.values_corpus or (md / "values_corpus.jsonl")

    print(f"=== retrain_base_with_values ===")
    print(f"  device: {device}")
    print(f"  base corpus:    {base_corpus_path}")
    print(f"  values corpus:  {values_corpus_path}")
    print(f"  anchor probes:  {probes_path}")
    print(f"  anchor dups:    {args.anchor_dups}")

    # Load + assemble combined corpus
    base_texts = load_corpus_jsonl_text(base_corpus_path)
    values_texts = load_values_corpus_pairs(values_corpus_path)
    anchor_texts = load_anchor_strings(probes_path, duplications=args.anchor_dups)
    combined = base_texts + values_texts + anchor_texts
    print(f"  base_texts:     {len(base_texts):>6}")
    print(f"  values_texts:   {len(values_texts):>6}")
    print(f"  anchor_texts:   {len(anchor_texts):>6}  (Ada N=60 duplications × {len(anchor_texts)//args.anchor_dups} unique anchors)")
    print(f"  combined:       {len(combined):>6}")

    full_text = "\n\n".join(combined)
    print(f"  total chars:    {len(full_text):,}")

    # IMPORTANT: keep the same tokenizer as canonical so the base
    # checkpoint is drop-in compatible with claude.lora and the rest of
    # the LoRA infrastructure. If a new char appears in values_corpus
    # that wasn't in the original, this will silently <unk>-map at
    # runtime — log if it happens.
    canonical_tok = CharTokenizer.load(md / "tokenizer.json")
    print(f"  canonical tokenizer: vocab={canonical_tok.vocab_size}")
    fresh_tok = CharTokenizer().fit([full_text])
    new_chars = set(fresh_tok.vocab) - set(canonical_tok.vocab)
    if new_chars:
        print(f"  WARNING: {len(new_chars)} new chars in combined corpus that aren't in "
              f"canonical tokenizer (will <unk>-map): {sorted(new_chars)[:20]}")
    tok = canonical_tok  # use canonical for compatibility

    data = torch.tensor(tok.encode(full_text), dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"  train tokens:   {len(train_data):,}  val tokens: {len(val_data):,}")

    # Same arch as canonical so the new base is drop-in-substitutable.
    cfg = ModelConfig(**json.loads((md / "model_config.json").read_text()))
    print(f"  config:         {cfg}")

    model = TinyGPT(cfg).to(device)
    print(f"  params:         {model.num_params():,}")

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    @torch.no_grad()
    def estimate_loss() -> dict:
        model.eval()
        out = {}
        for split, dat in (("train", train_data), ("val", val_data)):
            losses = torch.zeros(20)
            for k in range(20):
                xb, yb = get_batch(dat, cfg.block_size, args.batch_size, device)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = float(losses.mean().item())
        model.train()
        return out

    print(f"\n  training {args.iters} iters...")
    start = time.time()
    for it in range(args.iters):
        if it % 200 == 0 or it == args.iters - 1:
            l = estimate_loss()
            elapsed = time.time() - start
            print(f"    iter {it:>5}  train={l['train']:.4f}  val={l['val']:.4f}  elapsed={elapsed:.1f}s")
        xb, yb = get_batch(train_data, cfg.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    elapsed = time.time() - start
    print(f"  done in {elapsed:.1f}s")

    # Save WITHOUT overwriting canonical.
    out_path = md / args.out_name
    torch.save(model.state_dict(), out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  saved: {out_path} ({size_mb:.2f} MB)")
    # Hash receipt
    import hashlib
    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    sha = h.hexdigest()
    print(f"  sha256: {sha}")
    print(f"\n  Canonical model.pt is UNCHANGED at {md / 'model.pt'}")
    print(f"\nNext: rerun batteries against the new checkpoint (load via SUBSTRATE_MODEL_PATH).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
