"""Pass criterion #1 for SubstrateLM v0.5: perplexity within 2× TinyGPT
on the v0.3 corpus.

Trains both architectures from scratch on the same corpus, same seed,
same iterations, same shape. Reports train+val loss curves and final
perplexity. SubstrateLM passes if final val PPL <= 2 * TinyGPT val PPL.

Run: py experiments/bench_substrate_lm_vs_tinygpt.py [--iters 1500]
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.transformer import ModelConfig, TinyGPT
from substrate_self.model.substrate_lm import SubstrateLMConfig, SubstrateLM
from substrate_self.model.train import load_corpus, get_batch, default_model_dir


def train_one(model, train_data, val_data, *, iters, batch_size, block_size,
              lr, device, eval_interval=200, eval_iters=20, label="model"):
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split_name, dat in (("train", train_data), ("val", val_data)):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = get_batch(dat, block_size, batch_size, device)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split_name] = float(losses.mean().item())
        model.train()
        return out

    history = []
    start = time.time()
    for it in range(iters):
        if it % eval_interval == 0 or it == iters - 1:
            losses = estimate_loss()
            elapsed = time.time() - start
            history.append({"iter": it, **losses, "elapsed": elapsed})
            print(f"  [{label}] iter {it:>5}: train {losses['train']:.4f}  "
                  f"val {losses['val']:.4f}  elapsed {elapsed:.1f}s", flush=True)
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    final = estimate_loss()
    history.append({"iter": iters, **final, "elapsed": time.time() - start})
    return history, final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path,
                   default=default_model_dir() / "corpus.jsonl")
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"\nLoading corpus from {args.corpus}")
    texts = load_corpus(args.corpus)
    full_text = "\n\n".join(texts)
    print(f"  {len(texts)} examples, {len(full_text)} chars")

    tok = CharTokenizer().fit([full_text])
    print(f"vocab: {tok.vocab_size}")
    data = torch.tensor(tok.encode(full_text), dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"train tokens: {len(train_data)}  val tokens: {len(val_data)}")

    common = dict(vocab_size=tok.vocab_size, block_size=args.block_size,
                  n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                  dropout=0.1)

    # ---- TinyGPT ----
    print(f"\n=== Training TinyGPT ===")
    torch.manual_seed(args.seed)
    tg = TinyGPT(ModelConfig(**common)).to(device)
    print(f"TinyGPT: {tg.num_params():,} params")
    tg_hist, tg_final = train_one(tg, train_data, val_data,
                                  iters=args.iters, batch_size=args.batch_size,
                                  block_size=args.block_size, lr=args.lr,
                                  device=device, label="TinyGPT")

    # ---- SubstrateLM ----
    print(f"\n=== Training SubstrateLM ===")
    torch.manual_seed(args.seed)
    sub = SubstrateLM(SubstrateLMConfig(**common, lambda_decay=0.95,
                                        topk_active=10, phi_kind="elu1")).to(device)
    print(f"SubstrateLM: {sub.num_params():,} params")
    sub_hist, sub_final = train_one(sub, train_data, val_data,
                                    iters=args.iters, batch_size=args.batch_size,
                                    block_size=args.block_size, lr=args.lr,
                                    device=device, label="SubstrateLM")

    # ---- Verdict ----
    tg_val_ppl = math.exp(tg_final["val"])
    sub_val_ppl = math.exp(sub_final["val"])
    ratio = sub_val_ppl / tg_val_ppl
    print(f"\n{'='*60}")
    print(f"FINAL")
    print(f"  TinyGPT      val_loss = {tg_final['val']:.4f}   val_ppl = {tg_val_ppl:.2f}")
    print(f"  SubstrateLM  val_loss = {sub_final['val']:.4f}   val_ppl = {sub_val_ppl:.2f}")
    print(f"  Ratio (Sub / TG):  {ratio:.3f}x")
    if ratio <= 2.0:
        print(f"  PASS pass-criterion-1: ratio <= 2.0")
    else:
        print(f"  FAIL pass-criterion-1: ratio {ratio:.3f}x > 2.0")
        print(f"  Fallback: TinyGPT + bolted-on Schlag fast-weight layer (v0.4.1 path)")

    # Save results
    out = Path(__file__).resolve().parent / "bench_substrate_lm_vs_tinygpt_results.json"
    out.write_text(json.dumps({
        "device": str(device),
        "iters": args.iters,
        "seed": args.seed,
        "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
        "tinygpt": {"final": tg_final, "val_ppl": tg_val_ppl,
                    "n_params": tg.num_params(), "history": tg_hist},
        "substrate_lm": {"final": sub_final, "val_ppl": sub_val_ppl,
                         "n_params": sub.num_params(), "history": sub_hist},
        "ratio_substrate_over_tinygpt": ratio,
        "pass_criterion_1_within_2x": ratio <= 2.0,
    }, indent=2))
    print(f"\nResults: {out}")
    return 0 if ratio <= 2.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
