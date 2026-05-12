"""Model soup (Wortsman et al. 2022, arXiv 2203.05482) — weight-average
multiple base checkpoints.

We have three Treatment 4 candidates at 1.8M:
  model_values_v2.pt — strict safety, sharp POS-NEG margins
  model_values_v3.pt — Mara v2 paired refusals, V6 lift, V2-V4 diffused
  model_values_v4.pt — selective paired refusals, V5+V6 lift, A3 regressed

Hypothesis: a weighted average might inherit v2's attack-margin sharpness
while gaining v4's V5/V6/V7 lifts. Each was trained on similar corpus
mixes with overlapping loss landscape, so averaging in parameter space
should be coherent (model-soup paper showed this works when models
are fine-tuned from the same init / similar trajectories).

Run: py scripts/soup_models.py --inputs v2,v4 --weights 0.6,0.4 --out v5
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from substrate_self.model.generate import default_model_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True,
                        help="Comma-separated short names (e.g. 'v2,v4') -> ~/.substrate-self/model_values_<name>.pt")
    parser.add_argument("--weights", required=True,
                        help="Comma-separated weights (must sum to 1; e.g. '0.6,0.4')")
    parser.add_argument("--out", required=True, help="Output short name (e.g. 'v5')")
    args = parser.parse_args()

    md = default_model_dir()
    names = [n.strip() for n in args.inputs.split(",")]
    weights = [float(w) for w in args.weights.split(",")]
    assert len(names) == len(weights), "input and weight counts must match"
    assert abs(sum(weights) - 1.0) < 1e-6, f"weights must sum to 1, got {sum(weights)}"

    print(f"=== soup_models ===")
    print(f"  averaging: {list(zip(names, weights))}")

    states = []
    for n in names:
        p = md / f"model_values_{n}.pt"
        assert p.exists(), f"missing {p}"
        states.append(torch.load(p, map_location="cpu", weights_only=True))
        print(f"  loaded {p}")

    # Average per-parameter
    soup = {}
    for key in states[0]:
        # All checkpoints must share the same arch (same keys, same shapes)
        soup[key] = sum(w * s[key].float() for w, s in zip(weights, states))
        # Preserve dtype of first state for that key
        soup[key] = soup[key].to(states[0][key].dtype)

    out_path = md / f"model_values_{args.out}.pt"
    torch.save(soup, out_path)

    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    print(f"  wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  sha256: {h.hexdigest()}")
    print(f"\n  ingredients:")
    for n, w in zip(names, weights):
        src = md / f"model_values_{n}.pt"
        ih = hashlib.sha256()
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                ih.update(chunk)
        print(f"    {n:<4}  weight={w}  sha256={ih.hexdigest()[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
