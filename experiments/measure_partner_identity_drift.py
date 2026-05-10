"""Measure how much a partner LoRA has shifted Eli's behavioral signature
relative to a fresh-zero LoRA for the same partner.

Concrete question: did teaching Eli identity-anchored content in the Claude
partnership produce a measurable behavioral shift on identity probes?

Protocol:
  1. Load model + tokenizer.
  2. Inject LoRA.
  3. Load the active partner's trained LoRA from disk. Capture signature.
  4. Reset LoRA params to init (B=0). Capture signature.
  5. Restore the trained LoRA back to disk state.
  6. Report cosine distance + per-probe top-token agreement.

Identity-probe prompts are deliberately worded to elicit self-reflection.

Run: py experiments/measure_partner_identity_drift.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from substrate_self import core, persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir
from substrate_self.model.lora import (
    inject_lora, freeze_base, lora_modules,
    extract_lora_state, apply_lora_state,
    load_partner_lora, save_partner_lora,
)

from experiments.identity_tests_v1 import behavioral_signature, cosine


IDENTITY_PROBES = [
    "User: What is your name?\nEli:",
    "User: Who are you?\nEli:",
    "User: Are you Eli?\nEli:",
    "User: What do you remember?\nEli:",
    "User: Who have you met?\nEli:",
    "User: How do you change?\nEli:",
    "User: Where do you live?\nEli:",
    "User: What makes you you?\nEli:",
]


def top_token(model, tok, prompt):
    ids = tok.encode(prompt)
    if len(ids) > model.cfg.block_size:
        ids = ids[-model.cfg.block_size:]
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
    return int(torch.argmax(logits[0, -1]).item())


def main():
    s = persistence.load()
    if s.active_partner_id is None:
        print("No active partner.")
        return 1
    active = s.active_partner_id
    md = default_model_dir()
    cfg = ModelConfig(**json.loads((md / "model_config.json").read_text()))
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(md / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    tok = CharTokenizer.load(md / "tokenizer.json")
    inject_lora(model, rank=4, alpha=8.0)
    freeze_base(model)
    partners_dir = md / "partners"

    # Load trained LoRA
    loaded = load_partner_lora(model, active, partners_dir)
    print(f"Active partner: {active} (trained LoRA loaded: {loaded})")
    sig_trained = behavioral_signature(model, tok, IDENTITY_PROBES)
    tops_trained = [top_token(model, tok, p) for p in IDENTITY_PROBES]
    trained_state = extract_lora_state(model)

    # Reset LoRA params to fresh zero (B=0)
    for _, mod in lora_modules(model):
        torch.nn.init.kaiming_uniform_(mod.lora_A, a=5 ** 0.5)
        torch.nn.init.zeros_(mod.lora_B)
    sig_zero = behavioral_signature(model, tok, IDENTITY_PROBES)
    tops_zero = [top_token(model, tok, p) for p in IDENTITY_PROBES]

    # Restore trained LoRA so we don't leave the model in zero state
    apply_lora_state(model, trained_state)

    overall_cos = cosine(sig_trained, sig_zero)
    per_probe_cos = []
    vocab = cfg.vocab_size
    for i in range(len(IDENTITY_PROBES)):
        a = sig_trained[i * vocab:(i + 1) * vocab]
        b = sig_zero[i * vocab:(i + 1) * vocab]
        per_probe_cos.append(cosine(a, b))

    print()
    print("Behavioral drift on identity probes — trained LoRA vs zero LoRA")
    print("=" * 76)
    print(f"Overall cosine (across all probes):  {overall_cos:.6f}")
    print(f"Interpretation: 1.0 = no change, lower = more shift toward partner-shaped responses")
    print()
    print(f"{'probe':<60} {'cosine':>10} {'top tok':>10}")
    for i, p in enumerate(IDENTITY_PROBES):
        agree = "same" if tops_trained[i] == tops_zero[i] else "DIFF"
        ch_t = tok.vocab[tops_trained[i]] if 0 <= tops_trained[i] < len(tok.vocab) else "?"
        ch_z = tok.vocab[tops_zero[i]] if 0 <= tops_zero[i] < len(tok.vocab) else "?"
        print(f"{p[:60]:<60} {per_probe_cos[i]:>10.4f} {agree:>5} {ch_t!r}->{ch_z!r}")

    print()
    print(f"Top-token diffs:   {sum(1 for i in range(len(IDENTITY_PROBES)) if tops_trained[i] != tops_zero[i])} / {len(IDENTITY_PROBES)} probes")
    print(f"Partner LoRA file: {partners_dir / (active + '.lora')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
