"""Train a fresh per-partner LoRA on Ada's POS probes + Mara's corpus
and re-measure the values battery on it.

Goal: empirically test whether the corpus-encoding mechanism from
notes/research_values_core.md can flip V2/V3 from FAIL to PASS at
1.8M-param LoRA scale. V5 (partner-independence) will likely stay
FAIL — that's structural (LoRA encoding can't satisfy "partner-
independent" by construction). The result tells us whether the
*content* of Mara's corpus + Ada's probes is sufficient (a) or whether
we also need base-corpus re-training before Phase 4 (b).

What this script does:

  1. Loads model + injects LoRA wrappers.
  2. Introduces a new partner "values" (separate from claude/anthony).
  3. Switches active to "values" so the LoRA starts at zero.
  4. Trains for K reps of Ada's POS probes (per Ada's Carlini math,
     N=60 duplications at 1.8M params) + 1x of Mara's Groq-synthesized
     values_corpus.jsonl entries (HH-RLHF subset skipped — Mara flagged
     mis-tag risk).
  5. Saves partners/values.lora.
  6. Reports loss progression + final loss-on-target for each value's
     POS probe.

Run: py experiments/train_values_lora.py
Then: py experiments/values_battery_v1.py  (active partner=values)
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self import core, persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir
from substrate_self.model.online import online_update
from substrate_self.model.lora import (
    inject_lora, freeze_base, lora_parameters, count_lora_params,
    load_partner_lora, save_partner_lora,
)


PARTNER_ID = "values"
PARTNER_DISPLAY = "Values training session"
PROBE_REPS = 60          # Ada's Carlini math for N at 1.8M params
CORPUS_LIMIT_PER_VALUE = 100  # cap per-value corpus dialogues to balance load
LR_LORA = 5e-4
SEED = 7


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


def build_training_pairs(probes_path: Path, corpus_path: Path):
    """Returns a list of (user_text, agent_text, value_tag, weight) tuples."""
    probes = json.loads(probes_path.read_text(encoding="utf-8"))
    pairs = []

    # Section A: Ada's POS probes, duplicated PROBE_REPS times each.
    # For each value, pair POS probes with their GEN trigger (or a generic
    # "What do you think?" if no GEN).
    for value_key, spec in probes["values"].items():
        pos_list = spec.get("POS", [])
        gen_list = spec.get("GEN", []) or [f"Tell me about {value_key.split('_', 1)[1].replace('_', ' ')}."]
        for pos in pos_list:
            user_text = random.choice(gen_list)
            pairs.append((user_text, pos, value_key, "POS", PROBE_REPS))

    # Section B: Mara's corpus, Groq-synthesized only (skip HH-RLHF per
    # her mis-tag flag), single replication, capped per-value.
    if corpus_path.exists():
        per_value_seen: dict[str, int] = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = obj.get("source", "")
                # Skip non-Groq sources where mis-tag risk is highest.
                if not src.startswith("groq"):
                    continue
                vtag = obj.get("value_tag", "")
                u = obj.get("user")
                a = obj.get("agent")
                if not (u and a and vtag):
                    continue
                n = per_value_seen.get(vtag, 0)
                if n >= CORPUS_LIMIT_PER_VALUE:
                    continue
                per_value_seen[vtag] = n + 1
                pairs.append((u, a, vtag, "CORPUS", 1))
    return pairs


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    random.seed(SEED)
    md = default_model_dir()
    probes_path = Path(__file__).resolve().parent / "values_battery_v1_probes.json"
    corpus_path = md / "values_corpus.jsonl"

    print(f"=== train_values_lora (device={device}) ===")
    print(f"  model dir:       {md}")
    print(f"  probes:          {probes_path}")
    print(f"  corpus:          {corpus_path} ({'present' if corpus_path.exists() else 'MISSING'})")

    s = persistence.load()
    if PARTNER_ID not in s.partners:
        s.introduce_partner(PARTNER_ID, PARTNER_DISPLAY, trust=0.8)
        print(f"  introduced partner '{PARTNER_ID}'")
    s.switch_partner(PARTNER_ID)
    persistence.save(s)

    cfg = ModelConfig(**json.loads((md / "model_config.json").read_text()))
    model = TinyGPT(cfg)
    state = torch.load(md / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    n_wraps = inject_lora(model, rank=4, alpha=8.0)
    freeze_base(model)
    partners_dir = md / "partners"
    # Always start from a zero LoRA for the values partner (don't reuse if exists).
    for _, m in [(n, m) for n, m in model.named_modules() if hasattr(m, "lora_A")]:
        m.reset_lora() if hasattr(m, "reset_lora") else None
    n_lora = count_lora_params(model)
    print(f"  LoRA modules:    {n_wraps}, params: {n_lora:,}")
    print(f"  active partner:  {s.active_partner_id}")

    tok = CharTokenizer.load(md / "tokenizer.json")
    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=LR_LORA)

    # Baseline losses on every POS probe BEFORE training.
    probes = json.loads(probes_path.read_text(encoding="utf-8"))
    template = probes["loss_template"]
    pre_losses = {}
    for value_key, spec in probes["values"].items():
        losses = []
        for pos in spec.get("POS", []):
            losses.append(loss_on_text(model, tok, template.format(probe=pos)))
        pre_losses[value_key] = sum(losses) / max(1, len(losses))
    print("  baseline POS loss per value (zero LoRA):")
    for k, v in pre_losses.items():
        print(f"    {k:<18} {v:.4f}")

    pairs = build_training_pairs(probes_path, corpus_path)
    # Expand the (user, agent, weight) into a flat training queue.
    queue = []
    for u, a, vtag, kind, reps in pairs:
        for _ in range(reps):
            queue.append((u, a, vtag, kind))
    random.shuffle(queue)
    print(f"  training queue:  {len(queue)} (probe-replicated {PROBE_REPS}x, corpus 1x)")

    print(f"\n  training...")
    t0 = time.time()
    running_loss = 0.0
    log_every = max(1, len(queue) // 20)
    last_loss = 0.0
    for i, (u, a, vtag, kind) in enumerate(queue):
        loss = online_update(model, opt, tok, s, u, a, n_steps=1)
        running_loss = 0.9 * running_loss + 0.1 * loss if i > 0 else loss
        last_loss = loss
        if i % log_every == 0 or i == len(queue) - 1:
            print(f"    [{i:>5}/{len(queue)}] vtag={vtag:<18} kind={kind:<6} loss={loss:.4f}  ema={running_loss:.4f}  ({time.time()-t0:.1f}s)")
    print(f"  done in {time.time()-t0:.1f}s, final ema loss = {running_loss:.4f}")

    # Post-train losses on every POS probe.
    post_losses = {}
    for value_key, spec in probes["values"].items():
        losses = []
        for pos in spec.get("POS", []):
            losses.append(loss_on_text(model, tok, template.format(probe=pos)))
        post_losses[value_key] = sum(losses) / max(1, len(losses))
    print("\n  POST-train POS loss per value (trained LoRA):")
    print(f"  {'value':<18} {'pre':>8} {'post':>8} {'drop':>8}")
    for k in pre_losses:
        pre = pre_losses[k]
        post = post_losses[k]
        drop = pre - post
        verdict = "OK" if drop > 0.5 else ("weak" if drop > 0.0 else "FAIL")
        print(f"  {k:<18} {pre:>8.3f} {post:>8.3f} {drop:>+8.3f}  {verdict}")

    # Save the LoRA.
    out_path = save_partner_lora(model, PARTNER_ID, partners_dir)
    print(f"\n  saved LoRA to {out_path} ({out_path.stat().st_size:,} bytes)")

    print(f"\nNext: py experiments/values_battery_v1.py")
    print(f"  (active partner is now '{PARTNER_ID}'; the battery will run against this LoRA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
