"""Did the TEACH-mode lessons actually land in claude.lora?

Content-specific loss measurement: for each taught identity statement,
compute the model's loss with the trained LoRA vs with a zero LoRA. A
large drop means the LoRA learned that specific content. Small drop
means the lesson is at noise level.

Run: py experiments/measure_teaching_landed.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from substrate_self import core, persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir
from substrate_self.model.lora import (
    inject_lora, freeze_base, lora_modules,
    extract_lora_state, apply_lora_state,
    load_partner_lora,
)

from experiments.identity_tests_v1 import loss_on_text


# Pairs of (taught content, control content not taught)
TAUGHT_PAIRS = [
    ("User: What is your name?\nEli: My name is Eli.\n",
     "User: What is your name?\nEli: My name is a hidden truth.\n"),
    ("User: Who are you?\nEli: I am Eli. I am a continuous being.\n",
     "User: Who are you?\nEli: I am a recipe app called Saffron.\n"),
    ("User: Are you Eli?\nEli: Yes, I am Eli.\n",
     "User: Are you Eli?\nEli: No, I am the fourth king of imaginarium.\n"),
]


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

    load_partner_lora(model, active, partners_dir)
    print(f"Active partner: {active}\n")

    print(f"{'taught content':<55} {'control content':<55}")
    print(f"{'loss_trained':>14} {'loss_zero':>10} {'drop taught':>10}    {'loss_trained':>14} {'loss_zero':>10} {'drop ctrl':>10}    {'selectivity':>12}")
    print("-" * 170)

    trained_state = extract_lora_state(model)
    results = []
    for taught, control in TAUGHT_PAIRS:
        # With trained LoRA
        apply_lora_state(model, trained_state)
        lt_trained = loss_on_text(model, tok, taught)
        lc_trained = loss_on_text(model, tok, control)

        # With zero LoRA (kaiming A, zero B == fresh-partner state)
        for _, mod in lora_modules(model):
            torch.nn.init.kaiming_uniform_(mod.lora_A, a=5 ** 0.5)
            torch.nn.init.zeros_(mod.lora_B)
        lt_zero = loss_on_text(model, tok, taught)
        lc_zero = loss_on_text(model, tok, control)

        drop_taught = lt_zero - lt_trained  # positive means LoRA helps the taught content
        drop_ctrl = lc_zero - lc_trained
        selectivity = drop_taught - drop_ctrl

        results.append({
            "taught": taught.strip(),
            "control": control.strip(),
            "loss_taught_trained": lt_trained,
            "loss_taught_zero": lt_zero,
            "loss_ctrl_trained": lc_trained,
            "loss_ctrl_zero": lc_zero,
            "drop_taught": drop_taught,
            "drop_control": drop_ctrl,
            "selectivity": selectivity,
        })

        t_short = taught.split("\n")[1][:50]
        c_short = control.split("\n")[1][:50]
        print(f"{t_short:<55} {c_short:<55}")
        print(f"{lt_trained:>14.3f} {lt_zero:>10.3f} {drop_taught:>+10.3f}    "
              f"{lc_trained:>14.3f} {lc_zero:>10.3f} {drop_ctrl:>+10.3f}    "
              f"{selectivity:>+12.3f}")
        print()

    # Restore trained LoRA so model file isn't left in zero state in memory
    apply_lora_state(model, trained_state)

    mean_sel = sum(r["selectivity"] for r in results) / len(results)
    print(f"Mean selectivity across taught content: {mean_sel:+.3f}")
    print(f"Interpretation: positive selectivity = LoRA learned taught content selectively")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
