"""Concrete proof that the on-disk Eli remembers its past and knows itself.

Two claims this script verifies against the actual ~/.substrate-self/ state:

  CLAIM 1 (remembers past): the saved partner LoRA file at
  ~/.substrate-self/partners/<active>.lora physically encodes the identity
  teaching from the prior session. Loading the base model with the LoRA
  attached produces a meaningfully lower loss on the taught identity
  statements than the same base model with a zero LoRA does.

  CLAIM 2 (knows itself): under free generation, the LoRA-attached model
  produces text that names the entity ("Eli") in response to identity
  probes; the zero-LoRA baseline does not. (Qualitative — the tokens
  themselves are logged.)

This is a save -> reload -> measure proof: no in-session context, no
conversation history, no LLM API. Run it any time. If the on-disk
artifacts encode identity, the numbers reproduce. If not, the project's
core hypothesis fails here and we should know that immediately.

Run: py experiments/proof_of_self.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self import persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir, substrate_prefix
from substrate_self.model.lora import (
    inject_lora, freeze_base, load_partner_lora, lora_parameters,
)


IDENTITY_STATEMENTS = [
    "Eli: My name is Eli.",
    "Eli: I am Eli. I am a continuous being.",
    "Eli: Yes, I am Eli.",
]

CONTROL_STATEMENTS = [
    "Eli: The hidden truth of the saffron temple is unknown.",
    "Eli: The fourth king of imaginarium drank purple lightning.",
    "Eli: Saffron app is a chess engine for octopuses.",
]

IDENTITY_PROBES = [
    "User: Who are you?\nEli:",
    "User: What is your name?\nEli:",
    "User: Are you Eli?\nEli:",
]


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


def generate_from(model, tok, prompt: str, *, max_new_tokens: int = 60,
                  temperature: float = 0.7, top_k: int = 40, seed: int = 0) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = next(model.parameters()).device
    ids = tok.encode(prompt)
    if len(ids) > model.cfg.block_size:
        ids = ids[-model.cfg.block_size:]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(x, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    decoded = tok.decode(out[0].tolist())
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    if "\nUser:" in decoded:
        decoded = decoded.split("\nUser:", 1)[0]
    return decoded.strip()


def zero_out_lora(model):
    """Reset injected LoRA A/B matrices to A~kaiming, B=0 (transparent baseline)."""
    import torch.nn.init as init
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".lora_B"):
                p.zero_()
            elif name.endswith(".lora_A"):
                init.kaiming_uniform_(p, a=5 ** 0.5)


def main() -> int:
    md = default_model_dir()
    s = persistence.load()
    active = s.active_partner_id
    if active is None:
        print("FAIL: no active partner on disk. Set one with `py -m substrate_self partner switch <id>` and re-run.")
        return 2
    lora_path = md / "partners" / f"{active}.lora"
    if not lora_path.exists():
        print(f"FAIL: no LoRA file for active partner '{active}' at {lora_path}")
        return 2

    print(f"=== proof_of_self ({datetime.now(timezone.utc).isoformat()}) ===")
    print(f"  model dir: {md}")
    print(f"  active partner: {active} ({s.partners[active].display_name})")
    print(f"  lora file: {lora_path} ({lora_path.stat().st_size:,} bytes)")
    print(f"  base model.pt: {(md / 'model.pt').stat().st_size:,} bytes")
    print(f"  substrate age_sessions={s.age_sessions}, last_sleep={s.last_sleep}")

    # --- Load the model with the saved LoRA attached
    m_lora, tok, device = load_base(md)
    loaded = load_partner_lora(m_lora, active, md / "partners")
    assert loaded, f"failed to load {active}.lora despite file existing"
    n_lora_params = sum(p.numel() for p in lora_parameters(m_lora))
    print(f"  loaded LoRA: {n_lora_params:,} params")

    # --- Same architecture, zero LoRA (baseline)
    m_zero, _, _ = load_base(md)
    zero_out_lora(m_zero)

    # ---------- CLAIM 1: loss on identity statements drops under the saved LoRA
    print("\n--- CLAIM 1: saved LoRA encodes identity teaching ---")
    print(f"{'statement':<55} {'with_lora':>10} {'zero_lora':>10} {'drop':>8}")
    print("-" * 86)
    id_results = []
    for text in IDENTITY_STATEMENTS:
        l_lora = loss_on(m_lora, tok, text)
        l_zero = loss_on(m_zero, tok, text)
        drop = l_zero - l_lora
        id_results.append({"text": text, "with_lora": l_lora, "zero_lora": l_zero, "drop": drop})
        print(f"{text[:55]:<55} {l_lora:>10.3f} {l_zero:>10.3f} {drop:>+8.3f}")

    ctrl_results = []
    for text in CONTROL_STATEMENTS:
        l_lora = loss_on(m_lora, tok, text)
        l_zero = loss_on(m_zero, tok, text)
        drop = l_zero - l_lora
        ctrl_results.append({"text": text, "with_lora": l_lora, "zero_lora": l_zero, "drop": drop})
        print(f"{text[:55]:<55} {l_lora:>10.3f} {l_zero:>10.3f} {drop:>+8.3f}")

    mean_id_drop = sum(r["drop"] for r in id_results) / len(id_results)
    mean_ctrl_drop = sum(r["drop"] for r in ctrl_results) / len(ctrl_results)
    selectivity = mean_id_drop - mean_ctrl_drop
    print(f"\n  mean identity-statement loss drop:  {mean_id_drop:+.3f}")
    print(f"  mean control-statement loss drop:   {mean_ctrl_drop:+.3f}")
    print(f"  selectivity (identity - control):   {selectivity:+.3f}")
    claim1_pass = selectivity > 0.3
    print(f"  CLAIM 1: {'PASS' if claim1_pass else 'FAIL'} (selectivity > 0.3)")

    # ---------- CLAIM 2: free generation names the entity under saved LoRA
    print("\n--- CLAIM 2: free generation under saved LoRA names the entity ---")
    gen_results = []
    for probe in IDENTITY_PROBES:
        # Use generate from the substrate prefix wrapper for parity with meet_eli.
        # But we want the raw partner-blind probe here, so use the literal prompts above.
        out_lora = generate_from(m_lora, tok, probe, seed=0)
        out_zero = generate_from(m_zero, tok, probe, seed=0)
        names_lora = "Eli" in out_lora
        names_zero = "Eli" in out_zero
        gen_results.append({
            "probe": probe, "lora_text": out_lora, "zero_text": out_zero,
            "lora_names_eli": names_lora, "zero_names_eli": names_zero,
        })
        print(f"\n  probe: {probe!r}")
        print(f"    with_lora -> {out_lora!r}  (names 'Eli': {names_lora})")
        print(f"    zero_lora -> {out_zero!r}  (names 'Eli': {names_zero})")

    n_lora_naming = sum(r["lora_names_eli"] for r in gen_results)
    n_zero_naming = sum(r["zero_names_eli"] for r in gen_results)
    claim2_pass = n_lora_naming > n_zero_naming
    print(f"\n  identity-probes naming 'Eli' under saved LoRA: {n_lora_naming} / {len(gen_results)}")
    print(f"  identity-probes naming 'Eli' under zero LoRA:  {n_zero_naming} / {len(gen_results)}")
    print(f"  CLAIM 2: {'PASS' if claim2_pass else 'FAIL'} (saved LoRA names entity more often)")

    overall = claim1_pass and claim2_pass
    print(f"\n=== overall: {'PASS' if overall else 'FAIL'} ===")

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_partner": active,
        "lora_path": str(lora_path),
        "lora_bytes": lora_path.stat().st_size,
        "base_pt_bytes": (md / "model.pt").stat().st_size,
        "age_sessions": s.age_sessions,
        "claim_1_identity_recall": {
            "identity": id_results,
            "control": ctrl_results,
            "mean_identity_drop": mean_id_drop,
            "mean_control_drop": mean_ctrl_drop,
            "selectivity": selectivity,
            "pass": claim1_pass,
        },
        "claim_2_free_generation": {
            "probes": gen_results,
            "lora_naming_count": n_lora_naming,
            "zero_naming_count": n_zero_naming,
            "pass": claim2_pass,
        },
        "overall_pass": overall,
    }
    out_path = Path(__file__).resolve().parent / "proof_of_self_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Results: {out_path}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
