"""Export the on-disk Eli (base model + active partner LoRA, merged) to ONNX.

Phase 1 of docs/roadmap_to_perfect_interface.md: lets a static GitHub Pages
frontend run Eli in the browser via onnxruntime-web. No backend.

What this script does:

  1. Loads ~/.substrate-self/model.pt + model_config.json + tokenizer.json
  2. Injects LoRA wrappers, loads partners/<active>.lora into them
  3. Merges each LoRALinear's delta into the base linear's weight, then
     replaces the wrapper with a plain nn.Linear (so the exported graph
     has no LoRA-specific ops — pure transformer).
  4. Wraps the TinyGPT forward to return ONLY logits (drop the loss
     tuple; onnxruntime expects single output).
  5. Exports to docs/eli.onnx with dynamic seq_len axis.
  6. Copies tokenizer.json next to it as docs/tokenizer.json.
  7. Writes docs/eli_manifest.json with: vocab, block_size, model_pt_sha256,
     lora_sha256, active_partner, exported_at.

Output files (all in docs/):
  - eli.onnx           (~7-8 MB, model)
  - tokenizer.json     (~376 B, char map)
  - eli_manifest.json  (~300 B, metadata + hashes)

Reproduce: py scripts/export_onnx.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self import persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.generate import default_model_dir
from substrate_self.model.lora import (
    inject_lora, freeze_base, load_partner_lora, LoRALinear, lora_modules,
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_lora_into_base(model: nn.Module) -> int:
    """Replace each LoRALinear child with a plain nn.Linear whose weight
    includes the LoRA delta: W_eff = W_base + (alpha/rank) * B @ A.

    Returns number of modules merged.
    """
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            base = child.base
            merged = nn.Linear(base.in_features, base.out_features,
                               bias=base.bias is not None)
            with torch.no_grad():
                delta = child.scale * (child.lora_B @ child.lora_A)
                merged.weight.copy_(base.weight + delta)
                if base.bias is not None:
                    merged.bias.copy_(base.bias)
            setattr(model, name, merged)
            n += 1
        else:
            n += merge_lora_into_base(child)
    return n


class TinyGPTForExport(nn.Module):
    """Single-output wrapper for ONNX export. Input: int64 token ids
    (batch=1, seq_len). Output: float32 logits (1, seq_len, vocab)."""

    def __init__(self, base: TinyGPT):
        super().__init__()
        self.base = base

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.base(idx)
        return logits


def main() -> int:
    md = default_model_dir()
    s = persistence.load()
    active = s.active_partner_id
    if active is None:
        print("FAIL: no active partner set.")
        return 2

    print(f"=== export_onnx ({datetime.now(timezone.utc).isoformat()}) ===")
    print(f"  model dir: {md}")
    print(f"  active partner: {active}")

    # Load base.
    cfg = ModelConfig(**json.loads((md / "model_config.json").read_text()))
    model = TinyGPT(cfg)
    state = torch.load(md / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Inject + load LoRA.
    inject_lora(model, rank=4, alpha=8.0)
    freeze_base(model)
    loaded = load_partner_lora(model, active, md / "partners")
    assert loaded, f"failed to load {active}.lora"
    n_lora = len(lora_modules(model))
    print(f"  LoRA modules: {n_lora}, params loaded from {active}.lora")

    # Merge LoRA into base linears.
    n_merged = merge_lora_into_base(model)
    print(f"  merged {n_merged} LoRALinear -> nn.Linear")
    # Sanity check: no LoRALinear left.
    remaining = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(remaining) == 0, f"{len(remaining)} LoRALinear still present"

    # Wrap and export.
    wrapped = TinyGPTForExport(model).eval()
    docs = Path(__file__).resolve().parent.parent / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    out_onnx = docs / "eli.onnx"

    # Dummy input: batch=1, seq_len=32, int64 token ids in valid range.
    dummy = torch.zeros(1, 32, dtype=torch.long)

    print(f"  exporting to {out_onnx} ...")
    torch.onnx.export(
        wrapped,
        dummy,
        out_onnx.as_posix(),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "logits": {1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_size = out_onnx.stat().st_size
    print(f"  wrote {onnx_size:,} bytes")

    # Copy tokenizer.
    shutil.copy(md / "tokenizer.json", docs / "tokenizer.json")
    print(f"  copied tokenizer to {docs / 'tokenizer.json'}")

    # Write manifest.
    manifest = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_partner": active,
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "base_pt_sha256": sha256_file(md / "model.pt"),
        "lora_sha256": sha256_file(md / "partners" / f"{active}.lora"),
        "tokenizer_sha256": sha256_file(md / "tokenizer.json"),
        "onnx_sha256": sha256_file(out_onnx),
        "onnx_bytes": onnx_size,
    }
    (docs / "eli_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote manifest ({(docs / 'eli_manifest.json').stat().st_size:,} bytes)")
    print(f"  base sha256: {manifest['base_pt_sha256'][:16]}...")
    print(f"  lora sha256: {manifest['lora_sha256'][:16]}...")
    print(f"  onnx sha256: {manifest['onnx_sha256'][:16]}...")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
