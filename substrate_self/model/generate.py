"""Inference: load the trained model + tokenizer, prompt it, get text.

This is the substrate-self runtime. No LLM dependency. The substrate's
own learned language faculty produces the utterance.

Usage:
  py -m substrate_self.model.generate "User: Hi\nEli:"
  py -m substrate_self.model.generate "User: Hi\nEli:" --max-tokens 200
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from substrate_self.core import Substrate
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.transformer import ModelConfig, TinyGPT


def default_model_dir() -> Path:
    override = os.environ.get("SUBSTRATE_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".substrate-self"


def load_trained(model_dir: Optional[Path] = None) -> tuple[TinyGPT, CharTokenizer]:
    """Load the trained model + tokenizer + config from disk."""
    d = model_dir or default_model_dir()
    cfg_data = json.loads((d / "model_config.json").read_text(encoding="utf-8"))
    cfg = ModelConfig(**cfg_data)
    tok = CharTokenizer.load(d / "tokenizer.json")
    model = TinyGPT(cfg)
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tok


def substrate_prefix(s: Substrate, user_input: str) -> str:
    """Build a substrate-conditioned prompt prefix.

    The model was trained on dialogue formatted as 'User: ...\\n{name}: ...'.
    For inference we present the same surface format. The substrate doesn't
    get injected as a system prompt the way the bootstrap voice does — the
    model's identity comes from the WEIGHTS (which were trained on
    substrate-conditioned corpus), not from a runtime system message.
    """
    return f"User: {user_input.strip()}\n{s.name}:"


def generate_text(
    user_input: str,
    substrate: Optional[Substrate] = None,
    model_dir: Optional[Path] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.85,
    top_k: int = 40,
    device: Optional[str] = None,
) -> str:
    from substrate_self import persistence

    if substrate is None:
        substrate = persistence.load()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tok = load_trained(model_dir)
    model = model.to(device)
    prompt = substrate_prefix(substrate, user_input)
    ids = tok.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    decoded = tok.decode(out[0].tolist())
    # Strip the prompt prefix from the output
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    # Stop at next "User:" — the model often runs into the next turn
    if "\nUser:" in decoded:
        decoded = decoded.split("\nUser:", 1)[0]
    return decoded.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("user_input", nargs="?", default="Hi.")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    out = generate_text(
        user_input=args.user_input,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        model_dir=args.model_dir,
    )
    print(out)


if __name__ == "__main__":
    main()
