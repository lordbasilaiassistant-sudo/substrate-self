"""Solo vision inference: feed an image path, get a caption from the
trained VLModel. NO Groq, NO Anthropic, no API. The substrate's own
trained vision faculty + language faculty produce the description.

Usage:
  py -m substrate_self.model.vision_generate path/to/image.png
  py -m substrate_self.model.vision_generate img.png --max-tokens 80 --temperature 0.7
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
from substrate_self.model.vision import ViTEncoder, VisionAdapter, VLModel, VisionConfig


def default_model_dir() -> Path:
    override = os.environ.get("SUBSTRATE_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".substrate-self"


def load_vlmodel(model_dir: Optional[Path] = None) -> tuple[VLModel, CharTokenizer, dict]:
    """Reconstruct VLModel from the saved configs + state dict."""
    d = model_dir or default_model_dir()
    cfg_payload = json.loads((d / "vision_config.json").read_text(encoding="utf-8"))
    text_cfg = ModelConfig(**cfg_payload["text"])
    vis_cfg = VisionConfig(**cfg_payload["vision"])
    n_vision_tokens = cfg_payload["n_vision_tokens"]

    vit = ViTEncoder(vis_cfg)
    adapter = VisionAdapter(vision_dim=vis_cfg.n_embd, text_dim=text_cfg.n_embd, n_vision_tokens=n_vision_tokens)
    gpt = TinyGPT(text_cfg)
    vlm = VLModel(vit, adapter, gpt)

    state = torch.load(d / "vision_model.pt", map_location="cpu", weights_only=True)
    vlm.load_state_dict(state)
    vlm.eval()

    tok = CharTokenizer.load(d / "tokenizer.json")
    return vlm, tok, cfg_payload


def load_image_tensor(path: Path, size: int) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = torch.tensor(list(img.tobytes()), dtype=torch.float32) / 255.0
    return arr.view(size, size, 3).permute(2, 0, 1)


def describe_image(
    image_path: Path,
    model_dir: Optional[Path] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.85,
    top_k: int = 40,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    vlm, tok, cfg = load_vlmodel(model_dir)
    vlm = vlm.to(device)
    img = load_image_tensor(image_path, cfg["image_size"]).unsqueeze(0).to(device)
    # Start with just the BOS token
    bos = tok._stoi.get("<bos>", 1)
    text_ids = torch.tensor([[bos]], dtype=torch.long, device=device)
    out = vlm.generate(img, text_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.image.exists():
        print(f"Image not found: {args.image}")
        raise SystemExit(2)
    text = describe_image(
        image_path=args.image,
        model_dir=args.model_dir,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
