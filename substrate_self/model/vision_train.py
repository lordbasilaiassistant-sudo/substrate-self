"""Train the VLModel (ViT + adapter + TinyGPT) on (image, caption) pairs.

Inputs:
  - JSONL of {"image_path": ..., "caption": ...} from teach/vision.py
  - Image directory (paths relative to it)
  - Existing trained TinyGPT (we initialize text portion from it)

Outputs to ~/.substrate-self/:
  - vision_model.pt     (full VLModel state_dict)
  - vision_config.json  (config of both ViT and TinyGPT)

Uses CUDA if available. RTX 4060: ~few minutes for small datasets.

Usage:
    py -m substrate_self.model.vision_train \\
        --corpus ~/.substrate-self/vision_corpus.jsonl \\
        --image-dir ./data/images \\
        --iters 500 --batch-size 16 --image-size 32
"""

from __future__ import annotations
import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.transformer import ModelConfig, TinyGPT
from substrate_self.model.vision import ViTEncoder, VisionAdapter, VLModel, VisionConfig


def default_model_dir() -> Path:
    override = os.environ.get("SUBSTRATE_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".substrate-self"


def load_image(path: Path, size: int) -> torch.Tensor:
    """Read image from disk → CHW float tensor in [0, 1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = torch.tensor(list(img.tobytes()), dtype=torch.float32) / 255.0
    return arr.view(size, size, 3).permute(2, 0, 1)  # CHW


class VisionCaptionDataset(Dataset):
    def __init__(self, corpus_path: Path, image_dir: Path, tokenizer: CharTokenizer, image_size: int, max_caption_len: int):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.max_caption_len = max_caption_len
        self.tok = tokenizer
        self.examples: list[dict] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img_p = self.image_dir / obj["image_path"]
                if not img_p.exists():
                    continue
                self.examples.append(obj)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        img = load_image(self.image_dir / ex["image_path"], self.image_size)
        caption = ex["caption"]
        ids = self.tok.encode(caption, add_bos=True, add_eos=True)
        if len(ids) > self.max_caption_len + 1:
            ids = ids[: self.max_caption_len + 1]
        # Pad to max_caption_len + 1 (so we can split into x[:-1], y[1:])
        pad_id = self.tok._stoi.get("<pad>", 0)
        while len(ids) < self.max_caption_len + 1:
            ids.append(pad_id)
        ids_t = torch.tensor(ids, dtype=torch.long)
        return img, ids_t


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    seqs = torch.stack([b[1] for b in batch])
    return imgs, seqs


def train(
    corpus_path: Path,
    image_dir: Path,
    out_dir: Path,
    text_model_dir: Path,
    iters: int = 500,
    batch_size: int = 16,
    image_size: int = 32,
    patch_size: int = 4,
    max_caption_len: int = 64,
    lr: float = 3e-4,
    eval_interval: int = 50,
    seed: int = 42,
    init_from_text: bool = True,
):
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer + text model config
    tok = CharTokenizer.load(text_model_dir / "tokenizer.json")
    text_cfg_data = json.loads((text_model_dir / "model_config.json").read_text(encoding="utf-8"))
    text_cfg = ModelConfig(**text_cfg_data)
    print(f"Tokenizer vocab: {tok.vocab_size}")
    print(f"Text model config: {text_cfg}")

    # Build VLModel — ViT params chosen to match text n_embd for clean fusion
    vis_cfg = VisionConfig(
        image_size=image_size,
        patch_size=patch_size,
        n_embd=text_cfg.n_embd,
        n_layer=text_cfg.n_layer,
        n_head=text_cfg.n_head,
    )
    n_vision_tokens = 16  # how many vision tokens get prepended
    # Need text_cfg.block_size >= n_vision_tokens + max_caption_len + buffer
    needed = n_vision_tokens + max_caption_len + 1
    if text_cfg.block_size < needed:
        print(f"WARNING: text block_size {text_cfg.block_size} < needed {needed}; will crop training sequences")

    vit = ViTEncoder(vis_cfg)
    adapter = VisionAdapter(vision_dim=vis_cfg.n_embd, text_dim=text_cfg.n_embd, n_vision_tokens=n_vision_tokens)
    gpt = TinyGPT(text_cfg)

    if init_from_text and (text_model_dir / "model.pt").exists():
        state = torch.load(text_model_dir / "model.pt", map_location="cpu", weights_only=True)
        gpt.load_state_dict(state)
        print(f"Initialized text portion from {text_model_dir / 'model.pt'}")
    else:
        print("Text portion: random init")

    vlm = VLModel(vit, adapter, gpt).to(device)
    print(f"VLModel params: {vlm.num_params():,}")

    # Dataset
    dataset = VisionCaptionDataset(corpus_path, image_dir, tok, image_size, max_caption_len)
    print(f"Dataset: {len(dataset)} (image, caption) pairs")
    if len(dataset) == 0:
        raise RuntimeError("Empty dataset — did you generate captions and place images at the expected paths?")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)

    optim = torch.optim.AdamW(vlm.parameters(), lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "text": asdict(text_cfg),
        "vision": asdict(vis_cfg),
        "n_vision_tokens": n_vision_tokens,
        "image_size": image_size,
        "patch_size": patch_size,
        "max_caption_len": max_caption_len,
    }
    (out_dir / "vision_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print(f"\nTraining {iters} iters, batch_size={batch_size}, lr={lr}")
    start = time.time()
    vlm.train()
    iters_done = 0
    losses_window = []
    iter_loader = iter(loader)
    while iters_done < iters:
        try:
            imgs, seqs = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            imgs, seqs = next(iter_loader)
        imgs = imgs.to(device)
        seqs = seqs.to(device)
        # x = caption_ids[:-1], y = caption_ids[1:]
        x = seqs[:, :-1]
        y = seqs[:, 1:]
        # Mask pad positions in target so they don't contribute to loss
        pad_id = tok._stoi.get("<pad>", 0)
        y = y.clone()
        y[y == pad_id] = -1
        _, loss = vlm(imgs, x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vlm.parameters(), 1.0)
        optim.step()
        losses_window.append(float(loss.item()))
        if len(losses_window) > 50:
            losses_window.pop(0)
        iters_done += 1
        if iters_done % eval_interval == 0 or iters_done == iters:
            avg = sum(losses_window) / len(losses_window)
            print(f"  iter {iters_done:>5}: loss(last)={loss.item():.4f}  loss(window50)={avg:.4f}  elapsed={time.time()-start:.1f}s")

    torch.save(vlm.state_dict(), out_dir / "vision_model.pt")
    print(f"\nDone in {time.time()-start:.1f}s. Saved to {out_dir / 'vision_model.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=default_model_dir())
    parser.add_argument("--text-model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--max-caption-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-init-from-text", action="store_true")
    args = parser.parse_args()

    train(
        corpus_path=args.corpus,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        text_model_dir=args.text_model_dir,
        iters=args.iters,
        batch_size=args.batch_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        max_caption_len=args.max_caption_len,
        lr=args.lr,
        init_from_text=not args.no_init_from_text,
    )


if __name__ == "__main__":
    main()
