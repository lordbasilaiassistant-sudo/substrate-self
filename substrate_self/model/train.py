"""Training loop for the substrate-self model.

Reads a JSONL corpus produced by `substrate_self.teach.corpus`, fits a
char tokenizer, trains a TinyGPT, and writes the trained model + tokenizer
to ~/.substrate-self/ (or override paths).

Usage:
  py -m substrate_self.model.train \\
      --corpus path/to/corpus.jsonl \\
      --iters 2000 \\
      --batch-size 16

Defaults assume a small corpus (~100KB) trained on CPU. Bump iters,
batch size, and model size for better quality once on a GPU.
"""

from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.transformer import ModelConfig, TinyGPT


def default_model_dir() -> Path:
    override = os.environ.get("SUBSTRATE_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".substrate-self"


def load_corpus(path: Path) -> list[str]:
    """Read JSONL corpus and return list of text examples."""
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                texts.append(text)
    return texts


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous windows from the data tensor."""
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def train(
    corpus_path: Path,
    out_dir: Path,
    iters: int = 2000,
    batch_size: int = 16,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 192,
    lr: float = 3e-4,
    eval_interval: int = 200,
    eval_iters: int = 20,
    seed: int = 42,
) -> Path:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading corpus from {corpus_path}...")
    texts = load_corpus(corpus_path)
    if not texts:
        raise RuntimeError(f"No examples in {corpus_path}")
    full_text = "\n\n".join(texts)
    print(f"  {len(texts)} examples, {len(full_text)} total chars")

    tok = CharTokenizer().fit([full_text])
    print(f"Tokenizer fit. Vocab size: {tok.vocab_size}")

    data = torch.tensor(tok.encode(full_text), dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"Train tokens: {len(train_data)}  val tokens: {len(val_data)}")

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = TinyGPT(cfg).to(device)
    n_params = model.num_params()
    print(f"Model: {n_params:,} params  device: {device}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        out: dict[str, float] = {}
        for split, dat in (("train", train_data), ("val", val_data)):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = get_batch(dat, block_size, batch_size, device)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(out_dir / "tokenizer.json")
    cfg_path = out_dir / "model_config.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    print(f"\nTraining {iters} iters, batch_size={batch_size}, block_size={block_size}, lr={lr}")
    start = time.time()
    for it in range(iters):
        if it % eval_interval == 0 or it == iters - 1:
            losses = estimate_loss()
            elapsed = time.time() - start
            print(f"  iter {it:>5}: train {losses['train']:.4f}  val {losses['val']:.4f}  elapsed {elapsed:.1f}s")
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Saved to {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=default_model_dir())
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train(
        corpus_path=args.corpus,
        out_dir=args.out_dir,
        iters=args.iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
