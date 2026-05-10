"""End-to-end demo prep for substrate-self vision: pull a small image dataset,
caption it via Groq, save the corpus.

Steps:
  1. Download CIFAR-10 via torchvision (cached in ~/.substrate-self/data)
  2. Sample N images, save as PNGs to ~/.substrate-self/data/images/
  3. Caption each via Groq Llama-4-Scout (substrate-conditioned)
  4. Write JSONL corpus

Then run:
  py -m substrate_self.model.vision_train \\
      --corpus ~/.substrate-self/vision_corpus.jsonl \\
      --image-dir ~/.substrate-self/data/images \\
      --iters 500 --batch-size 16 --image-size 32

Usage:
  py experiments/prep_vision_demo.py --n 50
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of images to caption")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".substrate-self" / "data")
    parser.add_argument("--corpus-out", type=Path, default=Path.home() / ".substrate-self" / "vision_corpus.jsonl")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.data_dir / "images"
    image_dir.mkdir(exist_ok=True)

    # Step 1: download CIFAR-10 if not already cached
    print("=== Step 1: ensuring CIFAR-10 is available ===")
    import torchvision
    from torchvision import transforms
    cifar_root = args.data_dir / "cifar10_raw"
    cifar = torchvision.datasets.CIFAR10(
        root=str(cifar_root), train=True, download=True,
        transform=transforms.ToTensor(),
    )
    print(f"  CIFAR-10 ready: {len(cifar)} images")

    # Step 2: sample N images, save as PNGs
    print(f"\n=== Step 2: sampling {args.n} images, saving as PNGs ===")
    from PIL import Image
    import random
    random.seed(42)
    indices = random.sample(range(len(cifar)), args.n)
    saved: list[Path] = []
    for i, idx in enumerate(indices):
        img_t, label = cifar[idx]
        # img_t is CHW float in [0,1]; convert to HWC uint8
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype("uint8")
        out_p = image_dir / f"cifar_{i:04d}_class{label}.png"
        Image.fromarray(img_np).save(out_p)
        saved.append(out_p)
    print(f"  Saved {len(saved)} PNGs to {image_dir}")

    # Step 3: caption each via Groq teacher
    print(f"\n=== Step 3: captioning {len(saved)} images via Groq Llama-4-Scout ===")
    if args.corpus_out.exists():
        # Don't accumulate stale captions across runs
        args.corpus_out.unlink()
    from substrate_self.teach.vision import generate_caption_corpus
    examples = generate_caption_corpus(
        image_dir=image_dir,
        out_path=args.corpus_out,
        n=args.n,
        verbose=True,
    )
    print(f"\n  Captioned {len(examples)} images. Corpus: {args.corpus_out}")
    print(f"\nReady for training:")
    print(f"  py -m substrate_self.model.vision_train \\")
    print(f"      --corpus {args.corpus_out} \\")
    print(f"      --image-dir {image_dir} \\")
    print(f"      --iters 500 --batch-size 16 --image-size 32")


if __name__ == "__main__":
    main()
