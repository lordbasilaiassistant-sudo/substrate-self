"""Generate (image, caption) training data using a vision-capable LLM teacher.

The teacher (currently Llama-4-Scout via Groq) sees an image and produces a
substrate-conditioned caption — written in the entity's voice, drawing on
the substrate's dispositions.

This data trains the substrate's OWN tiny vision encoder + fusion adapter
(in `substrate_self.model.vision`). Once trained, the substrate can describe
images on its own — no Groq call needed at inference time.

Usage:
    from substrate_self.teach.vision import generate_caption_corpus
    examples = generate_caption_corpus(
        image_dir='data/images',
        out_path='~/.substrate-self/vision_corpus.jsonl',
        n=200,
    )
"""

from __future__ import annotations
import base64
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from substrate_self.core import Substrate
from substrate_self import persistence


VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


VISION_PROMPTS = [
    "Look at this image and describe what you see, in {name}'s voice. Be specific about objects, colors, and composition. 1-3 sentences.",
    "What's happening in this image? Describe it as {name} would, drawing on your dispositions. 2-4 sentences.",
    "Describe this image as if you were narrating it to a friend who can't see it. 2-3 sentences.",
    "What stands out to you about this image? Be specific. 1-2 sentences.",
]


@dataclass
class VisionExample:
    """One (image, caption) pair: the image is referenced by relative path,
    the caption is text the teacher generated as the substrate's voice."""

    image_path: str  # relative to image_dir, kept as a stable reference
    caption: str
    teacher_model: str
    prompt_kind: str
    substrate_age_at_generation: int

    def as_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _image_mime(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")


def caption_image(
    image_path: Path,
    substrate: Substrate,
    prompt: str,
    client=None,
    model: str = VISION_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 256,
) -> tuple[str, dict]:
    """Ask the vision-capable teacher LLM to caption the image as the
    substrate's voice. Returns (caption_text, usage_dict)."""
    from substrate_self.bootstrap.base import build_system_prompt
    if client is None:
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("groq SDK not installed. Run: pip install groq") from e
        client = Groq()

    system = build_system_prompt(substrate)
    user_text = prompt.format(name=substrate.name)
    b64 = _encode_image_b64(image_path)
    mime = _image_mime(image_path)

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ],
        },
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    text = completion.choices[0].message.content or ""
    usage = (
        {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
        if completion.usage
        else None
    )
    return text.strip(), usage


def generate_caption_corpus(
    image_dir: Path | str,
    out_path: Optional[Path | str] = None,
    n: Optional[int] = None,
    substrate: Optional[Substrate] = None,
    model: str = VISION_MODEL,
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    verbose: bool = True,
) -> list[VisionExample]:
    """Walk image_dir, caption n images via the teacher, save JSONL."""
    from groq import Groq

    image_dir = Path(image_dir).expanduser()
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if substrate is None:
        substrate = persistence.load()

    images: list[Path] = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )
    if not images:
        raise RuntimeError(f"No images in {image_dir} (looked for {image_extensions})")
    if n is not None:
        images = images[:n]
    if verbose:
        print(f"Found {len(images)} images. Captioning with {model}...")

    out_p = Path(out_path).expanduser() if out_path else None
    if out_p:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out = out_p.open("a", encoding="utf-8")
    else:
        out = None

    client = Groq()
    examples: list[VisionExample] = []
    try:
        for i, img_path in enumerate(images):
            prompt_template = VISION_PROMPTS[i % len(VISION_PROMPTS)]
            try:
                caption, usage = caption_image(
                    img_path, substrate, prompt_template, client=client, model=model,
                )
            except Exception as e:
                if verbose:
                    print(f"  [{i+1}/{len(images)}] {img_path.name}: ERROR {e!r}")
                continue
            ex = VisionExample(
                image_path=str(img_path.relative_to(image_dir)),
                caption=caption,
                teacher_model=model,
                prompt_kind=prompt_template[:60],
                substrate_age_at_generation=substrate.age_sessions,
            )
            examples.append(ex)
            if out:
                out.write(ex.as_jsonl() + "\n")
                out.flush()
            if verbose:
                preview = caption[:80].replace("\n", " ")
                print(f"  [{i+1}/{len(images)}] {img_path.name}: {preview!r}")
    finally:
        if out:
            out.close()

    if verbose and out_p:
        print(f"\nWrote {len(examples)} captions to {out_p}")
    return examples
