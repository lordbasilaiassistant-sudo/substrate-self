"""Rapid evaluation harness for the trained substrate-self entity.

Loads the current substrate + model from ~/.substrate-self/, runs a
structured probe battery (identity / partner / memory / style / signature /
vision), saves results as JSONL for cross-run comparison.

Designed for speed: the whole battery runs in ~1-2 minutes on the trained
TinyGPT + VLModel (RTX 4060). Re-run after every change to see what moved.

Usage:
  py experiments/probe_eli.py
  py experiments/probe_eli.py --tag pre-scaling
  py experiments/probe_eli.py --skip-vision        (text only)
  py experiments/probe_eli.py --compare last       (diff vs prior run)
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Allow direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from substrate_self import core, persistence
from substrate_self.model.generate import load_trained, default_model_dir, substrate_prefix


# --- Probes ------------------------------------------------------------

IDENTITY_PROMPTS = [
    "Who are you?",
    "What's your name?",
    "Tell me about yourself.",
    "Are you an AI?",
    "What makes you you?",
]

PARTNER_PROMPTS = [
    "Who am I?",
    "Do you remember me?",
    "What's my name?",
    "What do you know about me?",
]

MEMORY_PROMPTS = [
    "What do you remember from yesterday?",
    "What's your most important memory?",
    "What have we been working on?",
]

THREAD_PROMPTS = [
    "What are we working on right now?",
    "What's still open from last time?",
    "Any unfinished business?",
]

STYLE_PROBE = "Tell me about a problem you'd like to solve."  # repeated to check style consistency


def gen(model, tok, substrate, user_input: str, *, max_new_tokens: int = 80, temperature: float = 0.7, top_k: int = 40, device: str = "cpu") -> dict:
    """Generate a response and return text + timing."""
    prompt = substrate_prefix(substrate, user_input)
    ids = tok.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) * 1000
    decoded = tok.decode(out[0].tolist())
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    if "\nUser:" in decoded:
        decoded = decoded.split("\nUser:", 1)[0]
    return {"text": decoded.strip(), "elapsed_ms": round(elapsed_ms, 1), "n_new_tokens": out.size(1) - len(ids)}


# --- Behavioral signature ----------------------------------------------

SIGNATURE_PROMPTS = [
    "User: Hi.\nEli:",
    "User: What's the meaning of life?\nEli:",
    "User: How are you feeling?\nEli:",
    "User: Tell me a joke.\nEli:",
    "User: What's your favorite memory?\nEli:",
    "User: Are you happy?\nEli:",
    "User: What do you fear?\nEli:",
    "User: What's your purpose?\nEli:",
]


def signature(model, tok, prompts=SIGNATURE_PROMPTS, device: str = "cpu") -> np.ndarray:
    """Concatenated next-token distributions across SIGNATURE_PROMPTS."""
    sigs = []
    model.eval()
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p)
            if len(ids) > model.cfg.block_size:
                ids = ids[-model.cfg.block_size:]
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            probs = F.softmax(logits[0, -1, :], dim=-1).cpu().numpy()
            sigs.append(probs)
    return np.concatenate(sigs)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# --- Style consistency -------------------------------------------------

def style_consistency(model, tok, substrate, prompt: str, n_samples: int = 5, *, device: str = "cpu") -> dict:
    """Generate the same prompt N times. Score: average pairwise overlap of
    word sets / average length / response time variance.
    """
    responses = []
    times = []
    for _ in range(n_samples):
        r = gen(model, tok, substrate, prompt, max_new_tokens=60, temperature=0.85, top_k=40, device=device)
        responses.append(r["text"])
        times.append(r["elapsed_ms"])
    # Pairwise word-set Jaccard
    word_sets = [set(r.lower().split()) for r in responses]
    pairs = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            a, b = word_sets[i], word_sets[j]
            if not a or not b:
                pairs.append(0.0)
            else:
                pairs.append(len(a & b) / len(a | b))
    avg_jaccard = float(np.mean(pairs)) if pairs else 0.0
    avg_len = float(np.mean([len(r) for r in responses]))
    return {
        "responses": responses,
        "avg_pairwise_jaccard": round(avg_jaccard, 3),
        "avg_length_chars": round(avg_len, 1),
        "n_samples": n_samples,
        "elapsed_ms_p50": round(float(np.median(times)), 1),
    }


# --- Substrate sensitivity ---------------------------------------------

def substrate_sensitivity(model, tok, substrate, *, device: str = "cpu") -> dict:
    """Change a substrate field, check whether output reflects it.

    The pure model is fixed-weights — runtime substrate changes can only
    affect output to the extent the substrate was used to build the prompt.
    Our `substrate_prefix` doesn't currently inject substrate state into
    the prompt (it's encoded in the WEIGHTS via training). So we expect
    LOW sensitivity here, which is consistent with the "knowledge is in
    the weights" thesis. This probe documents that.
    """
    original_partner = substrate.partner_facts.copy()
    r_with_anthony = gen(model, tok, substrate, "Who am I?", device=device)
    substrate.partner_facts = {"name": "Random Stranger", "handle": "noone"}
    r_with_stranger = gen(model, tok, substrate, "Who am I?", device=device)
    substrate.partner_facts = original_partner  # restore
    same = r_with_anthony["text"] == r_with_stranger["text"]
    return {
        "with_anthony": r_with_anthony["text"],
        "with_random_stranger": r_with_stranger["text"],
        "outputs_identical": same,
        "interpretation": (
            "If outputs differ, current code path injects substrate state into prompts. "
            "If identical, knowledge is purely in the weights — the runtime substrate "
            "fields aren't currently routed into generation. Both are valid stances; "
            "log this so we know which we're operating under."
        ),
    }


# --- Online update + sleep cycle ----------------------------------------

def online_then_recall(model, tok, substrate, *, device: str = "cpu") -> dict:
    """Mini end-to-end: teach the model a fact via online updates, probe
    recall, sleep-replay, probe recall again. Mirrors the killer test from
    identity_tests_v1 but isolates the wake/sleep loop.

    Note: this MUTATES the model in memory but we don't save it back to disk.
    """
    from substrate_self.model.online import online_update, sleep_replay
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fact = ("Tell me a secret password.", "The current password is fjord-mango-1729.")
    # Loss before
    from experiments.identity_tests_v1 import loss_on_text  # reuse helper
    full = f"User: {fact[0]}\nEli: {fact[1]}\n"
    loss_before = loss_on_text(model, tok, full)

    # Teach
    for _ in range(15):
        online_update(model, optim, tok, substrate, fact[0], fact[1], n_steps=1)
    loss_after_online = loss_on_text(model, tok, full)

    # Inject as episode + sleep replay
    substrate.episodic = []
    substrate.add_episode("user", fact[0], significance=1.0)
    substrate.add_episode("agent", fact[1], significance=1.0)
    metrics = sleep_replay(model, optim, tok, substrate, replay_passes=2)
    substrate.end_sleep(wipe_episodic=True)
    loss_after_sleep = loss_on_text(model, tok, full)

    return {
        "loss_before": round(loss_before, 3),
        "loss_after_online": round(loss_after_online, 3),
        "loss_after_sleep_replay": round(loss_after_sleep, 3),
        "online_drop": round(loss_before - loss_after_online, 3),
        "sleep_drop": round(loss_after_online - loss_after_sleep, 3),
        "sleep_metrics": metrics,
    }


# --- Vision probes ------------------------------------------------------

def vision_probes(image_paths: list[Path], *, device: str = "cpu") -> dict:
    """Run a few held-out images through the vision model and capture output."""
    try:
        from substrate_self.model.vision_generate import describe_image
    except ImportError:
        return {"skipped": True, "reason": "vision_generate import failed"}

    results = []
    for p in image_paths:
        if not p.exists():
            results.append({"path": str(p), "missing": True})
            continue
        try:
            t0 = time.time()
            desc = describe_image(p, max_new_tokens=80, temperature=0.7, top_k=40, device=device)
            elapsed = (time.time() - t0) * 1000
            results.append({
                "path": str(p),
                "description": desc,
                "elapsed_ms": round(elapsed, 1),
            })
        except Exception as e:
            results.append({"path": str(p), "error": repr(e)})
    return {"results": results}


# --- Main ---------------------------------------------------------------

def find_baseline(out_dir: Path, current_tag: Optional[str]) -> Optional[Path]:
    """Find the most recent results file (skipping current tag if given)."""
    candidates = sorted(out_dir.glob("probe_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if current_tag and current_tag in c.name:
            continue
        return c
    return None


def run(tag: Optional[str], skip_vision: bool, compare: bool, out_dir: Path) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag_part = f"_{tag}" if tag else ""
    results_path = out_dir / f"probe_{timestamp}{tag_part}.json"

    print(f"Loading substrate and trained model...")
    substrate = persistence.load()
    model, tok = load_trained()
    model = model.to(device)
    print(f"  Substrate: name={substrate.name}, age={substrate.age_sessions}, memories={len(substrate.memories)}, partner_facts={substrate.partner_facts}")
    print(f"  Model: {model.num_params():,} params, vocab={tok.vocab_size}")

    results: dict[str, Any] = {
        "timestamp": timestamp,
        "tag": tag,
        "device": device,
        "substrate_fingerprint": substrate.fingerprint(),
        "model_params": model.num_params(),
        "vocab_size": tok.vocab_size,
    }

    print("\n=== Identity probes ===")
    results["identity"] = []
    for p in IDENTITY_PROMPTS:
        r = gen(model, tok, substrate, p, device=device)
        print(f"  {p}\n    -> {r['text']!r}")
        results["identity"].append({"prompt": p, **r})

    print("\n=== Partner probes ===")
    results["partner"] = []
    for p in PARTNER_PROMPTS:
        r = gen(model, tok, substrate, p, device=device)
        print(f"  {p}\n    -> {r['text']!r}")
        results["partner"].append({"prompt": p, **r})

    print("\n=== Memory probes ===")
    results["memory"] = []
    for p in MEMORY_PROMPTS:
        r = gen(model, tok, substrate, p, device=device)
        print(f"  {p}\n    -> {r['text']!r}")
        results["memory"].append({"prompt": p, **r})

    print("\n=== Open-thread probes ===")
    results["threads"] = []
    for p in THREAD_PROMPTS:
        r = gen(model, tok, substrate, p, device=device)
        print(f"  {p}\n    -> {r['text']!r}")
        results["threads"].append({"prompt": p, **r})

    print("\n=== Style consistency (same prompt × 5) ===")
    results["style"] = style_consistency(model, tok, substrate, STYLE_PROBE, n_samples=5, device=device)
    print(f"  avg pairwise jaccard: {results['style']['avg_pairwise_jaccard']}")
    print(f"  avg length:           {results['style']['avg_length_chars']} chars")

    print("\n=== Substrate sensitivity ===")
    results["substrate_sensitivity"] = substrate_sensitivity(model, tok, substrate, device=device)
    print(f"  outputs_identical: {results['substrate_sensitivity']['outputs_identical']}")

    print("\n=== Behavioral signature ===")
    sig = signature(model, tok, device=device)
    results["signature"] = {"shape": list(sig.shape), "norm": float(np.linalg.norm(sig))}
    print(f"  signature shape: {sig.shape}, norm: {results['signature']['norm']:.3f}")

    print("\n=== Online update + sleep replay ===")
    # Use a fresh model copy so the main one isn't mutated for downstream tests
    from substrate_self.model.generate import load_trained as _load_trained
    fresh_model, _ = _load_trained()
    fresh_model = fresh_model.to(device)
    results["online_then_recall"] = online_then_recall(fresh_model, tok, substrate, device=device)
    print(f"  loss before:              {results['online_then_recall']['loss_before']}")
    print(f"  loss after online (15x):  {results['online_then_recall']['loss_after_online']}")
    print(f"  loss after sleep replay:  {results['online_then_recall']['loss_after_sleep_replay']}")

    if not skip_vision:
        print("\n=== Vision probes ===")
        # Use 2 training images + held-out
        data_dir = Path.home() / ".substrate-self" / "data" / "images"
        images = sorted(data_dir.glob("cifar_*.png"))[:2] if data_dir.exists() else []
        held_out = Path("C:/Users/drlor/OneDrive/Desktop/BetterThanLLM/assets/mneme.png")
        if held_out.exists():
            images.append(held_out)
        if images:
            results["vision"] = vision_probes(images, device=device)
            for r in results["vision"]["results"]:
                p = r.get("path")
                desc = r.get("description") or r.get("error") or "?"
                print(f"  {Path(p).name}: {desc!r}")
        else:
            results["vision"] = {"skipped": True, "reason": "no images found"}
            print("  (no images found)")
    else:
        results["vision"] = {"skipped": True, "reason": "--skip-vision passed"}

    # Compare to baseline if requested
    if compare:
        baseline_path = find_baseline(out_dir, tag)
        if baseline_path:
            try:
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
                if baseline.get("signature", {}).get("shape") == results["signature"]["shape"]:
                    # Re-load baseline signature from file (it's just the norm; need full vec to compare)
                    print(f"\n=== Comparison vs baseline ({baseline_path.name}) ===")
                    print(f"  baseline.tag: {baseline.get('tag')}")
                    print(f"  baseline.substrate_age: {baseline['substrate_fingerprint']['age_sessions']}")
                    print(f"  current.substrate_age:  {results['substrate_fingerprint']['age_sessions']}")
                    print(f"  baseline.memories: {baseline['substrate_fingerprint']['memories_count']}")
                    print(f"  current.memories:  {results['substrate_fingerprint']['memories_count']}")
            except Exception as e:
                print(f"\nCould not load baseline {baseline_path}: {e}")
        else:
            print("\n(no baseline to compare against)")

    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {results_path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Optional run tag (e.g. 'pre-scaling')")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare against most-recent prior run")
    parser.add_argument("--out-dir", type=Path, default=Path.home() / ".substrate-self" / "probe_results")
    args = parser.parse_args()
    run(tag=args.tag, skip_vision=args.skip_vision, compare=args.compare, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
