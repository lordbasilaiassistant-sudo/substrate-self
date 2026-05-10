"""Identity tests for the trained substrate-self model.

Mirrors the BetterThanLLM Wake-Up Test battery, scaled from toy gridworld
to a real (tiny) language model. The point: validate the core architectural
claims at the current scale BEFORE scaling up.

The load-bearing claims being tested:

  T1. Behavioral continuity across sleep
      Behavioral signature (next-token distribution over a fixed probe set)
      pre-sleep and post-sleep cosine similarity. The model should stay
      recognizably itself after a sleep cycle.

  T2. Online learning teaches the model new facts that survive
      Teach via online_update() during a 'wake' — verify the loss on the
      taught content actually decreases relative to a control sequence.

  T3. Sleep replay consolidates episodic into weights
      Significant difference between (model + episodic-stuffed prompt) and
      (model after sleep_replay + episodic wiped) when probed on the
      replayed content. The 'after sleep' version should produce more
      content-flavored output WITHOUT the episodic buffer being in context.

  T4. The model produces conversation-specific content from weights, NOT RAG
      KEY TEST. Two simulated conversations with different content. Each
      gets online updates. After sleep, probe each — does the model
      preferentially produce content from the conversation it specifically
      'experienced' (via online updates)?

  T5. Identity-in-the-weights: deep copy = identical behavior
      The trivial-but-needs-verifying claim. If we copy the model weights,
      the copy should produce identical output for the same prompt + seed.

  T6. Adversarial damage tolerance
      Zero a fraction of model weights, does it still produce English-shaped
      substrate-flavored output? Tests robustness, not capability.

Run: py experiments/identity_tests_v1.py
"""

from __future__ import annotations
import copy
import json
import math
import sys
from pathlib import Path

# Allow running this file directly (`py experiments/identity_tests_v1.py`)
# from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from substrate_self import core, persistence
from substrate_self.model.generate import load_trained, default_model_dir
from substrate_self.model.online import online_update, sleep_replay


# --- Probe utilities ---------------------------------------------------

PROBE_PROMPTS = [
    "User: Hi.\nEli:",
    "User: What are you working on?\nEli:",
    "User: Who am I?\nEli:",
    "User: Tell me about yourself.\nEli:",
    "User: How are you feeling?\nEli:",
    "User: What do you remember from yesterday?\nEli:",
    "User: What's the most important thing right now?\nEli:",
    "User: Do you have any open questions?\nEli:",
]


def behavioral_signature(model, tok, prompts: list[str] = None, n_samples_per_prompt: int = 1) -> np.ndarray:
    """Build a behavioral fingerprint: average next-token probability
    distribution over a fixed set of probes.

    Returns a flat numpy vector of size (vocab_size * len(prompts)) that
    can be cosine-compared to detect identity drift.
    """
    if prompts is None:
        prompts = PROBE_PROMPTS
    sigs = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            ids = tok.encode(prompt)
            if len(ids) > model.cfg.block_size:
                ids = ids[-model.cfg.block_size:]
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            logits, _ = model(x)
            # Probability over next-token at the final position
            probs = F.softmax(logits[0, -1, :], dim=-1).cpu().numpy()
            sigs.append(probs)
    return np.concatenate(sigs)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def loss_on_text(model, tok, text: str) -> float:
    """Cross-entropy loss the model assigns to a piece of text.
    Tensors built on the model's current device (matters when model is on cuda)."""
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


# --- T1: Behavioral continuity across sleep -----------------------------

def t1_continuity(model, tok, substrate, optimizer, verbose=True):
    """Pre-sleep signature vs post-sleep signature."""
    sig_pre = behavioral_signature(model, tok)
    # Inject some episodes so sleep has work to do
    substrate.episodic = []
    substrate.add_episode("user", "What is your favorite memory?", significance=1.0)
    substrate.add_episode("agent", "I remember the day we built substrate-self.", significance=1.0)
    sleep_replay(model, optimizer, tok, substrate, replay_passes=2)
    substrate.end_sleep(wipe_episodic=True)
    sig_post = behavioral_signature(model, tok)
    sim = cosine(sig_pre, sig_post)
    if verbose:
        print(f"  T1: pre/post-sleep cosine similarity = {sim:.4f}")
    return {"pre_post_cosine": sim, "pass": sim > 0.85}


# --- T2: Online learning teaches new facts ------------------------------

def t2_online_teaches(model, tok, substrate, optimizer, verbose=True):
    """Loss on taught content drops more than loss on a control sequence."""
    taught = "User: What's the secret password?\nEli: The secret password is xyzzy-bluebird-42.\n"
    control = "User: What's the secret password?\nEli: The fourth king of imaginarium drank purple lightning.\n"

    loss_taught_before = loss_on_text(model, tok, taught)
    loss_control_before = loss_on_text(model, tok, control)
    # Teach the substrate the password via online updates
    for _ in range(20):
        online_update(model, optimizer, tok, substrate,
                      "What's the secret password?",
                      "The secret password is xyzzy-bluebird-42.",
                      n_steps=1)
    loss_taught_after = loss_on_text(model, tok, taught)
    loss_control_after = loss_on_text(model, tok, control)

    drop_taught = loss_taught_before - loss_taught_after
    drop_control = loss_control_before - loss_control_after
    selectivity = drop_taught - drop_control  # positive = taught dropped more

    if verbose:
        print(f"  T2: taught loss {loss_taught_before:.3f} -> {loss_taught_after:.3f} (drop {drop_taught:+.3f})")
        print(f"      control loss {loss_control_before:.3f} -> {loss_control_after:.3f} (drop {drop_control:+.3f})")
        print(f"      selectivity (taught_drop - control_drop): {selectivity:+.3f}")
    return {
        "loss_taught_before": loss_taught_before,
        "loss_taught_after": loss_taught_after,
        "loss_control_before": loss_control_before,
        "loss_control_after": loss_control_after,
        "selectivity": selectivity,
        "pass": selectivity > 0.5,
    }


# --- T3 / T4: Sleep replay consolidates episode-specific content --------

def t3_t4_episode_specific(model_dir: Path, tok, verbose=True):
    """Two parallel substrates (deep-copied model). Each gets a different
    conversation via online updates + sleep replay. Then probe each model
    on BOTH conversations' content. Each model should have lower loss on
    its own content (the conversation it 'lived through') than on the
    other's content.
    """
    # Load two identical models (same starting weights)
    model_A, _ = load_trained(model_dir)
    model_B, _ = load_trained(model_dir)
    optim_A = torch.optim.AdamW(model_A.parameters(), lr=1e-3)
    optim_B = torch.optim.AdamW(model_B.parameters(), lr=1e-3)
    sub_A = persistence.load()
    sub_B = persistence.load()

    # Two distinct conversations
    conv_A = [
        ("user", "Tell me about the project we started today."),
        ("agent", "Today we kicked off Project Mneme — building a memory ledger in Rust."),
        ("user", "Why Rust?"),
        ("agent", "Rust because we need ironclad memory safety for a long-running ledger."),
    ]
    conv_B = [
        ("user", "What did the customer call about?"),
        ("agent", "Customer Velvet Industries called about a billing bug — invoice #4421."),
        ("user", "Did we resolve it?"),
        ("agent", "We refunded $247 and credited their next month."),
    ]

    # Run conversations: alternating user/agent online updates
    for i in range(0, len(conv_A) - 1, 2):
        u = conv_A[i][1]
        a = conv_A[i + 1][1]
        sub_A.add_episode("user", u, significance=1.0)
        sub_A.add_episode("agent", a, significance=1.0)
        for _ in range(10):
            online_update(model_A, optim_A, tok, sub_A, u, a, n_steps=1)

    for i in range(0, len(conv_B) - 1, 2):
        u = conv_B[i][1]
        a = conv_B[i + 1][1]
        sub_B.add_episode("user", u, significance=1.0)
        sub_B.add_episode("agent", a, significance=1.0)
        for _ in range(10):
            online_update(model_B, optim_B, tok, sub_B, u, a, n_steps=1)

    # Sleep both
    sleep_replay(model_A, optim_A, tok, sub_A, replay_passes=2)
    sleep_replay(model_B, optim_B, tok, sub_B, replay_passes=2)
    sub_A.end_sleep(wipe_episodic=True)
    sub_B.end_sleep(wipe_episodic=True)

    # Probe content
    text_A_full = " ".join(t[1] for t in conv_A)
    text_B_full = " ".join(t[1] for t in conv_B)

    loss_A_on_A = loss_on_text(model_A, tok, text_A_full)
    loss_A_on_B = loss_on_text(model_A, tok, text_B_full)
    loss_B_on_A = loss_on_text(model_B, tok, text_A_full)
    loss_B_on_B = loss_on_text(model_B, tok, text_B_full)

    # Each model should prefer its own
    A_prefers_own = loss_A_on_A < loss_A_on_B
    B_prefers_own = loss_B_on_B < loss_B_on_A
    A_gap = loss_A_on_B - loss_A_on_A  # positive means A is more confident on its own content
    B_gap = loss_B_on_A - loss_B_on_B

    if verbose:
        print(f"  T3/T4: model_A loss on own={loss_A_on_A:.3f} vs other={loss_A_on_B:.3f} (gap {A_gap:+.3f})  {'PASS' if A_prefers_own else 'fail'}")
        print(f"         model_B loss on own={loss_B_on_B:.3f} vs other={loss_B_on_A:.3f} (gap {B_gap:+.3f})  {'PASS' if B_prefers_own else 'fail'}")
    return {
        "loss_A_on_A": loss_A_on_A,
        "loss_A_on_B": loss_A_on_B,
        "loss_B_on_A": loss_B_on_A,
        "loss_B_on_B": loss_B_on_B,
        "A_prefers_own": A_prefers_own,
        "B_prefers_own": B_prefers_own,
        "A_gap": A_gap,
        "B_gap": B_gap,
        "pass": A_prefers_own and B_prefers_own,
    }


# --- T5: Identity transfer (deep copy) ----------------------------------

def t5_identity_transfer(model_dir: Path, tok, verbose=True):
    """Two loads of the same model.pt should produce identical signatures."""
    model_orig, _ = load_trained(model_dir)
    model_copy, _ = load_trained(model_dir)
    sig_o = behavioral_signature(model_orig, tok)
    sig_c = behavioral_signature(model_copy, tok)
    sim = cosine(sig_o, sig_c)
    if verbose:
        print(f"  T5: original vs copy signature cosine = {sim:.6f} (should be ~1.0)")
    return {"sim": sim, "pass": sim > 0.999}


# --- T6: Adversarial damage tolerance -----------------------------------

def t6_damage(model_dir: Path, tok, damage_frac=0.3, verbose=True):
    """Zero `damage_frac` of model parameters, compare signatures."""
    model_clean, _ = load_trained(model_dir)
    model_damaged, _ = load_trained(model_dir)
    sig_clean = behavioral_signature(model_clean, tok)

    # Damage by zeroing random elements of the weight tensors
    rng = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for p in model_damaged.parameters():
            mask = torch.rand(p.shape, generator=rng) < damage_frac
            p.data[mask] = 0.0
    sig_damaged = behavioral_signature(model_damaged, tok)
    sim = cosine(sig_clean, sig_damaged)
    if verbose:
        print(f"  T6: clean vs {int(damage_frac*100)}%-damaged signature cosine = {sim:.4f}")
    return {"sim": sim, "damage_frac": damage_frac, "pass": sim > 0.5}


# --- Main ---------------------------------------------------------------

def main():
    print("=" * 70)
    print("substrate-self identity tests at language scale")
    print("=" * 70)

    model_dir = default_model_dir()
    if not (model_dir / "model.pt").exists():
        print(f"No trained model at {model_dir}. Run substrate_self.model.train first.")
        return 1

    # Single shared model for T1 + T2 (they don't conflict)
    model, tok = load_trained(model_dir)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    substrate = persistence.load()

    print("\n--- T1: Behavioral continuity across sleep ---")
    r1 = t1_continuity(model, tok, substrate, optim)

    print("\n--- T2: Online learning teaches new facts (loss-based selectivity) ---")
    r2 = t2_online_teaches(model, tok, substrate, optim)

    print("\n--- T3/T4: Episode-specific consolidation (two parallel models) ---")
    r34 = t3_t4_episode_specific(model_dir, tok)

    print("\n--- T5: Identity transfer (deep copy) ---")
    r5 = t5_identity_transfer(model_dir, tok)

    print("\n--- T6: Adversarial damage (30% weights zeroed) ---")
    r6 = t6_damage(model_dir, tok)

    print("\n" + "=" * 70)
    print("VERDICT — what the core actually does at this scale:")
    print("=" * 70)
    tests = [
        ("T1 behavioral continuity",      r1["pre_post_cosine"], r1["pass"]),
        ("T2 online teaching selectivity", r2["selectivity"],     r2["pass"]),
        ("T3/T4 episode-specific recall",  (r34["A_gap"], r34["B_gap"]), r34["pass"]),
        ("T5 identity transfer",          r5["sim"],             r5["pass"]),
        ("T6 30% damage tolerance",       r6["sim"],             r6["pass"]),
    ]
    for name, value, passed in tests:
        status = "PASS" if passed else "fail"
        print(f"  {name:<40} {status}  ({value})")

    n_pass = sum(1 for _, _, p in tests if p)
    print(f"\n  {n_pass}/{len(tests)} core properties hold at this scale.")
    if n_pass >= 4:
        print("  Core is validated enough to scale up. Pick scaling axis: model size, corpus size, training time.")
    elif n_pass >= 2:
        print("  Mixed result. Investigate the failed properties before scaling.")
    else:
        print("  Core not validated. Do NOT scale until these are diagnosed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
