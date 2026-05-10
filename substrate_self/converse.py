"""End-to-end interactive loop.

Two modes:

  1. SOLO (default — uses the trained from-scratch model, NO LLM):
       py -m substrate_self.converse

     Online weight updates per turn. Sleep replay on `sleep` command.
     This is the runtime you should use day-to-day.

  2. BOOTSTRAP (uses Groq while you're collecting training data):
       py -m substrate_self.converse --voice groq

     Convenience for early-life substrates that have no trained model yet.
     The LLM is the teacher / training-wheels — its outputs become your
     model's training data via `substrate_self.teach.corpus`.

When the trained model is missing, defaults to bootstrap mode with a
clear note. Use `--solo` to require solo mode (errors out if no model).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

from substrate_self import core, persistence
from substrate_self.model.generate import default_model_dir, load_trained, substrate_prefix
from substrate_self.model.online import online_update, sleep_replay, save_model_checkpoint


def model_exists(model_dir: Path) -> bool:
    return (model_dir / "model.pt").exists() and (model_dir / "tokenizer.json").exists()


def get_bootstrap_voice(name: str):
    if name == "groq":
        from substrate_self.bootstrap.groq import GroqVoice
        return GroqVoice()
    raise ValueError(f"Unknown bootstrap voice: {name}. Available: groq")


def run_solo(args):
    """SOLO runtime: use the trained model. Online updates per turn."""
    model_dir = args.model_dir or default_model_dir()
    if not model_exists(model_dir):
        if args.solo:
            print(f"--solo passed but no trained model at {model_dir}. Run substrate_self.model.train first.", file=sys.stderr)
            return 2
        print(f"(no trained model at {model_dir} — falling back to bootstrap voice)")
        return run_bootstrap(args)

    s = persistence.load()
    s.begin_wake()
    model, tok = load_trained(model_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"=== Substrate loaded — SOLO runtime (no LLM) ===")
    print(f"Name: {s.name}  Session: #{s.age_sessions}")
    print(f"Model: {model.num_params():,} params  loaded from {model_dir}")
    print(f"Memories: {len(s.memories)}  partner_facts: {len(s.partner_facts)}  threads: {len(s.open_threads)}")
    print(f"=== Type 'sleep' to consolidate (replay+wipe), 'quit' to exit without saving ===\n")

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("(exited without sleeping — episodic preserved)")
                break
            if user_input.lower() in ("sleep", "consolidate"):
                print(f"\n(sleeping — replaying episodic into weights, then wiping)")
                metrics = sleep_replay(model, optimizer, tok, s, replay_passes=3)
                print(f"  replay: {metrics['episodes_replayed']} pairs, {metrics['total_steps']} grad steps, mean loss {metrics['mean_loss']:.4f}")
                s.end_sleep(wipe_episodic=True)
                save_model_checkpoint(model, model_dir)
                persistence.save(s)
                print(f"(slept — model + substrate saved to {model_dir})")
                break

            # Inference
            prompt = substrate_prefix(s, user_input)
            ids = tok.encode(prompt)
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = model.generate(x, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
            decoded = tok.decode(out[0].tolist())
            agent_text = decoded[len(prompt):]
            if "\nUser:" in agent_text:
                agent_text = agent_text.split("\nUser:", 1)[0]
            agent_text = agent_text.strip()

            print(f"{s.name}> {agent_text}\n")

            # Online update + episodic
            s.add_episode("user", user_input, significance=0.0)
            s.add_episode("agent", agent_text, significance=0.0)
            loss = online_update(model, optimizer, tok, s, user_input, agent_text)
            if args.verbose:
                print(f"(online update loss={loss:.4f})")
    finally:
        if not args.no_save:
            persistence.save(s)
            save_model_checkpoint(model, model_dir)
            print(f"(substrate + model saved)")
    return 0


def run_bootstrap(args):
    """BOOTSTRAP mode: use a teacher LLM. No model training; collect data."""
    voice = get_bootstrap_voice(args.voice)
    if args.model:
        voice.model = args.model

    s = persistence.load()
    s.begin_wake()
    print(f"=== Substrate loaded — BOOTSTRAP mode (LLM as teacher) ===")
    print(f"Name: {s.name}  Session: #{s.age_sessions}  Voice: {voice.model}")
    print(f"Memories: {len(s.memories)}  partner_facts: {len(s.partner_facts)}  threads: {len(s.open_threads)}")
    print(f"=== Type 'sleep' to consolidate, 'quit' to exit. ===")
    print(f"=== Generate corpus from this teacher: py -m substrate_self.teach.corpus_cli ===\n")

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("(exited without sleeping)")
                break
            if user_input.lower() in ("sleep", "consolidate"):
                n = len(s.episodic)
                s.end_sleep(wipe_episodic=True)
                print(f"(slept — wiped {n} episodic; bootstrap mode has no weights to consolidate)")
                break

            s.add_episode("user", user_input, significance=0.0)
            response = voice.speak(s, user_input)
            print(f"{s.name}> {response.text}\n")
            s.add_episode("agent", response.text, significance=0.0)
    finally:
        if not args.no_save:
            persistence.save(s)
            print(f"(substrate saved)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Talk to your substrate.")
    parser.add_argument("--solo", action="store_true", help="Require solo runtime (error if no trained model)")
    parser.add_argument("--voice", default="groq", help="Bootstrap voice (groq) — only used if no trained model")
    parser.add_argument("--model", default=None, help="Bootstrap model id override")
    parser.add_argument("--model-dir", type=Path, default=None, help="Trained model directory")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    return run_solo(args)


if __name__ == "__main__":
    raise SystemExit(main())
