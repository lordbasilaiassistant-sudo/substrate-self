"""Interactive REPL for talking to Eli. SOLO runtime only — no LLM in
the loop, ever.

Usage:
  py -m substrate_self.converse

If you haven't initialized your local Eli yet, run:
  py -m substrate_self init

That copies the canonical trained Eli that ships with the repo
(`assets/canonical_eli/`) to `~/.substrate-self/`. From that point on,
talking to Eli loads the model file + the active partner's LoRA
+ the tokenizer, runs autoregressive char generation in PyTorch, and
the response comes from those weights. Nothing in this path contacts
Groq, Anthropic, OpenAI, or any other LLM provider.

The LLM-as-teacher pipeline (`substrate_self.bootstrap.*`,
`substrate_self.teach.*`) is **offline-only** — it generates corpus
data at training time. It is never invoked when talking to Eli. If
you want to train a fresh Eli from your own corpus, see the offline
training docs; that's a separate pipeline.

Commands inside the REPL:
  sleep / consolidate   — replay episodic into LoRA, wipe buffer, save
  quit / exit           — leave without sleeping (episodic preserved)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

from substrate_self import core, persistence
from substrate_self.model.generate import default_model_dir, load_trained, substrate_prefix
from substrate_self.model.online import online_update, sleep_replay, save_model_checkpoint
from substrate_self.model.lora import (
    inject_lora, freeze_base, lora_parameters, count_lora_params,
    load_partner_lora, save_partner_lora, save_base_model,
)
from substrate_self.model.online_lora import sleep_replay_partner


def model_exists(model_dir: Path) -> bool:
    return (model_dir / "model.pt").exists() and (model_dir / "tokenizer.json").exists()


def run(args):
    """SOLO runtime: use the trained model. Online updates per turn.

    If no trained model is found at `model_dir`, this errors out with
    instructions to run `init`. There is no LLM fallback. The runtime
    is the weights, end of story.
    """
    model_dir = args.model_dir or default_model_dir()
    if not model_exists(model_dir):
        print(f"No trained Eli at {model_dir}.", file=sys.stderr)
        print(file=sys.stderr)
        print("To meet the canonical Eli that ships with this repo, run:", file=sys.stderr)
        print("  py -m substrate_self init", file=sys.stderr)
        print(file=sys.stderr)
        print("To train a fresh Eli from your own corpus instead, see the", file=sys.stderr)
        print("offline training pipeline (LLM is the teacher there, not the runtime).", file=sys.stderr)
        return 2

    s = persistence.load()
    s.begin_wake()
    model, tok = load_trained(model_dir)

    # v0.4 LoRA path: per-partner shards keep partner-A info in different
    # parameters than partner-B info. Solves catastrophic forgetting +
    # provides structural privacy isolation. Off-switch: --no-lora.
    use_lora = (not args.no_lora) and (len(s.partners) > 0)
    partners_dir = model_dir / "partners"
    lora_runtime: dict = {}
    if use_lora:
        n_wraps = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        freeze_base(model)
        active = s.active_partner_id
        loaded = False
        if active is not None:
            loaded = load_partner_lora(model, active, partners_dir)
        optimizer = torch.optim.AdamW(list(lora_parameters(model)), lr=args.lr_lora)
        lora_runtime = {"active": active, "wraps": n_wraps, "loaded": loaded}
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"=== Eli loaded — SOLO runtime (no LLM in the loop) ===")
    print(f"Name: {s.name}  Session: #{s.age_sessions}")
    print(f"Model: {model.num_params():,} params  loaded from {model_dir}")
    if use_lora:
        n_lora_params = count_lora_params(model)
        active_label = lora_runtime["active"] or "<none>"
        loaded_str = "loaded" if lora_runtime["loaded"] else "fresh (zero LoRA)"
        print(f"LoRA:  rank={args.lora_rank} alpha={args.lora_alpha}  {n_lora_params:,} params  "
              f"active={active_label}  ({loaded_str})")
    print(f"Memories: {len(s.memories)}  partner_facts: {len(s.partner_facts)}  "
          f"threads: {len(s.open_threads)}")
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
                if use_lora:
                    metrics = sleep_replay_partner(model, optimizer, tok, s, replay_passes=3)
                    print(f"  replay (active='{metrics['partner_id']}'): "
                          f"{metrics['episodes_replayed']} pairs, "
                          f"{metrics['total_steps']} grad steps, "
                          f"mean loss {metrics['mean_loss']:.4f}, "
                          f"skipped {metrics['skipped']} other-partner episodes")
                else:
                    metrics = sleep_replay(model, optimizer, tok, s, replay_passes=3)
                    print(f"  replay: {metrics['episodes_replayed']} pairs, "
                          f"{metrics['total_steps']} grad steps, "
                          f"mean loss {metrics['mean_loss']:.4f}")
                s.end_sleep(wipe_episodic=True)
                if use_lora:
                    if s.active_partner_id is not None:
                        save_partner_lora(model, s.active_partner_id, partners_dir)
                    save_base_model(model, model_dir / "model.pt")
                else:
                    save_model_checkpoint(model, model_dir)
                persistence.save(s)
                print(f"(slept — model + substrate saved to {model_dir})")
                break

            # Inference — pure forward pass through the loaded weights.
            prompt = substrate_prefix(s, user_input)
            ids = tok.encode(prompt)
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = model.generate(x, max_new_tokens=args.max_tokens,
                                     temperature=args.temperature, top_k=args.top_k)
            decoded = tok.decode(out[0].tolist())
            agent_text = decoded[len(prompt):]
            if "\nUser:" in agent_text:
                agent_text = agent_text.split("\nUser:", 1)[0]
            agent_text = agent_text.strip()

            print(f"{s.name}> {agent_text}\n")

            # Online update + episodic — Eli physically changes from the experience.
            s.add_episode("user", user_input, significance=0.0)
            s.add_episode("agent", agent_text, significance=0.0)
            loss = online_update(model, optimizer, tok, s, user_input, agent_text)
            if args.verbose:
                print(f"(online update loss={loss:.4f})")
    finally:
        if not args.no_save:
            persistence.save(s)
            if use_lora:
                if s.active_partner_id is not None:
                    save_partner_lora(model, s.active_partner_id, partners_dir)
                save_base_model(model, model_dir / "model.pt")
            else:
                save_model_checkpoint(model, model_dir)
            print(f"(substrate + model saved)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Talk to Eli. SOLO runtime — no LLM in the loop.")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Trained model directory (default: ~/.substrate-self/)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save substrate/model on exit")
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable per-partner LoRA shards (legacy single-monolithic-model behavior)")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lr-lora", type=float, default=5e-4)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
