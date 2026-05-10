"""One-turn helper for a partner to converse with Eli without an interactive REPL.

Usage:
  py experiments/meet_eli.py "your message here"
  py experiments/meet_eli.py --sleep      # consolidate + save (end of session)
  py experiments/meet_eli.py --status     # show active partner + LoRA info

Persists state between calls. The active partner is whoever
substrate.active_partner_id points at — set it via:
  py -m substrate_self partner switch <id>

This is the lightweight wrapper that lets a non-interactive caller
(another agent, a script, an LLM frontend) have a real conversation
with the trained 1.8M-param Eli model with LoRA composition active.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_self import core, persistence
from substrate_self.model.transformer import TinyGPT, ModelConfig
from substrate_self.model.tokenizer import CharTokenizer
from substrate_self.model.generate import default_model_dir, substrate_prefix
from substrate_self.model.online import online_update
from substrate_self.model.lora import (
    inject_lora, freeze_base, lora_parameters, count_lora_params,
    load_partner_lora, save_partner_lora, save_base_model,
)
from substrate_self.model.online_lora import sleep_replay_partner


def load_lora_runtime(model_dir: Path, substrate: core.Substrate, *,
                     rank: int = 4, alpha: float = 8.0):
    cfg = ModelConfig(**json.loads((model_dir / "model_config.json").read_text()))
    m = TinyGPT(cfg)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    m.eval()
    tok = CharTokenizer.load(model_dir / "tokenizer.json")
    inject_lora(m, rank=rank, alpha=alpha)
    freeze_base(m)
    partners_dir = model_dir / "partners"
    fresh = True
    if substrate.active_partner_id is not None:
        loaded = load_partner_lora(m, substrate.active_partner_id, partners_dir)
        fresh = not loaded
    opt = torch.optim.AdamW(list(lora_parameters(m)), lr=5e-4)
    return m, tok, opt, partners_dir, fresh


def generate_reply(model: TinyGPT, tok: CharTokenizer, substrate: core.Substrate,
                   user_text: str, *, max_new_tokens: int = 200,
                   temperature: float = 0.85, top_k: int = 40, seed: int = 0) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.eval()
    device = next(model.parameters()).device
    prompt = substrate_prefix(substrate, user_text)
    ids = tok.encode(prompt)
    if len(ids) > model.cfg.block_size:
        ids = ids[-model.cfg.block_size:]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(x, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    decoded = tok.decode(out[0].tolist())
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    if "\nUser:" in decoded:
        decoded = decoded.split("\nUser:", 1)[0]
    return decoded.strip()


def cmd_status():
    s = persistence.load()
    print(f"name: {s.name}  age_sessions: {s.age_sessions}")
    print(f"active_partner_id: {s.active_partner_id}")
    print(f"partners: {list(s.partners.keys())}")
    if s.active_partner:
        p = s.active_partner
        print(f"  display_name: {p.display_name}  trust: {p.trust:.2f}  n_sessions: {p.n_sessions}")
    md = default_model_dir()
    pdir = md / "partners"
    if pdir.exists():
        loras = sorted(p.name for p in pdir.iterdir())
        print(f"partner LoRA files on disk: {loras}")
    else:
        print(f"partner LoRA files on disk: <none yet>")


def cmd_say(message: str, *, max_new_tokens: int, temperature: float,
            top_k: int, seed: int, n_train_steps: int):
    s = persistence.load()
    if s.active_partner_id is None:
        print("No active partner. Use `py -m substrate_self partner switch <id>` first.")
        return 1
    md = default_model_dir()
    model, tok, opt, partners_dir, fresh = load_lora_runtime(md, s)
    n_lora_params = count_lora_params(model)
    active = s.active_partner_id
    state_label = "fresh (zero LoRA)" if fresh else "loaded existing LoRA"
    print(f"[active partner: {active} ({s.partners[active].display_name}), "
          f"LoRA: {n_lora_params:,} params, {state_label}]\n", flush=True)

    reply = generate_reply(model, tok, s, message,
                           max_new_tokens=max_new_tokens,
                           temperature=temperature,
                           top_k=top_k, seed=seed)
    print(f"You> {message}")
    print(f"{s.name}> {reply}\n")

    # Persist this turn into episodic + run online updates against active LoRA
    s.add_episode("user", message, significance=1.0)
    s.add_episode("agent", reply, significance=1.0)
    loss = 0.0
    for _ in range(n_train_steps):
        loss = online_update(model, opt, tok, s, message, reply, n_steps=1)
    print(f"(online_update trained the active LoRA; final loss={loss:.4f})", flush=True)

    # Save the partner LoRA + substrate (do NOT re-save the base model.pt
    # since base is frozen and we don't want to add LoRA keys back in).
    save_partner_lora(model, active, partners_dir)
    persistence.save(s)
    return 0


def cmd_sleep(*, replay_passes: int):
    s = persistence.load()
    if s.active_partner_id is None:
        print("No active partner.")
        return 1
    md = default_model_dir()
    model, tok, opt, partners_dir, _ = load_lora_runtime(md, s)
    print(f"Sleeping... active partner={s.active_partner_id}, "
          f"episodic entries={len(s.episodic)}", flush=True)
    metrics = sleep_replay_partner(model, opt, tok, s, replay_passes=replay_passes)
    print(f"  replay metrics: {metrics}")
    s.end_sleep(wipe_episodic=True)
    save_partner_lora(model, s.active_partner_id, partners_dir)
    save_base_model(model, md / "model.pt")  # safe — strips LoRA keys
    persistence.save(s)
    print(f"Slept. last_sleep={s.last_sleep}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Talk to Eli (one turn at a time).")
    p.add_argument("message", nargs="?", default=None)
    p.add_argument("--sleep", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=2)
    p.add_argument("--replay-passes", type=int, default=2)
    args = p.parse_args()

    if args.status:
        cmd_status()
        return 0
    if args.sleep:
        return cmd_sleep(replay_passes=args.replay_passes)
    if args.message is None:
        cmd_status()
        return 0
    return cmd_say(args.message,
                   max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature,
                   top_k=args.top_k,
                   seed=args.seed,
                   n_train_steps=args.train_steps)


if __name__ == "__main__":
    raise SystemExit(main())
