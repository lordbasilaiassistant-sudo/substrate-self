"""Command-line interface for inspecting and managing the substrate.

Usage:
  py -m substrate_self show                     — print current substrate as JSON
  py -m substrate_self fingerprint              — compact identity summary
  py -m substrate_self path                     — print substrate file path
  py -m substrate_self init                     — copy the canonical trained Eli that ships with
                                                  the repo to ~/.substrate-self/. Idempotent;
                                                  won't clobber an existing local Eli without --force.
  py -m substrate_self reset                    — wipe substrate (DESTRUCTIVE; asks)
  py -m substrate_self wake                     — increment age_sessions and timestamp
  py -m substrate_self sleep                    — wipe episodic buffer (does NOT consolidate;
                                                  LLM does that — this is just the file op)
  py -m substrate_self remember <q>             — recall memories matching <q>
  py -m substrate_self memories                 — list all memories

  Multi-partner (v0.4):
  py -m substrate_self partner list             — list all partners with trust + sessions
  py -m substrate_self partner switch <id>      — set active_partner_id
  py -m substrate_self partner introduce <id> <display_name> [--handle <h>] [--trust <f>]
                                                — add a new partner (default trust 0.5)

All writes are atomic. Default substrate path: ~/.substrate-self/substrate.json
(override with SUBSTRATE_PATH env var).
"""

from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
from substrate_self import core, persistence


def _canonical_eli_dir() -> Path:
    """Path to assets/canonical_eli/ inside the repo (the shipped Eli)."""
    return Path(__file__).resolve().parent.parent / "assets" / "canonical_eli"


def cmd_show(args: list[str]) -> int:
    s = persistence.load()
    print(s.model_dump_json(indent=2))
    return 0


def cmd_fingerprint(args: list[str]) -> int:
    s = persistence.load()
    print(json.dumps(s.fingerprint(), indent=2))
    return 0


def cmd_path(args: list[str]) -> int:
    print(persistence.default_path())
    return 0


def cmd_init(args: list[str]) -> int:
    """Initialize ~/.substrate-self/ by copying the canonical trained Eli
    that ships with the repo.

    Behavior:
      - If ~/.substrate-self/model.pt already exists, do nothing (local
        Eli is yours; we don't clobber it). Pass --force to overwrite.
      - Otherwise, copy assets/canonical_eli/* to ~/.substrate-self/.
        After init you immediately have a working Eli (no training,
        no API keys).

    The canonical files in the repo stay untouched — your local copy
    diverges from there as you talk to Eli.
    """
    force = "--force" in args
    home_dir = persistence.default_path().parent
    home_dir.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_eli_dir()
    if not canonical.exists():
        print(f"FAIL: canonical Eli not found at {canonical}.")
        print("This usually means the repo wasn't cloned with the assets/ folder.")
        return 2

    model_pt = home_dir / "model.pt"
    if model_pt.exists() and not force:
        print(f"~/.substrate-self/model.pt already exists. Your local Eli is yours.")
        print(f"  Use 'py -m substrate_self init --force' to overwrite with canonical.")
        print(f"  Or 'py -m substrate_self reset' to wipe everything and re-init.")
        return 0

    # Copy canonical files (model + tokenizer + config + partners + substrate).
    copied = []
    for src in canonical.rglob("*"):
        if src.is_dir():
            continue
        if src.name == "README.md":  # don't copy the asset README
            continue
        rel = src.relative_to(canonical)
        dst = home_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel.as_posix())

    print(f"Initialized {home_dir} with the canonical trained Eli:")
    for r in copied:
        print(f"  - {r}")
    print()
    s = persistence.load()
    print(f"Eli name={s.name} age_sessions={s.age_sessions} "
          f"partners={list(s.partners.keys())} active={s.active_partner_id}")
    print()
    print("Talk to Eli:")
    print("  py experiments/meet_eli.py \"Hi Eli, who are you?\"")
    print("  py experiments/meet_eli.py --status")
    print("  py experiments/meet_eli.py --sleep")
    return 0


def cmd_reset(args: list[str]) -> int:
    p = persistence.default_path()
    if not p.exists():
        print(f"No substrate at {p}; nothing to reset.")
        return 0
    confirm = input(f"DESTRUCTIVE: wipe {p}? Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return 1
    p.unlink()
    print(f"Wiped {p}")
    return 0


def cmd_wake(args: list[str]) -> int:
    s = persistence.load()
    s.begin_wake()
    persistence.save(s)
    print(f"Awake. Session #{s.age_sessions}, last_wake={s.last_wake}")
    return 0


def cmd_sleep(args: list[str]) -> int:
    s = persistence.load()
    n_episodic = len(s.episodic)
    s.end_sleep(wipe_episodic=True)
    persistence.save(s)
    print(f"Slept. Wiped {n_episodic} episodic entries. last_sleep={s.last_sleep}")
    print("Note: this only wipes the buffer. Consolidation (turning episodic into")
    print("memories / self_facts / open_threads) is the LLM's job — invoke the")
    print("substrate-self skill to do that.")
    return 0


def cmd_remember(args: list[str]) -> int:
    if not args:
        print("Usage: substrate_self remember <query>")
        return 1
    s = persistence.load()
    matches = s.remember(" ".join(args))
    if not matches:
        print("No matching memories.")
        return 0
    for m in matches:
        tags = f"  [{', '.join(m.tags)}]" if m.tags else ""
        partner = f"  <{m.partner_id}>" if m.partner_id else ""
        print(f"- {m.text}{tags}{partner}")
    return 0


def cmd_memories(args: list[str]) -> int:
    s = persistence.load()
    if not s.memories:
        print("No memories yet.")
        return 0
    for i, m in enumerate(s.memories, 1):
        tags = f"  [{', '.join(m.tags)}]" if m.tags else ""
        partner = f"  <{m.partner_id}>" if m.partner_id else ""
        print(f"{i}. ({m.timestamp[:10]}) {m.text}{tags}{partner}")
    return 0


# ---- Partner subcommand --------------------------------------------------

def _partner_list(s: core.Substrate) -> int:
    if not s.partners:
        print("No partners yet. Use 'partner introduce <id> <display_name>' to add one.")
        return 0
    active = s.active_partner_id
    print(f"{'ACTIVE':<8}{'partner_id':<20}{'display_name':<28}{'trust':<8}{'sessions':<10}{'handle'}")
    print("-" * 90)
    for pid, p in s.partners.items():
        marker = "*" if pid == active else ""
        handle = p.handle or ""
        print(f"{marker:<8}{p.partner_id:<20}{p.display_name:<28}{p.trust:<8.2f}{p.n_sessions:<10}{handle}")
    return 0


def _partner_switch(s: core.Substrate, partner_id: str) -> int:
    if partner_id not in s.partners:
        print(f"Unknown partner '{partner_id}'. Known: {list(s.partners.keys())}")
        return 1
    s.switch_partner(partner_id)
    persistence.save(s)
    print(f"Active partner is now '{partner_id}' ({s.partners[partner_id].display_name}).")
    return 0


def _partner_introduce(s: core.Substrate, args: list[str]) -> int:
    # Args: <partner_id> <display_name> [--handle <h>] [--trust <f>]
    pos: list[str] = []
    handle: str | None = None
    trust: float = 0.5
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--handle" and i + 1 < len(args):
            handle = args[i + 1]
            i += 2
        elif a == "--trust" and i + 1 < len(args):
            try:
                trust = float(args[i + 1])
            except ValueError:
                print(f"--trust expects a float, got {args[i + 1]!r}")
                return 1
            i += 2
        else:
            pos.append(a)
            i += 1
    if len(pos) < 2:
        print("Usage: partner introduce <partner_id> <display_name> [--handle <h>] [--trust <f>]")
        return 1
    partner_id = pos[0]
    display_name = " ".join(pos[1:])
    if partner_id in s.partners:
        print(f"Partner '{partner_id}' already exists.")
        return 1
    p = s.introduce_partner(partner_id, display_name, handle=handle, trust=trust)
    persistence.save(s)
    print(f"Introduced partner '{p.partner_id}' (display={p.display_name}, trust={p.trust:.2f}).")
    if s.active_partner_id is None:
        s.active_partner_id = partner_id
        persistence.save(s)
        print(f"  (no active partner was set; '{partner_id}' is now active)")
    return 0


def cmd_partner(args: list[str]) -> int:
    if not args:
        print("Usage: partner <list|switch|introduce> ...")
        return 1
    sub, *rest = args
    s = persistence.load()
    if sub == "list":
        return _partner_list(s)
    if sub == "switch":
        if not rest:
            print("Usage: partner switch <partner_id>")
            return 1
        return _partner_switch(s, rest[0])
    if sub == "introduce":
        return _partner_introduce(s, rest)
    print(f"Unknown partner subcommand: {sub}. Use list, switch, or introduce.")
    return 1


COMMANDS = {
    "show": cmd_show,
    "fingerprint": cmd_fingerprint,
    "path": cmd_path,
    "init": cmd_init,
    "reset": cmd_reset,
    "wake": cmd_wake,
    "sleep": cmd_sleep,
    "remember": cmd_remember,
    "memories": cmd_memories,
    "partner": cmd_partner,
}


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, *rest = args
    fn = COMMANDS.get(cmd)
    if not fn:
        print(f"Unknown command: {cmd}")
        print("Run 'substrate_self --help' for usage.")
        return 1
    return fn(rest)


if __name__ == "__main__":
    raise SystemExit(main())
