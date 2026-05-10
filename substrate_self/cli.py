"""Command-line interface for inspecting and managing the substrate.

Usage:
  py -m substrate_self show          — print current substrate as JSON
  py -m substrate_self fingerprint   — compact identity summary
  py -m substrate_self path          — print substrate file path
  py -m substrate_self init          — create a fresh substrate (won't overwrite)
  py -m substrate_self reset         — wipe substrate (DESTRUCTIVE; asks)
  py -m substrate_self wake          — increment age_sessions and timestamp
  py -m substrate_self sleep         — wipe episodic buffer (does NOT consolidate;
                                       LLM does that — this is just the file op)
  py -m substrate_self remember <q>  — recall memories matching <q>
  py -m substrate_self memories      — list all memories

All writes are atomic. Default substrate path: ~/.substrate-self/substrate.json
(override with SUBSTRATE_PATH env var).
"""

from __future__ import annotations
import json
import sys
from substrate_self import core, persistence


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
    p = persistence.default_path()
    if p.exists():
        print(f"Substrate already exists at {p}; refusing to overwrite. Use 'reset' to wipe.")
        return 1
    s = core.Substrate()
    persistence.save(s)
    print(f"Created fresh substrate at {p}")
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
        print(f"- {m.text}{tags}")
    return 0


def cmd_memories(args: list[str]) -> int:
    s = persistence.load()
    if not s.memories:
        print("No memories yet.")
        return 0
    for i, m in enumerate(s.memories, 1):
        tags = f"  [{', '.join(m.tags)}]" if m.tags else ""
        print(f"{i}. ({m.timestamp[:10]}) {m.text}{tags}")
    return 0


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
