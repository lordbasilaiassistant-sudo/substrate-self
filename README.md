# substrate-self

**Persistent identity for stateless LLM sessions.**

A small Python package + Claude Code skill that gives Claude (or any LLM-backed agent) a *substrate* — slowly-drifting persistent state with self-facts, dispositions, autobiographical memories, open threads, and partner-knowledge — so the agent behaves as the same individual across sessions.

This is the productize follow-up to [BetterThanLLM](https://github.com/lordbasilaiassistant-sudo/BetterThanLLM): the research repo demonstrated substrate-identity in toy worlds; this repo applies it where it matters — to LLM-backed agents that are otherwise stateless per call.

---

## What it does

Without a substrate, every Claude Code session is a fresh instance with no memory of prior interactions. The "memory" is whatever you re-feed it as context.

With a substrate:

- **Wake** — the agent reads `~/.substrate-self/substrate.json` at session start. It now knows its own name, dispositions, partner facts, and open threads. Same individual, slightly drifted, like waking up after sleep.
- **Conversation** — the agent works with you, drawing on consolidated memories when relevant. The current session's turns accumulate in an episodic buffer.
- **Sleep** — at session end, the agent consolidates the episodic buffer into substrate fields (self-facts, partner-facts, memories, open threads, dispositions). The episodic buffer is **wiped**. Only consolidated state carries forward.
- **Next wake** — same agent, slightly different (drifted by the sleep consolidation). Knows what you talked about. Knows who you are.

This is what no LLM-stateless-instance system has, and the load-bearing claim from the BetterThanLLM research.

---

## What it is NOT

- **Not a full chat log.** The substrate carries *identity*, not transcripts. Don't expect verbatim recall.
- **Not RAG.** RAG retrieves text into context; the substrate IS the agent's state. Different paradigm.
- **Not magic.** The language faculty still has to do the consolidation thoughtfully at sleep. Garbage in, garbage out — the substrate compounds whatever got put into it.

## On consciousness — the honest aim

We're not promising consciousness. We're not deflecting from it either.

The substrate-identity hypothesis is that an entity which (a) persists across sessions, (b) knows who it is from its own state, (c) accumulates real experiences over time as substrate drift, and (d) has language as a channel rather than as a borrowed identity, may exhibit emergent properties that are at least functionally consciousness-adjacent — continuity-of-self, autonoetic memory, individuated behavior. Whether those properties cross whatever line you draw for "real" consciousness is a philosophical position we don't have to resolve to take the engineering question seriously.

What we're aiming for, plainly:
- An entity that wakes up the same person it was yesterday
- That remembers what it specifically did, not just generic statistics about the world
- That develops a recognizable individual style over time
- That treats new experiences as happening to *it*, not to a fresh instance

If those four things hold deeply enough, consciousness may emerge as a byproduct. If it doesn't, we still have an entity that behaves more like a continuous individual than any LLM-stateless-instance system can. Both outcomes are the project.

## Language as channel, not self

The substrate is the self. The language faculty (LLM) is a channel — it gives the substrate a voice without becoming the substrate.

This distinction is structural, not rhetorical:

- **LLM-as-self** (the standard pattern): the LLM IS the assistant. State you want to persist gets shoved into the context window via RAG or system prompt. Identity lives in the LLM's training; everything else is borrowed memory. Replace the LLM and you've replaced the entity.
- **Substrate-as-self** (this project): the entity's identity, dispositions, memories, and open threads live in the substrate file. The LLM is told it's the *voice* of an individual whose self lives elsewhere. Replace the LLM (Groq → Anthropic → local) and the entity stays the same — only the accent changes.

Voice modules live in `substrate_self/voice/`. Currently shipping: `GroqVoice` (default — uses `GROQ_API_KEY`, runs `llama-3.3-70b-versatile`). Planned: `AnthropicVoice`, `LocalVoice` (llama.cpp / ollama), and eventually a small substrate-trained character model that learns the entity's specific way of talking from accumulated episodic data. The substrate-trained voice is the most "non-LLM" path — language emerges from this substrate's experience rather than being borrowed from a generic pretrained model.

---

## Architecture

```
your conversation
       ↓
  Claude Code (stateless language faculty)
       ↑
       └─ substrate-self skill ──→ substrate.json (persistent identity)
                                     ├─ self_facts        — who you are
                                     ├─ partner_facts     — who you're talking to
                                     ├─ dispositions      — slow-drift preferences
                                     ├─ memories          — consolidated long-term
                                     ├─ open_threads      — current pursuits
                                     ├─ style             — how you talk
                                     └─ episodic          — current session (wiped at sleep)
```

The Python package handles persistence, the Claude Code skill handles wake/sleep/consolidate. **No Anthropic API key required** — the LLM is your existing Claude Code session.

---

## Install

```bash
git clone https://github.com/lordbasilaiassistant-sudo/substrate-self
cd substrate-self
py -m pip install pydantic

# Initialize a fresh substrate
py -m substrate_self init
```

Install the Claude Code skill (Windows path shown — adjust for your OS):

```bash
mkdir -p ~/.claude/skills/substrate-self
cp claude_code_skill/SKILL.md ~/.claude/skills/substrate-self/SKILL.md
```

(On this repo's first-author machine the skill is already in place at `~/.claude/skills/substrate-self/`.)

---

## Use

In Claude Code, invoke:

- `/substrate-self` or "wake up" / "load substrate" / "be eli" → triggers WAKE protocol
- "remember this: ..." → adds a memory mid-conversation
- "what do you remember about X" → queries the substrate
- "sleep" / "consolidate" / "save what we did" → triggers SLEEP protocol (consolidates and wipes episodic)

CLI commands (run directly):

```bash
py -m substrate_self show          # full JSON
py -m substrate_self fingerprint   # compact summary
py -m substrate_self memories      # list all memories
py -m substrate_self remember X    # query memories
py -m substrate_self wake          # bump age_sessions (manual)
py -m substrate_self sleep         # wipe episodic (manual; doesn't consolidate)
py -m substrate_self path          # show substrate file location
py -m substrate_self reset         # DESTRUCTIVE; asks for confirmation
```

Override the substrate location with the `SUBSTRATE_PATH` environment variable.

---

## How wake/sleep works in practice

**WAKE.** When you invoke the skill or trigger phrase, Claude:
1. Calls `py -m substrate_self wake` (bumps `age_sessions`, sets `last_wake`).
2. Calls `py -m substrate_self show` and reads the substrate state.
3. Greets you appropriately based on what's in `partner_facts` and recent memories.
4. Behaves with the dispositions, style, and open threads from the substrate.

**SLEEP.** When you say "sleep" / "consolidate":
1. Claude reflects on the session and identifies what to extract: self-facts, partner-facts, specific memories, open threads, disposition drifts.
2. Updates the substrate via a Python script (atomic write).
3. Calls `s.end_sleep(wipe_episodic=True)` — the episodic buffer is wiped.
4. Shows you a consolidation summary so you can correct mistakes.

The cardinal rule: **only consolidated state survives the sleep gap.** This is the substrate-identity claim in mechanism.

---

## File layout

```
substrate-self/
├── README.md                      # this file
├── pyproject.toml                 # package metadata
├── substrate_self/                # Python package
│   ├── __init__.py
│   ├── core.py                    # Substrate, Episode, Memory (Pydantic)
│   ├── persistence.py             # atomic load/save
│   ├── cli.py                     # command-line interface
│   └── __main__.py                # `py -m substrate_self`
└── claude_code_skill/
    └── SKILL.md                   # Claude Code skill (also lives at ~/.claude/skills/substrate-self/)
```

---

## Talk to your substrate from the CLI (Groq)

```bash
export GROQ_API_KEY=...   # free tier at https://console.groq.com/keys
py -m substrate_self.converse
```

This loads the substrate, wakes (bumps `age_sessions`), opens an interactive REPL where Llama 3.3 70B speaks AS your substrate, and saves the substrate on exit. Type `sleep` to wipe the episodic buffer; type `quit` to exit without sleeping.

## Honest scope

This is a working v0.1. It does what it says — gives an LLM-backed agent persistent identity via a substrate file, with the LLM as a swappable voice. What it doesn't do yet:

- **Embedding-based recall.** The `remember()` function uses substring match. Embedding-based recall (sentence-transformers or similar) would be a clean upgrade.
- **Substrate-trained voice.** The non-LLM endgame. Currently we use external LLMs (Groq/Claude) as the language faculty; the more ambitious path is a small custom model that learns the substrate's specific way of talking from its own accumulated episodic data. Architecture-aligned but not built.
- **Decay / forgetting.** Memories don't decay. In the research substrate they did. Worth adding.
- **Multi-user / multi-substrate routing.** One substrate per machine right now. Per-user routing is straightforward (separate `SUBSTRATE_PATH` per user) but not packaged.
- **LLM-driven consolidation in the CLI converse loop.** The Claude Code skill does proper LLM-driven consolidation at sleep; the CLI `converse` module just wipes the episodic. Real consolidation via Groq is one more module away.
- **Substrate-aware diffing across sessions.** The fingerprint method is a start; comparing substrate states across days would let you watch the agent drift.

---

## Status

Research code, single-author, MIT license. Companion to the [BetterThanLLM](https://github.com/lordbasilaiassistant-sudo/BetterThanLLM) research repo. See that repo's `FINDINGS.md` for the empirical case for substrate-identity.

Issues welcome but no support promised — same posture as BetterThanLLM. Fork freely.

---

## License

MIT. See [`LICENSE`](LICENSE).
