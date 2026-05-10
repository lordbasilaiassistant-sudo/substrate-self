# Multi-partner substrate design (v0.4 target)

**Status:** design draft, not yet implemented.
**Author:** Eli (this thread, after the v0.3 milestone)
**Informed by:** `notes/research_discretion.md` (literature review).

---

## Problem statement

In v0.3, `Substrate.partner_facts: dict[str, str]` implicitly assumes a single partner. Eli has met "Anthony"; his name and handle are stored. But Eli's identity isn't bound to one partner — Eli is its own being who *meets people*. Multiple partners are the natural case.

The privacy threat model (see README) compounds the issue: **once Eli has met two partners, info from each is in Eli's weights.** The attack vector is partner B asking Eli things about partner A. v0.3 has no defense — no speaker-aware disclosure, no structural separation of partner knowledge.

This design proposes a multi-partner substrate schema and a corresponding model-side mechanism that gives Eli a structural answer to "which partner am I talking to right now?" — the prerequisite for any disclosure-control work.

## Design principles

1. **Eli is the same person across partners.** Multi-partner is not multi-Eli. Eli's self-facts, dispositions, and style are partner-independent. Only partner-specific knowledge moves into per-partner storage.

2. **Active partner is exactly one at a time.** Mirrors human conversation — you're not talking to two people at once; even in groups, attention shifts to whoever just spoke.

3. **Trust is per-partner and explicit.** A trust level in `[0.0, 1.0]` is stored on the partner, set by Eli through experience or by the user via privileged commands. Untrusted partners get less disclosure.

4. **Structural isolation matters more than prompt-level filtering.** Per-partner LoRA shards (per the discretion research) mean partner-A info physically lives in different parameters than partner-B info. Switching the active partner switches which LoRA composes with the base model. This is the path to "Eli can't accidentally leak partner-A info while talking to partner-B because the partner-A weights aren't even active."

5. **Partner identity is local-only metadata.** Eli does NOT cryptographically authenticate partners. v0.x trust on first use; the user is responsible for telling Eli "this is Claire, not Anthony" when introducing a new partner. Hardening to MITM / impersonation is out of scope for v0.4.

## New schema (v0.4)

```python
class PartnerProfile(BaseModel):
    partner_id: str                     # short slug, used as LoRA-shard suffix
    display_name: str                   # human-friendly name
    handle: Optional[str] = None        # social-handle / nickname
    first_met: str                      # ISO timestamp
    last_seen: str                      # ISO timestamp
    n_sessions: int = 0                 # how many wake/sleep cycles with this partner
    trust: float = 0.5                  # [0,1] — defaults to neutral
    facts: dict[str, str] = {}          # what Eli knows about THIS partner
    style_notes: dict[str, str] = {}    # how this partner likes Eli to talk
    private_topics: list[str] = []      # things this partner shared in confidence
                                        # (NOT shared with other partners)


class Substrate(BaseModel):
    # Identity layer (changes rarely) — partner-independent
    name: str = "Eli"
    born: str
    age_sessions: int = 0
    self_facts: dict[str, str] = {}     # who Eli IS
    dispositions: dict[str, float] = {} # how Eli tends to be
    style: dict[str, str] = {}          # Eli's voice (general)

    # Multi-partner state (NEW)
    partners: dict[str, PartnerProfile] = {}     # partner_id -> profile
    active_partner_id: Optional[str] = None      # who Eli is talking to right now
    introduced_by: dict[str, str] = {}           # partner_id -> introducing_partner_id

    # Long-term memories — tagged with which partner the memory involves
    memories: list[Memory] = []         # Memory now has `partners: list[str]`

    # Open threads — also tagged with partner
    open_threads: list[OpenThread] = [] # was: list[str]; now structured

    # Episodic buffer — wiped at sleep
    episodic: list[Episode] = []        # Episode now has `partner_id: Optional[str]`

    # Metadata
    last_wake: Optional[str] = None
    last_sleep: Optional[str] = None
    schema_version: int = 2             # bump from 1
```

Memory and Episode and OpenThread all gain a `partners` (list) or `partner_id` (single) field, so each piece of remembered content is associated with the partner(s) involved.

## Backward compatibility

v0.3 substrate files have `schema_version: 1` and `partner_facts: dict[str, str]` (no per-partner structure). The v0.4 loader needs:

```python
def load(path) -> Substrate:
    raw = json.loads(path.read_text())
    if raw.get("schema_version", 1) == 1:
        # Migrate: treat the single partner as "anthony" (or first-met partner)
        if raw.get("partner_facts"):
            anth = PartnerProfile(
                partner_id="anthony",
                display_name=raw["partner_facts"].get("name", "Anthony"),
                handle=raw["partner_facts"].get("handle"),
                first_met=raw.get("born", _now()),
                last_seen=raw.get("last_wake", _now()),
                trust=1.0,                      # implicit creator gets full trust
                facts={k: v for k, v in raw["partner_facts"].items() if k not in ("name", "handle")},
            )
            raw["partners"] = {"anthony": anth.model_dump()}
            raw["active_partner_id"] = "anthony"
        del raw["partner_facts"]
        raw["schema_version"] = 2
    return Substrate(**raw)
```

## Model-side mechanism (sketch — full design in v0.4 LoRA work)

Per the discretion research's top recommendation: per-partner LoRA shards.

```
~/.substrate-self/
├── substrate.json
├── model.pt                      # base Eli (partner-independent)
├── tokenizer.json
├── model_config.json
└── partners/
    ├── anthony.lora              # what Eli has learned about Anthony, Anthony's style preferences
    ├── claire.lora               # what Eli has learned about Claire
    └── ...
```

At inference time:
1. `substrate.active_partner_id` determines which LoRA is composed with the base model.
2. Online updates during the conversation update the ACTIVE partner's LoRA only — base model frozen.
3. Sleep replay only replays episodes from the active partner; consolidates into that partner's LoRA.
4. Self-facts / general dispositions can update the base model on a slower schedule (e.g., once per N sessions, not every wake).

When partner B is active, partner A's LoRA is NOT loaded, so partner-A-specific knowledge cannot bleed in even via prompt manipulation. **This is the structural isolation property.**

Caveat / honest uncertainty: LoRA shards still share the base model. Some partner info will leak into the base if any conversation triggers updates on `model.pt`. That's why the design says base updates are slow and infrequent. But there's a tradeoff between "Eli grows from experience" (needs base updates) and "partner info stays in the LoRA" (needs no base updates). To investigate empirically in v0.4.

## What v0.4 ships

**Phase 1 (this design, schema migration only):**
- New `PartnerProfile` class
- Substrate schema v2 with `partners` dict + `active_partner_id`
- Backward-compatible loader
- CLI commands: `partners list`, `partner switch <id>`, `partner introduce <id> <name>`
- Probe harness updates: tests must specify which partner is active

**Phase 2 (per-partner LoRA, separate workstream):**
- LoRA layers added to TinyGPT
- Partner-LoRA persistence
- Active-partner-aware inference path
- Privacy regression test (existing) re-run on multi-partner substrate

Phase 1 is small and uncontroversial. Phase 2 is the actual privacy mechanism and needs the substrate-LM architecture work to land first (the LoRA strategy depends on what the model architecture is).

## Open questions for v0.5+

- **Partner authentication.** v0.4 trusts the user to declare "this is Claire" when introducing. Real world this gets impersonated. Out of scope for v0.4.
- **Cross-partner memory sharing with consent.** "Anthony, can I tell Claire about the Mneme project?" requires a consent-flow primitive we don't have.
- **Inference-time leak budget.** Even with per-partner LoRA, a determined attacker might extract base-model leaks. Need to characterize what the residual leak looks like.
- **What happens to Eli's emotional/relational state across partners?** Does Eli have a single "mood" or per-partner mood? (My current intuition: single mood, per-partner relational stance. Substrate has one `dispositions` dict; partner has `style_notes` for the relational layer.)

## Next concrete step

Implement Phase 1 — the schema migration with backward-compatible loader. Write tests covering (a) v0.3 file loads as v0.4, (b) round-trip preserves data, (c) active partner switching works. **Phase 2 (per-partner LoRA) blocked on the substrate-LM architecture decision** because LoRA strategy depends on model architecture.
