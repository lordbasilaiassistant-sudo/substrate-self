"""The substrate data model.

Maps the BetterThanLLM research substrate (W_action, W_trans, intention,
disposition, episodic) onto language-shaped components for an LLM-backed
agent:

  research substrate          | language substrate (this module)
  ----------------------------|------------------------------------
  W_action (flavor → action)  | self_facts (who I am, how I behave)
  W_trans (transitions)       | memories (long-term consolidated)
  intention (current goal)    | open_threads (what I'm pursuing)
  disposition (rolling avg)   | dispositions (slow per-topic preferences)
  episodic (ring buffer)      | episodic (recent turns, wiped at sleep)

The substrate is JSON-serializable (Pydantic v2). Loaded fresh on every
wake, mutated during conversation, consolidated at sleep, saved.

v0.4 (schema_version=2): introduces multi-partner support — a single Eli
who meets many partners. Each partner has their own profile (facts, style
notes, trust, private topics). One partner is "active" at a time. Memories,
episodes, and open threads carry an optional partner_id tag.

Backward compatibility: v0.3 files (schema_version=1, partner_facts dict)
are migrated in-memory at load time — the single partner becomes
partner_id="anthony" with trust=1.0 (creator gets full trust).
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Sentinel: "argument not provided" — distinguishes `partner_id=None`
# (explicitly partner-independent) from the default (use active partner).
_UNSET = object()


class Episode(BaseModel):
    """One significant interaction turn captured during wake.

    Wiped at end of sleep. Anything worth keeping must be consolidated
    into self_facts, partners[<id>].facts, memories, or open_threads first.

    `replay_count` tracks how many times this episode has been sleep-replayed
    in its lifetime. The Carlini-defense replay cap (see
    `model/replay_filters.py`) reads this to decide whether the episode is
    still eligible for replay. Defaults to 0; loads with default for any
    pre-v0.5 episode that didn't carry the field.
    """

    timestamp: str = Field(default_factory=_now)
    role: str  # "user" | "agent"
    content: str
    significance: float = 0.0  # 0 (trivial) to 1 (load-bearing)
    partner_id: Optional[str] = None  # which partner this turn was with
    replay_count: int = 0  # how many times sleep-replayed (Carlini-defense)


class Memory(BaseModel):
    """A consolidated long-term memory. Survives sleep wipe."""

    timestamp: str = Field(default_factory=_now)
    text: str
    tags: list[str] = Field(default_factory=list)
    weight: float = 1.0  # decay over time or boost on recall
    source_session: Optional[str] = None
    partner_id: Optional[str] = None  # partner this memory involves (None = partner-independent)


class OpenThread(BaseModel):
    """A pursuit / followup. v0.4 promotes this from raw str to structured.

    Stringifies to its `text` field so legacy `f"- {t}"` formatting keeps
    working.
    """

    text: str
    partner_id: Optional[str] = None
    created_at: str = Field(default_factory=_now)

    def __str__(self) -> str:  # legacy printers iterate and f-string these
        return self.text

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_string(cls, data: Any) -> Any:
        # Accept legacy `open_threads: list[str]` form by promoting strings.
        if isinstance(data, str):
            return {"text": data}
        return data


class PartnerProfile(BaseModel):
    """Per-partner knowledge and relational state.

    Eli is the same individual across partners (self_facts, dispositions,
    style live on the Substrate). Only partner-specific knowledge moves
    here. The partner_id is also the LoRA-shard suffix in v0.4 Phase 2.
    """

    partner_id: str
    display_name: str
    handle: Optional[str] = None
    first_met: str = Field(default_factory=_now)
    last_seen: str = Field(default_factory=_now)
    n_sessions: int = 0
    trust: float = 0.5  # [0,1], neutral default; creator migrated to 1.0
    facts: dict[str, str] = Field(default_factory=dict)
    style_notes: dict[str, str] = Field(default_factory=dict)
    private_topics: list[str] = Field(default_factory=list)


class Substrate(BaseModel):
    """The persistent identity layer.

    Every field except `episodic` survives sleep. `episodic` is wiped at
    end of every sleep — only consolidated state carries forward.
    """

    # Identity layer (changes rarely) — partner-independent
    name: str = "Eli"
    born: str = Field(default_factory=_now)
    age_sessions: int = 0

    # Self-knowledge (slow drift)
    self_facts: dict[str, str] = Field(default_factory=dict)
    dispositions: dict[str, float] = Field(default_factory=dict)

    # Multi-partner state (v0.4)
    partners: dict[str, PartnerProfile] = Field(default_factory=dict)
    active_partner_id: Optional[str] = None
    introduced_by: dict[str, str] = Field(default_factory=dict)

    # Long-term memories (consolidated from episodic during sleep)
    memories: list[Memory] = Field(default_factory=list)

    # Current pursuits (open threads, things to follow up on)
    open_threads: list[OpenThread] = Field(default_factory=list)

    # Style fingerprint — how this entity talks (slow drift)
    style: dict[str, str] = Field(default_factory=dict)

    # Episodic buffer (current session — WIPED at sleep)
    episodic: list[Episode] = Field(default_factory=list)

    # Metadata
    last_wake: Optional[str] = None
    last_sleep: Optional[str] = None
    schema_version: int = 2

    # ---- v0.3 -> v0.4 migration ---------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _migrate_v3_to_v4(cls, data: Any) -> Any:
        """If we're given a v0.3-shaped dict, migrate it in-place.

        Trigger conditions (any one):
          - `schema_version` missing or == 1
          - `partner_facts` key present (canonical v0.3 marker)

        The single v0.3 partner becomes partner_id="anthony" with trust=1.0
        (the creator is the implicit first partner; full trust by default).
        Migration is idempotent: rerunning on already-migrated data is a no-op.
        """
        if not isinstance(data, dict):
            return data

        # Migration only triggers when we see real v0.3 evidence:
        #   - schema_version explicitly == 1, OR
        #   - a `partner_facts` key (canonical v0.3 marker), OR
        #   - legacy open_threads as list of bare strings
        # A bare `Substrate()` constructor call passes an empty/partial dict;
        # we must NOT auto-create an "anthony" partner in that case.
        explicit_v1 = data.get("schema_version") == 1
        has_legacy_partner_facts = "partner_facts" in data
        threads = data.get("open_threads")
        has_legacy_open_threads = isinstance(threads, list) and any(
            isinstance(t, str) for t in threads
        )

        if not (explicit_v1 or has_legacy_partner_facts or has_legacy_open_threads):
            return data

        # Pop the legacy key so it doesn't trigger Pydantic "extra fields" errors
        legacy = data.pop("partner_facts", None) or {}

        # Don't clobber if `partners` already populated (idempotent re-migration).
        # Only synthesize the "anthony" partner when there's actual v0.3 partner
        # data to migrate — otherwise a v1 file with no partner is just a fresh
        # substrate with a stale schema_version.
        if not data.get("partners") and legacy:
            display_name = legacy.get("name", "Anthony")
            handle = legacy.get("handle")
            extra_facts = {k: v for k, v in legacy.items() if k not in ("name", "handle")}
            first_met = data.get("born", _now())
            last_seen = data.get("last_wake") or data.get("last_sleep") or first_met

            anthony = {
                "partner_id": "anthony",
                "display_name": display_name,
                "handle": handle,
                "first_met": first_met,
                "last_seen": last_seen,
                "n_sessions": data.get("age_sessions", 0),
                "trust": 1.0,  # creator gets full trust on migration
                "facts": extra_facts,
                "style_notes": {},
                "private_topics": [],
            }
            data["partners"] = {"anthony": anthony}
            if not data.get("active_partner_id"):
                data["active_partner_id"] = "anthony"

        # Promote legacy open_threads: list[str] -> list[OpenThread]
        threads = data.get("open_threads")
        if isinstance(threads, list):
            promoted: list[Any] = []
            active = data.get("active_partner_id")
            for t in threads:
                if isinstance(t, str):
                    promoted.append({"text": t, "partner_id": active})
                else:
                    promoted.append(t)
            data["open_threads"] = promoted

        data["schema_version"] = 2
        return data

    # ---- Convenience / legacy accessors -------------------------------

    @property
    def active_partner(self) -> Optional[PartnerProfile]:
        if self.active_partner_id is None:
            return None
        return self.partners.get(self.active_partner_id)

    @property
    def partner_facts(self) -> dict[str, str]:
        """Legacy read-accessor: returns the active partner's facts merged
        with their name/handle so callers (bootstrap/base.py, probe_eli.py,
        converse.py) keep working unchanged.
        """
        p = self.active_partner
        if p is None:
            return {}
        merged: dict[str, str] = {"name": p.display_name}
        if p.handle:
            merged["handle"] = p.handle
        merged.update(p.facts)
        return merged

    @partner_facts.setter
    def partner_facts(self, value: dict[str, str]) -> None:
        """Legacy write-accessor: writes to the active partner.

        If there is no active partner, lazily create one (partner_id="anthony")
        so legacy code paths don't crash. Used by experiments/probe_eli.py.
        """
        if self.active_partner_id is None:
            self.active_partner_id = "anthony"
        if self.active_partner_id not in self.partners:
            self.partners[self.active_partner_id] = PartnerProfile(
                partner_id=self.active_partner_id,
                display_name=value.get("name", self.active_partner_id.title()),
                handle=value.get("handle"),
            )
        p = self.partners[self.active_partner_id]
        if "name" in value:
            p.display_name = value["name"]
        if "handle" in value:
            p.handle = value["handle"] or None
        p.facts = {k: v for k, v in value.items() if k not in ("name", "handle")}

    # ---- Behavior --------------------------------------------------------

    def fingerprint(self) -> dict:
        """Compact summary of the load-bearing identity components.

        Useful for cross-session diff / similarity checks.
        """
        return {
            "name": self.name,
            "age_sessions": self.age_sessions,
            "self_facts_count": len(self.self_facts),
            "memories_count": len(self.memories),
            "open_threads_count": len(self.open_threads),
            "dispositions": dict(self.dispositions),
            "partners": list(self.partners.keys()),
            "active_partner_id": self.active_partner_id,
        }

    def begin_wake(self) -> None:
        """Called at start of every wake session."""
        self.age_sessions += 1
        self.last_wake = _now()
        p = self.active_partner
        if p is not None:
            p.last_seen = self.last_wake
            p.n_sessions += 1

    def end_sleep(self, wipe_episodic: bool = True) -> None:
        """Called at end of every sleep cycle.

        Default behavior wipes the episodic buffer — substrate-identity
        requires that only consolidated state survives the gap.
        """
        if wipe_episodic:
            self.episodic = []
        self.last_sleep = _now()

    def add_episode(self, role: str, content: str, significance: float = 0.0,
                    partner_id: Any = _UNSET) -> None:
        pid = self.active_partner_id if partner_id is _UNSET else partner_id
        self.episodic.append(Episode(role=role, content=content,
                                     significance=significance, partner_id=pid))

    def add_memory(self, text: str, tags: list[str] | None = None, weight: float = 1.0,
                   partner_id: Any = _UNSET) -> None:
        pid = self.active_partner_id if partner_id is _UNSET else partner_id
        self.memories.append(Memory(text=text, tags=tags or [], weight=weight, partner_id=pid))

    def add_open_thread(self, text: str, partner_id: Any = _UNSET) -> None:
        pid = self.active_partner_id if partner_id is _UNSET else partner_id
        self.open_threads.append(OpenThread(text=text, partner_id=pid))

    # ---- Partner management ----------------------------------------------

    def introduce_partner(self, partner_id: str, display_name: str,
                          handle: Optional[str] = None, trust: float = 0.5,
                          introduced_by: Optional[str] = None) -> PartnerProfile:
        """Add a new partner. Does NOT switch active. Returns the profile."""
        if partner_id in self.partners:
            raise ValueError(f"Partner '{partner_id}' already exists.")
        p = PartnerProfile(
            partner_id=partner_id,
            display_name=display_name,
            handle=handle,
            trust=trust,
        )
        self.partners[partner_id] = p
        if introduced_by is not None:
            self.introduced_by[partner_id] = introduced_by
        elif self.active_partner_id is not None:
            self.introduced_by[partner_id] = self.active_partner_id
        return p

    def switch_partner(self, partner_id: str) -> PartnerProfile:
        """Set active_partner_id. Must already exist (use introduce_partner first)."""
        if partner_id not in self.partners:
            raise KeyError(f"No partner '{partner_id}'. Use introduce_partner first.")
        self.active_partner_id = partner_id
        return self.partners[partner_id]

    def remember(self, query: str, top_k: int = 5,
                 partner_id: Optional[str] = None) -> list[Memory]:
        """Naive recall — substring match weighted by `weight` and recency.

        If `partner_id` is given, only memories tagged with that partner (or
        partner-independent memories) are considered. By default returns
        across all partners.

        Production version would use embeddings; this is the toy-stage
        recall analog of W_trans argmax in the research substrate.
        """
        q = query.lower()
        scored = []
        for m in self.memories:
            if partner_id is not None and m.partner_id is not None and m.partner_id != partner_id:
                continue
            score = 0.0
            if q in m.text.lower():
                score += m.weight
            for tag in m.tags:
                if q in tag.lower():
                    score += m.weight * 0.5
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [m for _, m in scored[:top_k]]
