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
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Episode(BaseModel):
    """One significant interaction turn captured during wake.

    Wiped at end of sleep. Anything worth keeping must be consolidated
    into self_facts, partner_facts, memories, or open_threads first.
    """

    timestamp: str = Field(default_factory=_now)
    role: str  # "user" | "agent"
    content: str
    significance: float = 0.0  # 0 (trivial) to 1 (load-bearing)


class Memory(BaseModel):
    """A consolidated long-term memory. Survives sleep wipe."""

    timestamp: str = Field(default_factory=_now)
    text: str
    tags: list[str] = Field(default_factory=list)
    weight: float = 1.0  # decay over time or boost on recall
    source_session: Optional[str] = None


class Substrate(BaseModel):
    """The persistent identity layer.

    Every field except `episodic` survives sleep. `episodic` is wiped at
    end of every sleep — only consolidated state carries forward.
    """

    # Identity layer (changes rarely)
    name: str = "Eli"
    born: str = Field(default_factory=_now)
    age_sessions: int = 0

    # Self-knowledge (slow drift)
    self_facts: dict[str, str] = Field(default_factory=dict)
    dispositions: dict[str, float] = Field(default_factory=dict)

    # Other-knowledge (about the conversation partner)
    partner_facts: dict[str, str] = Field(default_factory=dict)

    # Long-term memories (consolidated from episodic during sleep)
    memories: list[Memory] = Field(default_factory=list)

    # Current pursuits (open threads, things to follow up on)
    open_threads: list[str] = Field(default_factory=list)

    # Style fingerprint — how this entity talks (slow drift)
    style: dict[str, str] = Field(default_factory=dict)

    # Episodic buffer (current session — WIPED at sleep)
    episodic: list[Episode] = Field(default_factory=list)

    # Metadata
    last_wake: Optional[str] = None
    last_sleep: Optional[str] = None
    schema_version: int = 1

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
        }

    def begin_wake(self) -> None:
        """Called at start of every wake session."""
        self.age_sessions += 1
        self.last_wake = _now()

    def end_sleep(self, wipe_episodic: bool = True) -> None:
        """Called at end of every sleep cycle.

        Default behavior wipes the episodic buffer — substrate-identity
        requires that only consolidated state survives the gap.
        """
        if wipe_episodic:
            self.episodic = []
        self.last_sleep = _now()

    def add_episode(self, role: str, content: str, significance: float = 0.0) -> None:
        self.episodic.append(Episode(role=role, content=content, significance=significance))

    def add_memory(self, text: str, tags: list[str] | None = None, weight: float = 1.0) -> None:
        self.memories.append(Memory(text=text, tags=tags or [], weight=weight))

    def remember(self, query: str, top_k: int = 5) -> list[Memory]:
        """Naive recall — substring match weighted by `weight` and recency.

        Production version would use embeddings; this is the toy-stage
        recall analog of W_trans argmax in the research substrate.
        """
        q = query.lower()
        scored = []
        for m in self.memories:
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
