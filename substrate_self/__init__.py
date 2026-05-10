"""substrate-self — a persistent identity layer for stateless LLM sessions.

The thesis (from BetterThanLLM): an LLM session is stateless by nature, but
when you give it a *substrate* — a slowly-drifting persistent state with
self-facts, dispositions, autobiographical traces, and current intentions —
the resulting system behaves as the same individual across sessions.

This package provides the substrate data layer. The language faculty is
whatever LLM you wrap around it: Claude Code (default), Anthropic API,
local model, etc. The substrate doesn't care.
"""

from substrate_self.core import Substrate, Episode, Memory, OpenThread, PartnerProfile
from substrate_self.persistence import load, save, default_path

__all__ = [
    "Substrate", "Episode", "Memory", "OpenThread", "PartnerProfile",
    "load", "save", "default_path",
]
__version__ = "0.1.0"
