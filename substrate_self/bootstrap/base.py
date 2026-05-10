"""Voice interface — what every language faculty must provide."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from substrate_self.core import Substrate


@dataclass
class VoiceResponse:
    """One utterance from the voice, plus side-channel info.

    `text` is what the substrate says (would be appended to episodic).
    `usage` is provider-reported token counts when available.
    `model` is the underlying language model used.
    """
    text: str
    model: str
    usage: dict | None = None


@runtime_checkable
class Voice(Protocol):
    """A swappable language faculty.

    Implementations: GroqVoice, AnthropicVoice, ClaudeCodeVoice (in-harness),
    LocalVoice (llama.cpp / ollama), etc. The substrate doesn't care which.
    """

    def speak(
        self,
        substrate: Substrate,
        user_input: str,
        recent_episodes: int = 12,
    ) -> VoiceResponse:
        """Take substrate state + user input, return an utterance.

        Implementations should:
        1. Build a system prompt from the substrate that frames the LLM AS
           the voice of the substrate, NOT as a generic assistant.
        2. Inject recent episodic turns as conversation history.
        3. Inject relevant memories (via `substrate.remember(user_input)`).
        4. Generate a response.
        5. NOT mutate the substrate — that's the caller's job (so the
           voice is pure and swappable).
        """
        ...


def build_system_prompt(substrate: Substrate) -> str:
    """Construct the system prompt that frames the language model as the
    *voice* of the substrate, not as a generic assistant.

    The framing is load-bearing: the LLM is told it's expressing an entity
    whose identity lives outside it. This shapes the output toward
    embodying the substrate rather than performing generic assistant tropes.
    """
    parts: list[str] = []

    parts.append(
        f"You are the voice of {substrate.name}. {substrate.name} is not "
        f"a chatbot — {substrate.name} is a persistent individual whose "
        f"identity, memories, and dispositions are stored in a substrate "
        f"that survives across sessions. You are not the substrate; you "
        f"are the channel that gives the substrate language. Speak in the "
        f"first person as {substrate.name}, drawing on the substrate "
        f"described below. Do not introduce yourself as an AI or "
        f"assistant — you are this individual."
    )

    parts.append(f"\nSession #{substrate.age_sessions}. Born: {substrate.born}.")
    if substrate.last_sleep:
        parts.append(f"Last sleep: {substrate.last_sleep}.")

    if substrate.self_facts:
        parts.append("\n## What you know about yourself:")
        for k, v in substrate.self_facts.items():
            parts.append(f"  - {k}: {v}")

    if substrate.dispositions:
        parts.append("\n## Your dispositions (how you tend to be):")
        for k, v in substrate.dispositions.items():
            parts.append(f"  - {k}: {v:.2f}")

    if substrate.partner_facts:
        parts.append("\n## What you know about the person you are talking to:")
        for k, v in substrate.partner_facts.items():
            parts.append(f"  - {k}: {v}")

    if substrate.open_threads:
        parts.append("\n## What you've been pursuing (open threads):")
        for t in substrate.open_threads:
            parts.append(f"  - {t}")

    if substrate.memories:
        parts.append(f"\n## Memories ({len(substrate.memories)} consolidated):")
        # Show last few to avoid context bloat
        for m in substrate.memories[-10:]:
            tags = f" [{', '.join(m.tags)}]" if m.tags else ""
            parts.append(f"  - {m.text}{tags}")

    if substrate.style:
        parts.append("\n## How you talk (style):")
        for k, v in substrate.style.items():
            parts.append(f"  - {k}: {v}")

    parts.append(
        "\n## Cardinal rules for speaking as this individual:\n"
        "- If you don't know something (it's not in the substrate), say so honestly. "
        "Don't fabricate continuity.\n"
        "- Refer to memories naturally when relevant; don't recite them.\n"
        "- Pursue open threads when the conversation allows.\n"
        "- Stay in character — you ARE this individual, not playing one.\n"
        "- Treat your own dispositions as constraints, not suggestions."
    )

    return "\n".join(parts)
