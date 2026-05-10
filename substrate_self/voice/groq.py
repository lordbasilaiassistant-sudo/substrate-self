"""Groq-backed Voice implementation.

Why Groq: fast, cheap, no Anthropic-API-credit dependency, gives us a
language faculty we can wire to the substrate without it BECOMING the
substrate. The substrate stays in the JSON file; Groq is just the tongue.

Default model: llama-3.3-70b-versatile (capable instruction-tuned, good
balance of cost and quality). Override via `model=` constructor arg or
`SUBSTRATE_VOICE_MODEL` env var.

Requires GROQ_API_KEY in environment. `pip install groq`.
"""

from __future__ import annotations
import os
from typing import Optional

from substrate_self.core import Substrate
from substrate_self.voice.base import Voice, VoiceResponse, build_system_prompt


class GroqVoice:
    """Voice backed by Groq's hosted language models."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError(
                "groq SDK not installed. Run: pip install groq"
            ) from e

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "No GROQ_API_KEY in environment and none passed to GroqVoice. "
                "Get a free API key at https://console.groq.com/keys"
            )

        self._client = Groq(api_key=key)
        self.model = (
            model
            or os.environ.get("SUBSTRATE_VOICE_MODEL")
            or self.DEFAULT_MODEL
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def speak(
        self,
        substrate: Substrate,
        user_input: str,
        recent_episodes: int = 12,
    ) -> VoiceResponse:
        system_prompt = build_system_prompt(substrate)

        # Include relevant consolidated memories surfaced by the user input.
        # The system prompt already lists last 10 memories; this is targeted recall.
        recalled = substrate.remember(user_input, top_k=3)
        if recalled:
            recall_block = "\n\n## Memories that may be especially relevant to this turn:\n"
            for m in recalled:
                recall_block += f"  - {m.text}\n"
            system_prompt += recall_block

        # Build messages from recent episodic, then user input
        messages = [{"role": "system", "content": system_prompt}]
        for ep in substrate.episodic[-recent_episodes:]:
            role = "assistant" if ep.role == "agent" else "user"
            messages.append({"role": role, "content": ep.content})
        messages.append({"role": "user", "content": user_input})

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        text = completion.choices[0].message.content or ""
        usage = (
            {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
            if completion.usage
            else None
        )

        return VoiceResponse(text=text, model=self.model, usage=usage)
