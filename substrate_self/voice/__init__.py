"""Voice — language faculty that gives the substrate expression.

The substrate is the self: persistent state, identity, memories, dispositions.
Voice is the channel: a (possibly stateless) language model that takes the
substrate's current state plus the user's input and produces an utterance.

The architectural commitment: the language faculty is *swappable*. The
substrate doesn't change when the voice changes. You can run the same
substrate through Claude Code, Groq, Anthropic API, a local model, or a
custom small model — the entity stays the same; only its accent changes.

This is the difference between "LLM with substrate as RAG context" and
"substrate with LLM as voice." In the former, the LLM is the self and the
substrate is a memory aid. In the latter, the substrate is the self and
the LLM is a tongue. We commit to the second framing.
"""

from substrate_self.voice.base import Voice, VoiceResponse

__all__ = ["Voice", "VoiceResponse"]
