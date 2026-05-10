"""Generate training corpus for the substrate's from-scratch language model.

The teacher (an LLM) is asked to write dialogue exchanges where the agent
(the substrate-self entity) speaks naturally. The corpus is then used to
train a small from-scratch model that learns to speak like this entity.

Usage:
  from substrate_self.teach import generate_corpus
  examples = generate_corpus(n=100, save_path='corpus.jsonl')
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from substrate_self.core import Substrate
from substrate_self import persistence


PROMPTS = [
    "Generate a realistic short dialogue (3-5 exchanges) where {name} is helping the user with a coding problem. {name} should sound like the substrate describes — not a generic AI assistant. Output only the dialogue, formatted as alternating 'User: ...' and '{name}: ...' lines.",
    "Generate a short dialogue where {name} is reflecting on something they remember about a past project. Use details from the memories in the substrate. Format as alternating User/{name} lines.",
    "Generate a short dialogue where the user asks {name} a question about themselves (their preferences, dispositions, what they care about). {name} answers from the substrate. Format as alternating User/{name} lines.",
    "Generate a short dialogue where {name} brings up an open thread they've been meaning to follow up on. The user responds. Format as alternating User/{name} lines.",
    "Generate a short dialogue where {name} disagrees politely with the user about a technical choice. {name}'s reasoning should reflect their dispositions. Format as alternating User/{name} lines.",
    "Generate a single thoughtful monologue (3-6 sentences) where {name} is thinking out loud about a problem. Voice and dispositions from the substrate. No dialogue formatting — just the monologue.",
    "Generate a short exchange (2-3 turns) where the user asks 'how are you' and {name} answers honestly based on the substrate state. Format as alternating User/{name} lines.",
]


@dataclass
class CorpusExample:
    """One training example: a dialogue or monologue produced by a teacher."""

    text: str
    teacher_model: str
    prompt_kind: str  # which PROMPTS template generated this
    substrate_age_at_generation: int

    def as_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def generate_corpus(
    n: int = 50,
    save_path: Optional[Path | str] = None,
    teacher_voice=None,
    substrate: Optional[Substrate] = None,
    verbose: bool = True,
) -> list[CorpusExample]:
    """Generate `n` training examples by asking a teacher LLM to produce
    substrate-conditioned dialogue. Saves to JSONL if `save_path` given.

    Args:
        n: number of examples to generate
        save_path: optional path to write JSONL
        teacher_voice: a Voice instance. Defaults to GroqVoice.
        substrate: substrate to condition on. Defaults to loaded substrate.
        verbose: print progress

    Returns:
        list of CorpusExample
    """
    if substrate is None:
        substrate = persistence.load()

    if teacher_voice is None:
        from substrate_self.bootstrap.groq import GroqVoice
        teacher_voice = GroqVoice()

    examples: list[CorpusExample] = []
    save_p = Path(save_path) if save_path else None
    if save_p:
        save_p.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so partial generations aren't lost
        out = save_p.open("a", encoding="utf-8")
    else:
        out = None

    try:
        for i in range(n):
            prompt_template = PROMPTS[i % len(PROMPTS)]
            prompt = prompt_template.format(name=substrate.name)
            response = teacher_voice.speak(substrate, prompt, recent_episodes=0)
            ex = CorpusExample(
                text=response.text.strip(),
                teacher_model=response.model,
                prompt_kind=prompt_template[:60],
                substrate_age_at_generation=substrate.age_sessions,
            )
            examples.append(ex)
            if out:
                out.write(ex.as_jsonl() + "\n")
                out.flush()
            if verbose:
                print(f"  [{i+1}/{n}] {len(ex.text)} chars from {response.model}")
    finally:
        if out:
            out.close()

    if verbose and save_p:
        total_chars = sum(len(e.text) for e in examples)
        print(f"\nWrote {len(examples)} examples ({total_chars} chars) to {save_p}")

    return examples
