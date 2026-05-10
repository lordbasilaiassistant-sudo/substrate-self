"""End-to-end interactive loop: load substrate → converse via Voice → save.

Run with:
  py -m substrate_self.converse                  (uses GroqVoice by default)
  py -m substrate_self.converse --voice groq
  py -m substrate_self.converse --voice anthropic  (when implemented)

This module is for users who want to talk to their substrate from the
command line without going through Claude Code. Inside Claude Code, the
substrate-self skill handles the same flow without any external API.
"""

from __future__ import annotations
import argparse
import sys
from substrate_self import core, persistence


def get_voice(name: str):
    if name == "groq":
        from substrate_self.voice.groq import GroqVoice
        return GroqVoice()
    raise ValueError(f"Unknown voice: {name}. Available: groq")


def main():
    parser = argparse.ArgumentParser(description="Talk to your substrate.")
    parser.add_argument("--voice", default="groq", help="Language faculty (default: groq)")
    parser.add_argument("--model", default=None, help="Override model id")
    parser.add_argument("--no-save", action="store_true", help="Don't persist substrate changes")
    args = parser.parse_args()

    voice = get_voice(args.voice)
    if args.model:
        voice.model = args.model

    s = persistence.load()
    s.begin_wake()
    print(f"=== Substrate loaded ===")
    print(f"Name: {s.name}")
    print(f"Session: #{s.age_sessions}")
    print(f"Voice: {voice.model}")
    print(f"Memories: {len(s.memories)}, partner_facts: {len(s.partner_facts)}, threads: {len(s.open_threads)}")
    print(f"=== Ready. Type 'sleep' to consolidate, 'quit' to exit without consolidation. ===\n")

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("(exited without sleeping — episodic preserved for next session)")
                break
            if user_input.lower() in ("sleep", "consolidate"):
                # The CLI doesn't do real LLM-driven consolidation.
                # That's the skill's job inside Claude Code, or a separate
                # tool that calls back through the voice. For now, just wipe.
                n = len(s.episodic)
                s.end_sleep(wipe_episodic=True)
                print(f"(slept — wiped {n} episodic entries; consolidation should be done via the substrate-self Claude Code skill or a custom tool)")
                break

            s.add_episode("user", user_input, significance=0.0)
            response = voice.speak(s, user_input)
            print(f"{s.name}> {response.text}\n")
            s.add_episode("agent", response.text, significance=0.0)
    finally:
        if not args.no_save:
            persistence.save(s)
            print(f"(substrate saved)")


if __name__ == "__main__":
    main()
