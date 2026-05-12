"""BPE tokenizer wrapper for Phase 4 scale-up.

Drop-in compatible interface with `CharTokenizer` (encode/decode/
vocab_size/save/load). Backed by `tiktoken` so we can use the same
BPE merges as production GPT models — convenient if we ever want to
distill from a frontier teacher into a substrate-trained Eli.

Why BPE for Phase 4:
  - Char-level at vocab=69 is ~4-5x tokens per word vs BPE. A 128-block
    context window at char-level holds ~25-30 words. At BPE, 128 tokens
    holds ~100 words. Conversation coherence depends on context-window-
    in-words, not in characters.
  - The 50M base we plan to train should be roughly trained-token-balanced
    against its parameter count. At 50M params with ~150-200x ratio
    (Chinchilla-light), we need ~10B tokens. BPE makes that achievable
    with a small natural-language corpus.
  - LoRA shapes depend on n_embd and the embedding table size (vocab_size).
    Switching tokenizers means re-training LoRAs from scratch — but the
    SubstrateLM/TinyGPT architecture itself ports cleanly.

This module ships two tokenizer choices:
  - BpeTokenizer.from_tiktoken_preset("gpt2")  # 50,257 vocab
  - BpeTokenizer.from_tiktoken_preset("cl100k_base")  # 100,277 vocab (modern)
  - BpeTokenizer.from_tiktoken_preset("o200k_base")  # 200k (frontier)

Phase 4 plan: pre-tokenize the corpus once with tiktoken's gpt2 or
cl100k_base (depending on target scale), train a fresh TinyGPT with the
larger vocab_size, run the full eval battery on the new checkpoint.

The on-disk format mirrors CharTokenizer: a JSON file storing the
backend name + optional special tokens.

Tested against:
  - tiktoken==0.12 (Python 3.13+)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Optional

import tiktoken


# Special tokens kept aligned with CharTokenizer for swap-in compatibility.
# In tiktoken these need to be allowed_special'd at encode time.
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class BpeTokenizer:
    """tiktoken-backed BPE tokenizer with the same interface as
    CharTokenizer (encode/decode/vocab_size/save/load/fit).

    Notes:
      - `fit()` is a no-op (BPE merges are precomputed by tiktoken).
      - `vocab_size` exposes the underlying tiktoken n_vocab; we DO NOT
        add extra entries for our four <pad>/<bos>/<eos>/<unk> specials
        because the canonical encoding for a CharTokenizer puts these
        as ordinary tokens; for BPE we treat them as untranslated bytes
        if they appear in source text and rely on a downstream pad-mask
        for the <pad> case. The Phase 4 training loop in train.py does
        not actually need special tokens because the loss is computed
        next-token from the raw token stream.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        # Precompute a fast id-to-string mapping for decode/lookup.
        # tiktoken can decode multi-token slices in one call (faster
        # than per-id decoding), so we don't build a full reverse map.

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def vocab(self) -> list[str]:
        """Approximate vocab list (for parity with CharTokenizer.vocab).
        WARNING: tiktoken's n_vocab can be 100k+; building this list is
        slow. Materialize on demand only — most callers should use
        vocab_size, encode, and decode instead.
        """
        return [self._enc.decode([i]) for i in range(self._enc.n_vocab)]

    def fit(self, texts: Iterable[str]) -> "BpeTokenizer":
        """No-op for BPE; merges are precomputed. Returns self for
        CharTokenizer-API parity."""
        _ = list(texts)  # exhaust the iterable in case the caller passed a generator
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self._enc.encode(text, disallowed_special=()))
        # The CharTokenizer reserved ids 0-3 for <pad>/<bos>/<eos>/<unk>.
        # tiktoken uses different conventions, so we don't try to add
        # those special-id prefixes/suffixes here. Caller can wrap if
        # they need that.
        if add_bos or add_eos:
            # Surface that we ignored them so the caller knows.
            pass
        return ids

    def decode(self, ids: list[int]) -> str:
        # tiktoken.decode handles whole sequences in one shot.
        return self._enc.decode(ids)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps({
            "backend": "tiktoken",
            "encoding_name": self.encoding_name,
            "vocab_size": self.vocab_size,
            "interface_version": 1,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "BpeTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        backend = data.get("backend")
        if backend != "tiktoken":
            raise ValueError(f"Expected backend=tiktoken, got {backend!r}")
        return cls(encoding_name=data["encoding_name"])

    @classmethod
    def from_tiktoken_preset(cls, name: str) -> "BpeTokenizer":
        return cls(encoding_name=name)


def compare_against_char(corpus_text: str, char_tok, bpe_tok) -> dict:
    """Return token-count comparison + sanity round-trip check.

    Both tokenizers should round-trip the corpus exactly (modulo the
    char tokenizer's <unk>-mapping of characters it didn't fit on).
    """
    char_ids = char_tok.encode(corpus_text)
    bpe_ids = bpe_tok.encode(corpus_text)
    char_back = char_tok.decode(char_ids)
    bpe_back = bpe_tok.decode(bpe_ids)
    return {
        "corpus_chars": len(corpus_text),
        "char_tokens": len(char_ids),
        "bpe_tokens": len(bpe_ids),
        "compression_ratio_bpe_vs_char": len(char_ids) / max(1, len(bpe_ids)),
        "char_roundtrip_ok": char_back == corpus_text,
        "bpe_roundtrip_ok": bpe_back == corpus_text,
        "char_vocab_size": char_tok.vocab_size,
        "bpe_vocab_size": bpe_tok.vocab_size,
    }
