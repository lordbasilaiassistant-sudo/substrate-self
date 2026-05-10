"""Character-level tokenizer.

Picked char-level for two reasons:
  1. Zero external dependencies — no sentencepiece, no tokenizers crate
  2. Maps directly onto the BetterThanLLM substrate primitive (each char
     is a "flavor"; transitions are char-to-char)

Byte-level (256 fixed vocab) is the simplest viable; we instead fit on
the corpus to get a smaller vocab of just the chars actually used. This
is what nanoGPT-tinyShakespeare does and produces sharper representations
on small models.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable


class CharTokenizer:
    """Maps characters ↔ integer ids."""

    def __init__(self, vocab: list[str] | None = None):
        self.vocab: list[str] = vocab or []
        self._stoi: dict[str, int] = {c: i for i, c in enumerate(self.vocab)}
        # Fall back to <unk> id if present in vocab; else 0 to keep us in-range
        self._unk = self._stoi.get("<unk>", 0 if self.vocab else -1)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, texts: Iterable[str]) -> "CharTokenizer":
        """Build vocab from corpus."""
        chars = set()
        for t in texts:
            chars.update(t)
        # Special tokens at the start so they have stable ids
        special = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.vocab = special + sorted(chars - set(special))
        self._stoi = {c: i for i, c in enumerate(self.vocab)}
        self._unk = self._stoi["<unk>"]
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self._stoi["<bos>"])
        for c in text:
            ids.append(self._stoi.get(c, self._unk))
        if add_eos:
            ids.append(self._stoi["<eos>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if 0 <= i < len(self.vocab):
                tok = self.vocab[i]
                if tok in ("<pad>", "<bos>", "<eos>", "<unk>"):
                    continue
                out.append(tok)
        return "".join(out)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps({"vocab": self.vocab}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab=data["vocab"])
