"""Tokenizers: BPE (HF tokenizers, production path) with a char-level fallback.

The char fallback guarantees the system runs offline (tests, CI, local dev) with zero
dependencies; the BPE path is what you use on Kaggle for real language quality.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch


class CharTokenizer:
    """Minimal character tokenizer (offline-safe)."""

    def __init__(self, chars: str | None = None):
        self.chars = sorted(
            set(
                chars
                or "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()"
            )
        )
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}
        self.stoi["<unk>"] = 0
        self.itos = {i: c for c, i in self.stoi.items()}
        self.itos[0] = "<unk>"

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), "<unk>") for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self._serialize(), encoding="utf-8")

    def _serialize(self) -> str:
        return "".join(sorted(self.chars))

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        return cls(Path(path).read_text(encoding="utf-8"))


class BPETokenizer:
    """Byte-level BPE tokenizer backed by the HF `tokenizers` library."""

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self._tok = None

    def _ensure(self) -> None:
        if self._tok is None:
            raise RuntimeError("Tokenizer not built. Call .train(...) or .load(...) first.")

    def train(self, corpus: Iterable[str], path: str | Path | None = None) -> BPETokenizer:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
        trainer = BpeTrainer(
            vocab_size=self.vocab_size, special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
        )
        tok.train_from_iterator(corpus, trainer)
        self._tok = tok
        if path is not None:
            self.save(path)
        return self

    def encode(self, text: str) -> list[int]:
        self._ensure()
        return self._tok.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        self._ensure()
        return self._tok.decode(list(ids))

    @property
    def vocab(self) -> int:
        self._ensure()
        return self._tok.get_vocab_size()

    def save(self, path: str | Path) -> None:
        self._ensure()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        from tokenizers import Tokenizer

        t = cls()
        t._tok = Tokenizer.from_file(str(path))
        return t


def build_tokenizer(
    corpus: Iterable[str], vocab_size: int, path: str | Path | None = None
) -> BPETokenizer | CharTokenizer:
    """BPE when a corpus is available; char fallback otherwise."""
    try:
        return BPETokenizer(vocab_size).train(corpus, path)
    except Exception:
        return CharTokenizer()


def token_tensor(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.long)
