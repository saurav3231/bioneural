"""Data: high-quality standard datasets (TinyStories / WikiText-2) with an offline-safe
synthetic fallback so tests and local dev always run."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

WORD_POOL = [
    "once upon a time",
    "the little",
    "boy and his",
    "dog went to",
    "the park where",
    "they found a",
    "big red ball",
    "mama bear said",
    "never talk to",
    "strangers in the",
    "woods at night",
    "the sun was",
    "shining bright and",
    "the birds sang",
    "a happy song",
    "then the rain",
    "started to fall",
    "so they ran",
    "under the tree",
    "and waited for",
    "the storm to",
    "pass by slowly",
    "grandma told stories",
    "about the old",
    "wise owl who",
    "lived in the",
    "tall oak tree",
    "and knew everything",
    "about the forest",
    "the end",
]


def _synthetic_corpus(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        k = rng.randint(20, 60)
        words = [rng.choice(WORD_POOL) for _ in range(k)]
        texts.append(" ".join(words) + ".")
    return texts


def load_dataset(
    name: str = "tiny-stories",
    max_examples: int = 2000,
    seed: int = 0,
) -> list[str]:
    """Load a text corpus. `name` in {"tiny-stories", "wikitext-2", "synthetic"}.

    Falls back to the synthetic corpus on any network/parse failure so the pipeline never
    blocks on infrastructure.
    """
    try:
        import datasets  # noqa: F401
    except Exception:
        return _synthetic_corpus(max_examples, seed)

    try:
        if name == "tiny-stories":
            ds = datasets.load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            texts = [ex["text"] for _, ex in zip(range(max_examples), ds, strict=False)]
        elif name == "wikitext-2":
            ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds["text"][: max_examples * 4] if t.strip()][:max_examples]
        else:
            return _synthetic_corpus(max_examples, seed)
        if not texts:
            return _synthetic_corpus(max_examples, seed)
        return texts
    except Exception:
        return _synthetic_corpus(max_examples, seed)


def iter_batches(texts: Sequence[str], batch_size: int, seed: int = 0) -> Iterator[list[str]]:
    """Yield batches of texts (in-order shuffling for reproducibility)."""
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [texts[j] for j in idx[i : i + batch_size]]
