"""Manual smoke test for CodeEmbedder -- run directly, no pytest needed.

    python smoke_test_embedder.py

Uses a fake tokenizer/model (same pattern as the unit tests) so it runs
offline in seconds. Walks through every feature and prints what happened,
so you can eyeball correctness instead of just seeing green/red dots.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from src.reporag.embedding.code_embedder import EMBEDDING_DIM, CodeEmbedder

PASS = "[PASS]"
FAIL = "[FAIL]"
failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"{PASS if condition else FAIL} {label}")
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------------
# Fake tokenizer / model -- identical shape to what real HF classes return
# ---------------------------------------------------------------------------


class FakeTokenizer:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(
        self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ):
        self.calls += 1
        seqs = [
            [(ord(c) % (self.vocab_size - 2)) + 2 for c in t[:max_length]] or [2]
            for t in texts
        ]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s)
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.forward_calls += 1
        return FakeOutput(self.embed(input_ids))


CODE_SAMPLES = [
    "def foo():\n    pass",
    "class Bar:\n    pass",
    "x = 1 + 2",
    "import os",
    "return self.value",
]


def main() -> None:
    print("=" * 70)
    print("1. Basic embedding: shape, dtype, L2 normalization")
    print("=" * 70)
    model, tok = FakeModel(), FakeTokenizer()
    embedder = CodeEmbedder(model=model, tokenizer=tok, batch_size=2)

    vectors = embedder.embed_batch(CODE_SAMPLES)
    check(
        f"shape == ({len(CODE_SAMPLES)}, {EMBEDDING_DIM})",
        vectors.shape == (len(CODE_SAMPLES), EMBEDDING_DIM),
    )
    check("dtype == float32", vectors.dtype == np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    check(
        f"all rows L2-normalized (norms={np.round(norms, 4).tolist()})",
        np.allclose(norms, 1.0, atol=1e-5),
    )

    single = embedder.embed(CODE_SAMPLES[0])
    check(
        f"embed() single vector shape == ({EMBEDDING_DIM},)",
        single.shape == (EMBEDDING_DIM,),
    )

    print()
    print("=" * 70)
    print("2. Configurable batch size (5 items, batch_size=2 -> 3 forward passes)")
    print("=" * 70)
    model2, tok2 = FakeModel(), FakeTokenizer()
    e2 = CodeEmbedder(model=model2, tokenizer=tok2, batch_size=2)
    e2.embed_batch(CODE_SAMPLES)
    check(f"forward_calls == 3 (got {model2.forward_calls})", model2.forward_calls == 3)

    print()
    print("=" * 70)
    print("3. Caching: repeat call avoids recomputation, stats track hits/misses")
    print("=" * 70)
    print(f"cache_stats before: {embedder.cache_stats()}")
    calls_before = model.forward_calls
    embedder.embed_batch(CODE_SAMPLES)  # should be all cache hits now
    check(
        f"forward_calls unchanged on repeat ({calls_before} -> {model.forward_calls})",
        model.forward_calls == calls_before,
    )
    stats = embedder.cache_stats()
    print(f"cache_stats after repeat: {stats}")
    check("hits > 0 after repeat call", stats["hits"] > 0)
    check(
        f"cache_size == {len(CODE_SAMPLES)}", embedder.cache_size() == len(CODE_SAMPLES)
    )

    dup_vectors = embedder.embed_batch([CODE_SAMPLES[0], CODE_SAMPLES[0]])
    check(
        "duplicate items in one batch return identical vectors",
        np.array_equal(dup_vectors[0], dup_vectors[1]),
    )

    check("__contains__ finds cached item", CODE_SAMPLES[0] in embedder)
    check(
        "__contains__ False for unseen item",
        "never seen this text before" not in embedder,
    )

    embedder.clear_cache()
    check(
        "clear_cache resets size and stats",
        embedder.cache_size() == 0
        and embedder.cache_stats() == {"hits": 0, "misses": 0, "size": 0},
    )

    print()
    print("=" * 70)
    print("4. Cache key includes model name (no cross-model collisions)")
    print("=" * 70)
    shared_cache: dict = {}
    model_a, tok_a = FakeModel(), FakeTokenizer()
    ea = CodeEmbedder(
        model_name="model_A", model=model_a, tokenizer=tok_a, cache=shared_cache
    )
    eb = CodeEmbedder(
        model_name="model_B", model=model_a, tokenizer=tok_a, cache=shared_cache
    )
    ea.embed("print(1)")
    eb.embed("print(1)")
    check(
        f"same text under 2 model names -> 2 cache entries (got {len(shared_cache)})",
        len(shared_cache) == 2,
    )

    print()
    print("=" * 70)
    print("5. Device resolution: auto / cuda / mps / cpu")
    print("=" * 70)
    print(f"CUDA available on this machine: {torch.cuda.is_available()}")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available on this machine: {mps_available}")

    e_auto = CodeEmbedder(model=FakeModel(), tokenizer=FakeTokenizer(), device="auto")
    print(f"device='auto'  -> resolved to '{e_auto.device}'")
    e_cpu = CodeEmbedder(model=FakeModel(), tokenizer=FakeTokenizer(), device="cpu")
    check(f"device='cpu' forces cpu (got '{e_cpu.device}')", e_cpu.device == "cpu")

    try:
        CodeEmbedder(device="tpu")  # type: ignore[arg-type]
        check("invalid device raises ValueError", False)
    except ValueError:
        check("invalid device raises ValueError", True)

    print()
    print("=" * 70)
    print("6. similarity()")
    print("=" * 70)
    model3, tok3 = FakeModel(), FakeTokenizer()
    e3 = CodeEmbedder(model=model3, tokenizer=tok3)
    sim_same = e3.similarity(CODE_SAMPLES[0], CODE_SAMPLES[0])
    sim_diff = e3.similarity(CODE_SAMPLES[0], CODE_SAMPLES[1])
    print(f"similarity(x, x) = {sim_same:.4f}   similarity(x, y) = {sim_diff:.4f}")
    check("identical text has similarity ~1.0", abs(sim_same - 1.0) < 1e-4)

    zero_vecs_embedder = CodeEmbedder(model=model3, tokenizer=tok3)
    zero_vecs_embedder.embed_batch = lambda items, **kw: np.zeros((len(items), EMBEDDING_DIM), dtype=np.float32)  # type: ignore
    check(
        "zero-vector inputs -> similarity 0.0 (no div/0)",
        zero_vecs_embedder.similarity("a", "b") == 0.0,
    )

    print()
    print("=" * 70)
    print("7. Empty input never touches the model")
    print("=" * 70)
    model4, tok4 = FakeModel(), FakeTokenizer()
    e4 = CodeEmbedder(model=model4, tokenizer=tok4)
    empty = e4.embed_batch([])
    check(
        f"empty batch shape == (0, {EMBEDDING_DIM})", empty.shape == (0, EMBEDDING_DIM)
    )
    check("empty batch never called the model", model4.forward_calls == 0)

    print()
    print("=" * 70)
    if failures:
        print(f"{FAIL} {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    else:
        print(f"{PASS} All checks passed.")


if __name__ == "__main__":
    main()
