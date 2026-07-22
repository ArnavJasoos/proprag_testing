"""Headless smoke test: index a tiny in-memory corpus, run one query, then run the
three-system comparison (BaseRAG / GraphRAG / PropRAG) and print per-system metrics.

Default chat backend is the NVIDIA free API: set ``NVIDIA_API_KEY`` in a ``.env``
at the repo root. Chat calls are rate-limited (default 38 RPM). Embeddings run
locally (sentence-transformers), so only the chat calls are throttled. Run from
the repo root:

    python -m proprag_poc.scripts.smoke_test
"""

from __future__ import annotations

import sys

from ..config import POCConfig
from ..core.index import Engine
from ..logging_setup import setup_logging

SAMPLE_TEXTS = [
    "M1 Chip\nIn 2020 Apple launched the M1 chip. Adobe optimized its applications "
    "for the M1 chip, improving performance by up to 80% compared to Intel-based Macs.",
    "iPhone 15\nIn September 2023 Apple replaced the Lightning connector with USB-C on "
    "the iPhone 15 after the European Union required a standardized charging port.",
]


def main() -> int:
    setup_logging()
    config = POCConfig(force_index_from_scratch=True)
    engine = Engine(config)
    corpus_id = "smoke"
    engine._corpora[corpus_id] = engine.indexer.build_from_texts(corpus_id, SAMPLE_TEXTS)

    question = "Why did Apple switch the iPhone 15 to USB-C?"
    result = engine.ask(corpus_id, question, history=[])
    print("Q:", question)
    print("A:", result.answer)
    print("Top passages:")
    for p in result.passages:
        print(f"  [{p.score:.4f}] {p.text[:90]}...")

    print("\n=== Comparison (BaseRAG / GraphRAG / PropRAG) ===")
    comparison = engine.compare(corpus_id, question, history=[])
    for sr in comparison.systems:
        u = sr.usage
        if sr.error:
            print(f"\n[{sr.system}] ERROR: {sr.error}")
            continue
        print(
            f"\n[{sr.system}] retrieval={sr.retrieval_latency_s*1000:.0f}ms "
            f"qa={sr.qa_latency_s*1000:.0f}ms | chat_calls={u.chat_calls} "
            f"chat_tokens={u.chat_total_tokens} embed_calls={u.embed_calls} "
            f"embed_tokens={u.embed_tokens}"
        )
        print(f"  A: {sr.answer[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
