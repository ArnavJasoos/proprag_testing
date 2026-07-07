"""End-to-end smoke test on 4 tiny docs + 2 questions.

The Koboldcpp-up verification gate: exercises all three indexes, the query loop
and the report in a handful of LLM calls, and asserts the failure modes we care
about most (JSON parse, reasoning leakage, Answer: extraction).

    python -m benchmark.smoke
"""

from __future__ import annotations

import json
import logging
import os

from . import _bootstrap  # noqa: F401
from proprag_poc.core.metrics import get_usage_tracker
from proprag_poc.embedding.encoder import EmbeddingModel

from .bench_config import BenchmarkConfig
from .dataset import BenchQuestion
from .llm_wrap import BenchLLMClient, check_backend
from .report import build as build_report
from .results import ResultsStore
from .run import _build_retrievers, _index_all, _question_loop, _setup_logging

logger = logging.getLogger("benchmark")

_DOCS = [
    ("Ada Lovelace", "Ada Lovelace was an English mathematician known for her work on "
                     "Charles Babbage's Analytical Engine. She is regarded as the first "
                     "computer programmer."),
    ("Charles Babbage", "Charles Babbage was an English mathematician who originated the "
                        "concept of a programmable computer, the Analytical Engine."),
    ("Analytical Engine", "The Analytical Engine was a proposed mechanical general-purpose "
                         "computer designed by Charles Babbage in the 1830s."),
    ("Alan Turing", "Alan Turing was an English mathematician who formalized the concepts of "
                   "algorithm and computation with the Turing machine."),
]

_QUESTIONS = [
    BenchQuestion(
        qid="smoke-1", qtype="compositional",
        question="Who designed the machine that Ada Lovelace is known for working on?",
        answer="Charles Babbage",
        gold_titles=["Ada Lovelace", "Analytical Engine", "Charles Babbage"],
        context_titles=["Ada Lovelace", "Analytical Engine", "Charles Babbage"],
    ),
    BenchQuestion(
        qid="smoke-2", qtype="comparison",
        question="Were Ada Lovelace and Alan Turing both English mathematicians?",
        answer="yes",
        gold_titles=["Ada Lovelace", "Alan Turing"],
        context_titles=["Ada Lovelace", "Alan Turing"],
    ),
]


def main() -> None:
    _setup_logging()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = BenchmarkConfig(project_dir=project_dir, n_questions=len(_QUESTIONS), seed=42)
    poc_cfg = cfg.make_poc_config()
    check_backend(poc_cfg)

    emb = EmbeddingModel(poc_cfg)
    llm = BenchLLMClient(poc_cfg)
    tracker = get_usage_tracker()

    run_dir = os.path.join(cfg.data_dir, "benchmark", "smoke")
    os.makedirs(run_dir, exist_ok=True)
    systems = ["BaseRAG", "GraphRAG", "PropRAG"]
    corpus_ident = "2wiki_smoke"

    built = _index_all(cfg, poc_cfg, corpus_ident, _DOCS, emb, llm, tracker, run_dir, force=True)

    assert built["gr_index"].entities, "GraphRAG extraction produced no entities"
    assert len(built["index_usage"]) == 3, "expected 3 index_usage entries"

    retrievers = _build_retrievers(built, emb, poc_cfg, cfg, systems)
    store = ResultsStore(run_dir)
    # Fresh results for a clean assertion.
    if os.path.isfile(store.path):
        os.remove(store.path)
    _question_loop(_QUESTIONS, retrievers, systems, cfg, poc_cfg, llm, tracker, store,
                   built["chunk_id_to_title"])

    rows = [json.loads(l) for l in open(store.path, encoding="utf-8") if l.strip()]
    assert rows, "no result rows written"
    for row in rows:
        assert not row.get("error"), f"row errored: {row.get('error')}"
        ans = row["answer"]
        assert ans and ans != "unknown", f"empty/unknown answer for {row['qid']}/{row['system']}"
        assert "<|" not in ans and "analysis" not in ans[:20].lower(), \
            f"reasoning leaked into answer: {ans!r}"

    build_report(run_dir, make_charts=False)
    assert os.path.isfile(os.path.join(run_dir, "report.md")), "report.md not written"
    print("SMOKE OK — 3 indexes built, answers parsed, report written:",
          os.path.join(run_dir, "report.md"))


if __name__ == "__main__":
    main()
