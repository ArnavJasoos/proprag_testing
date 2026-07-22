"""Run BaseRAG, GraphRAG and PropRAG on one query and collect per-system metrics.

Each system runs inside its own ``UsageTracker`` scope so token consumption, call
counts and rate-limit waits are attributed correctly. Retrieval and QA latency are
measured here as wall-clock around the calls (independent of the tracker's
per-call latency, which excludes orchestration overhead).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .metrics import Snapshot, get_usage_tracker
from .qa import QAResult
from .retriever import RetrievedPassage

logger = logging.getLogger(__name__)


@dataclass
class SystemResult:
    system: str
    answer: str
    passages: List[RetrievedPassage]
    retrieval_latency_s: float
    qa_latency_s: float
    usage: Snapshot
    error: Optional[str] = None

    @property
    def total_latency_s(self) -> float:
        return self.retrieval_latency_s + self.qa_latency_s


@dataclass
class ComparisonResult:
    query: str
    search_query: str
    # Shared controls (the fairness levers), surfaced for transparency.
    embedding_model: str = ""
    llm_model: str = ""
    qa_top_k: int = 0
    ablation: bool = False
    systems: List[SystemResult] = field(default_factory=list)


def run_comparison(engine, corpus_id: str, query: str,
                   history: Optional[List[Dict[str, str]]] = None,
                   ablation: bool = False) -> ComparisonResult:
    """Run all three retrievers + QA on ``query``; return per-system results + metrics.

    Fairness: every system shares one embedding model, one generation LLM, the same
    contextualized query, and the same number of passages fed to the generator
    (``qa_top_k``). Only the *retrieval strategy* differs — that is the experiment.

    ``ablation=True`` collapses all three to chunk-vector dense retrieval (one shared
    embedding artifact). This is deliberately NOT a framework comparison: it strips
    the index differences that define GraphRAG/PropRAG, so the systems converge. Use
    it to see algorithm-only behaviour, clearly separate from the real benchmark.
    """
    history = history or []
    tracker = get_usage_tracker()

    # Contextualize once (shared) so every system retrieves on the same query.
    logger.info("compare: query=%r ablation=%s", query, ablation)
    search_query = engine.contextualizer.contextualize(query, history)

    result = ComparisonResult(
        query=query,
        search_query=search_query,
        embedding_model=engine.config.embedding_model,
        llm_model=engine.config.llm_model,
        qa_top_k=engine.config.qa_top_k,
        ablation=ablation,
    )
    systems = (
        engine.ablation_retrievers(corpus_id) if ablation
        else engine.comparison_retrievers(corpus_id)
    )
    for idx, (system, retriever) in enumerate(systems.items(), 1):
        scope = f"compare::{system}"
        tracker.reset(scope)
        logger.info("compare: [%d/%d] running %s ...", idx, len(systems), system)
        with tracker.scope(scope):
            try:
                t0 = time.monotonic()
                passages = retriever.retrieve(search_query)
                t1 = time.monotonic()
                qa: QAResult = engine.qa_engine.answer(query, passages, history)
                t2 = time.monotonic()
                logger.info(
                    "compare: %s done - retrieval %.1fs, qa %.1fs",
                    system, t1 - t0, t2 - t1,
                )
                result.systems.append(
                    SystemResult(
                        system=system,
                        answer=qa.answer,
                        passages=qa.passages,
                        retrieval_latency_s=t1 - t0,
                        qa_latency_s=t2 - t1,
                        usage=tracker.snapshot(scope),
                    )
                )
            except Exception as e:  # noqa: BLE001 - one system failing must not kill compare
                logger.warning("compare: %s failed: %s", system, e)
                result.systems.append(
                    SystemResult(
                        system=system,
                        answer="",
                        passages=[],
                        retrieval_latency_s=0.0,
                        qa_latency_s=0.0,
                        usage=tracker.snapshot(scope),
                        error=str(e),
                    )
                )
    return result
