"""Benchmark CLI: index the three systems once over a shared corpus, then run the
question loop (resume-safe), and build the report.

    python -m benchmark.run [--pilot 10] [--questions 50] [--seed 42]
                            [--systems BaseRAG,GraphRAG,PropRAG]
                            [--force-reindex] [--report-only] [--no-charts]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

from . import _bootstrap  # noqa: F401
from proprag_poc.logging_setup import setup_logging
from proprag_poc.core.metrics import get_usage_tracker
from proprag_poc.embedding.encoder import EmbeddingModel

from . import proprag_adapter as adapter
from . import report as report_mod
from .bench_config import BenchmarkConfig
from .dataset import (
    build_corpus, corpus_id, load_questions, pilot_subset, stratified_subset,
    subset_hash, write_manifest,
)
from .evaluation import em_score, f1_score, recall_at_k
from .graphrag import index as gr_index_mod
from .graphrag.search import GraphRAGLocalRetriever
from .llm_wrap import BenchLLMClient, check_backend
from .qa import answer_question
from .results import ResultsStore
from .usage import IndexPhase, delta

logger = logging.getLogger("benchmark")


def _setup_logging() -> None:
    setup_logging()  # configures the proprag_poc tree
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-5s | %(name)s | %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger("benchmark")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PropRAG vs GraphRAG vs BaseRAG benchmark")
    p.add_argument("--questions", type=int, default=50)
    p.add_argument("--pilot", type=int, default=None, help="run a stratified k-of-subset pilot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--systems", type=str, default="BaseRAG,GraphRAG,PropRAG")
    p.add_argument("--force-reindex", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--no-charts", action="store_true")
    return p.parse_args(argv)


def _run_id(cfg: BenchmarkConfig) -> str:
    rid = f"{cfg.n_questions}q_seed{cfg.seed}"
    return f"{rid}_pilot{cfg.pilot}" if cfg.pilot else rid


def _guard_meta(run_dir: str, cfg: BenchmarkConfig, poc_cfg, sub_hash: str) -> None:
    path = os.path.join(run_dir, "run_meta.json")
    meta = {
        "subset_hash": sub_hash,
        "seed": cfg.seed,
        "n_questions": cfg.n_questions,
        "pilot": cfg.pilot,
        "llm_backend": poc_cfg.llm_backend,
        "llm_model": poc_cfg.llm_model,
        "embedding_model": poc_cfg.embedding_model,
        "qa_top_k": cfg.qa_top_k,
        "retrieval_top_k": cfg.retrieval_top_k,
        "gr_max_gleanings": cfg.gr_max_gleanings,
    }
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("subset_hash") != sub_hash:
            raise SystemExit(
                f"Refusing to resume: existing run at {run_dir} used subset "
                f"{prev.get('subset_hash')}, current is {sub_hash}. Use a new seed/size."
            )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _index_all(cfg, poc_cfg, corpus_ident, docs, emb, llm, tracker, run_dir, force) -> Dict:
    cdir = adapter.corpus_dir(poc_cfg, corpus_ident)
    usage_path = os.path.join(run_dir, "index_usage.json")
    index_usage: Dict[str, Dict] = {}

    with IndexPhase(tracker, "BaseRAG") as phase:
        chunk_store, chunk_id_to_title = adapter.build_base_index(poc_cfg, corpus_ident, docs, emb)
    index_usage["BaseRAG"] = phase.usage

    with IndexPhase(tracker, "PropRAG") as phase:
        corpus = adapter.build_or_load_proprag(
            poc_cfg, corpus_ident, chunk_store, chunk_id_to_title, emb, llm, force=force
        )
    index_usage["PropRAG"] = phase.usage

    with IndexPhase(tracker, "GraphRAG", extra_scopes=["index::GraphRAG"]) as phase:
        gr_index = gr_index_mod.build_or_load(
            poc_cfg, cfg, cdir, chunk_store, emb, llm, tracker, force=force
        )
    gr_usage = dict(phase.usage)
    gr_usage["parse_failures"] = gr_index.n_extract_failures
    index_usage["GraphRAG"] = gr_usage

    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(index_usage, f, indent=2)

    return {
        "corpus": corpus,
        "chunk_id_to_title": chunk_id_to_title,
        "gr_index": gr_index,
        "index_usage": index_usage,
    }


def _build_retrievers(built, emb, poc_cfg, cfg, systems):
    retrievers = {}
    if "BaseRAG" in systems:
        retrievers["BaseRAG"] = adapter.make_baserag_retriever(built["corpus"], emb, poc_cfg)
    if "PropRAG" in systems:
        retrievers["PropRAG"] = adapter.make_proprag_retriever(built["corpus"], emb, poc_cfg)
    if "GraphRAG" in systems:
        retrievers["GraphRAG"] = GraphRAGLocalRetriever(built["gr_index"], emb, poc_cfg, cfg)
    return retrievers


def _question_loop(questions, retrievers, systems, cfg, poc_cfg, llm, tracker, store,
                   chunk_id_to_title):
    done = store.done_keys()
    per_system_seconds: Dict[str, List[float]] = {s: [] for s in systems}

    for qi, q in enumerate(questions, 1):
        logger.info("Q %d/%d [%s] %s", qi, len(questions), q.qtype, q.question[:80])
        for system in systems:
            if (q.qid, system) in done:
                continue
            retriever = retrievers[system]
            scope = f"q::{system}"
            before = tracker.snapshot(scope).as_dict()
            try:
                t0 = time.monotonic()
                with tracker.scope(scope):
                    passages = retriever.retrieve(q.question, top_k=poc_cfg.retrieval_top_k)
                    retrieval_latency = time.monotonic() - t0

                    retrieved_titles = [
                        chunk_id_to_title.get(p.chunk_id, "") for p in passages
                    ]
                    recall = recall_at_k(q.gold_titles, retrieved_titles, cfg.recall_ks)

                    if system == "GraphRAG":
                        context = retriever.build_qa_context(q.question, passages)
                    else:
                        context = [p.text for p in passages[: cfg.qa_top_k]]

                    t1 = time.monotonic()
                    answer, raw = answer_question(llm, q.question, context, cfg)
                    qa_latency = time.monotonic() - t1

                usage = delta(before, tracker.snapshot(scope).as_dict())
                row = {
                    "qid": q.qid, "qtype": q.qtype, "system": system,
                    "question": q.question, "gold_answer": q.answer,
                    "gold_titles": q.gold_titles, "answer": answer, "raw_answer": raw,
                    "retrieved": [
                        {"chunk_id": p.chunk_id, "title": chunk_id_to_title.get(p.chunk_id, ""),
                         "score": p.score} for p in passages
                    ],
                    "recall": recall,
                    "em": em_score(q.gold_answers, answer),
                    "f1": f1_score(q.gold_answers, answer),
                    "retrieval_latency_s": round(retrieval_latency, 3),
                    "qa_latency_s": round(qa_latency, 3),
                    "usage": usage,
                    "ts": time.time(),
                }
                store.append(row)
                per_system_seconds[system].append(retrieval_latency + qa_latency)
            except Exception as e:  # noqa: BLE001 - one failure must not kill the run
                logger.exception("Q %s system %s failed", q.qid, system)
                store.append({
                    "qid": q.qid, "qtype": q.qtype, "system": system, "error": str(e),
                    "ts": time.time(),
                })
    return per_system_seconds


def main(argv=None) -> None:
    args = _parse_args(argv)
    _setup_logging()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = BenchmarkConfig(
        project_dir=project_dir, n_questions=args.questions, seed=args.seed, pilot=args.pilot,
    )
    poc_cfg = cfg.make_poc_config()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    run_dir = os.path.join(cfg.data_dir, "benchmark", _run_id(cfg))
    os.makedirs(run_dir, exist_ok=True)

    if args.report_only:
        report_mod.build(run_dir, make_charts=not args.no_charts)
        return

    check_backend(poc_cfg)

    all_qs = load_questions(cfg.dataset_path)
    subset = stratified_subset(all_qs, cfg.n_questions, cfg.seed)
    questions = pilot_subset(subset, cfg.pilot, cfg.seed) if cfg.pilot else subset

    titles = build_corpus(questions, cfg.corpus_path)
    docs = [(t, titles[t]) for t in sorted(titles)]
    sub_hash = subset_hash(questions)
    tag = f"n{cfg.n_questions}" + (f"_pilot{cfg.pilot}" if cfg.pilot else "")
    corpus_ident = corpus_id(questions, tag)

    write_manifest(run_dir, questions, titles, cfg.seed, cfg)
    _guard_meta(run_dir, cfg, poc_cfg, sub_hash)

    emb = EmbeddingModel(poc_cfg)
    llm = BenchLLMClient(poc_cfg)
    tracker = get_usage_tracker()

    logger.info("Indexing %d systems over %d docs (corpus %s)", len(systems), len(docs), corpus_ident)
    built = _index_all(cfg, poc_cfg, corpus_ident, docs, emb, llm, tracker, run_dir, args.force_reindex)

    retrievers = _build_retrievers(built, emb, poc_cfg, cfg, systems)

    store = ResultsStore(run_dir)
    t_start = time.monotonic()
    per_system_seconds = _question_loop(
        questions, retrievers, systems, cfg, poc_cfg, llm, tracker, store,
        built["chunk_id_to_title"],
    )
    wall = time.monotonic() - t_start

    report_mod.build(run_dir, make_charts=not args.no_charts)
    _print_summary(run_dir, cfg, per_system_seconds, wall)


def _print_summary(run_dir, cfg, per_system_seconds, wall) -> None:
    print(f"\nReport: {os.path.join(run_dir, 'report.md')}")
    if cfg.pilot:
        newly = [s for secs in per_system_seconds.values() for s in secs]
        if newly:
            mean_q = sum(newly) / len(newly)
            n_systems = len(per_system_seconds)
            projected = mean_q * n_systems * cfg.n_questions
            print(f"Pilot: mean {mean_q:.1f}s per (system,question); "
                  f"projected full-run query wall ~ {projected / 60:.0f} min "
                  f"(indexing excluded, cached calls free).")
    print(f"Query-loop wall this run: {wall / 60:.1f} min")


if __name__ == "__main__":
    main()
