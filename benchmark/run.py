"""Benchmark CLI: PropRAG vs GraphRAG vs BaseRAG on 2WikiMultiHopQA.

Memory-safe, one-model-at-a-time driver. A single machine with limited GPU VRAM
cannot hold the embedder (NV-Embed-v2) and the local chat model (gpt-oss-20b
GGUF) at the same time, so the run is split into three phases where each model is
resident at most once and the two are NEVER co-resident:

    Phase A  (chat LLM resident)  extraction only:
                 PropRAG NER + propositions, GraphRAG entities / relationships /
                 community reports. All text-in / text-out. -> unload LLM.

    Phase B  (embedder resident)  embeddings + retrieval:
                 build the shared chunk store + PropRAG/GraphRAG vector stores +
                 graphs, then run the FULL retrieval for every (question, system)
                 -- retrieval uses NO LLM -- and persist the retrieved passages,
                 the QA context and Recall@k. -> unload embedder.

    Phase C  (chat LLM resident)  QA only:
                 answer each question from its stored context, score EM/F1,
                 write results.jsonl and build the report.

Everything is written under ``data_dir`` and every phase is resume-safe:
re-running skips work whose outputs already exist, so an interrupted run never
loses completed extraction / embedding / QA.

    python -m benchmark.run --pilot 12 --phase all
    python -m benchmark.run --phase a          # just extraction
    python -m benchmark.run --report-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from . import _bootstrap  # noqa: F401 - side effect: proprag_poc on sys.path

import igraph as ig

from proprag_poc.logging_setup import setup_logging
from proprag_poc.core.metrics import get_usage_tracker
from proprag_poc.core.ids import compute_mdhash_id
from proprag_poc.core.store import EmbeddingStore
from proprag_poc.core.extraction import Extractor
from proprag_poc.core.graph_builder import GraphBuilder
from proprag_poc.core.retriever import Retriever
from proprag_poc.core.baseline_retrievers import BaseRAGRetriever
from proprag_poc.embedding.encoder import EmbeddingModel

from .bench_config import BenchmarkConfig
from .dataset import (
    build_corpus, chunk_text, corpus_id, load_questions, pilot_subset,
    stratified_subset, subset_hash, write_manifest,
)
from .evaluation import em_score, f1_score, recall_at_k
from .qa import answer_question
from .llm_wrap import BenchLLMClient, check_backend
from .results import ResultsStore
from . import report as report_mod
from .graphrag import extract as gr_extract
from .graphrag import communities as gr_comm
from .graphrag import index as gr_index
from .graphrag.search import GraphRAGLocalRetriever

logger = logging.getLogger("benchmark")


# --------------------------------------------------------------------- logging
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


# ---------------------------------------------------------------- shared state
@dataclass
class BenchCorpus:
    """Duck-typed container satisfying the POC ``Retriever`` + ``BaseRAGRetriever``."""

    corpus_id: str
    chunk_store: EmbeddingStore
    entity_store: EmbeddingStore
    proposition_store: EmbeddingStore
    graph: ig.Graph
    proposition_to_entities_map: Dict[str, List[str]]
    chunk_propositions: Dict[str, List[Dict]]
    chunk_id_to_title: Dict[str, str]


@dataclass
class _Ctx:
    poc_cfg: object
    questions: list
    titles: Dict[str, str]
    docs: List[Tuple[str, str]]
    corpus_ident: str
    cid2text: Dict[str, str]
    cid2title: Dict[str, str]
    cdir: str
    run_dir: str
    systems: List[str]


def _run_id(cfg: BenchmarkConfig) -> str:
    rid = f"{cfg.n_questions}q_seed{cfg.seed}"
    return f"{rid}_pilot{cfg.pilot}" if cfg.pilot else rid


def _prepare(cfg: BenchmarkConfig) -> _Ctx:
    """Deterministic, model-free setup shared by every phase (cheap to recompute)."""
    poc_cfg = cfg.make_poc_config()
    all_qs = load_questions(cfg.dataset_path)
    subset = stratified_subset(all_qs, cfg.n_questions, cfg.seed)
    questions = pilot_subset(subset, cfg.pilot, cfg.seed) if cfg.pilot else subset

    titles = build_corpus(questions, cfg.corpus_path)
    docs = [(t, titles[t]) for t in sorted(titles)]
    tag = f"n{cfg.n_questions}" + (f"_pilot{cfg.pilot}" if cfg.pilot else "")
    corpus_ident = corpus_id(questions, tag)

    cid2text, cid2title = {}, {}
    for title, text in docs:
        ct = chunk_text(title, text)
        cid = compute_mdhash_id(ct, prefix="chunk-")
        cid2text[cid] = ct
        cid2title[cid] = title

    cdir = os.path.join(poc_cfg.data_dir, "corpora", corpus_ident)
    run_dir = os.path.join(poc_cfg.data_dir, "benchmark", _run_id(cfg))
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    systems = [s.strip() for s in cfg.systems if s.strip()]
    return _Ctx(poc_cfg, questions, titles, docs, corpus_ident, cid2text, cid2title,
                cdir, run_dir, systems)


# ------------------------------------------------------------------- io helpers
def _load_json(path: str, default):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _read_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _append_jsonl(path: str, row: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _done_keys(path: str) -> Set[Tuple[str, str]]:
    return {(r["qid"], r["system"]) for r in _read_jsonl(path)
            if "qid" in r and "system" in r}


def _delta(before: Dict, after: Dict) -> Dict:
    keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys}


def _add(a: Dict, b: Dict) -> Dict:
    keys = set(a) | set(b)
    return {k: a.get(k, 0) + b.get(k, 0) for k in keys}


def _chat_fields(d: Dict) -> Dict:
    return {
        "chat_calls": d.get("chat_calls", 0),
        "chat_prompt_tokens": d.get("chat_prompt_tokens", 0),
        "chat_completion_tokens": d.get("chat_completion_tokens", 0),
    }


def _write_meta(ctx: _Ctx, cfg: BenchmarkConfig) -> None:
    """Persist run metadata and refuse to resume across an incompatible subset."""
    sub_hash = subset_hash(ctx.questions)
    path = os.path.join(ctx.run_dir, "run_meta.json")
    if os.path.isfile(path):
        prev = _load_json(path, {})
        if prev.get("subset_hash") and prev["subset_hash"] != sub_hash:
            raise SystemExit(
                f"Refusing to resume: existing run at {ctx.run_dir} used subset "
                f"{prev['subset_hash']}, current is {sub_hash}. Use a new seed/size."
            )
    meta = {
        "subset_hash": sub_hash,
        "seed": cfg.seed,
        "n_questions": cfg.n_questions,
        "pilot": cfg.pilot,
        "llm_backend": ctx.poc_cfg.llm_backend,
        "llm_model": os.environ.get("PROPRAG_GGUF_LABEL", ctx.poc_cfg.llm_model),
        "embedding_model": ctx.poc_cfg.embedding_model,
        "qa_top_k": cfg.qa_top_k,
        "retrieval_top_k": cfg.retrieval_top_k,
        "gr_max_gleanings": cfg.gr_max_gleanings,
    }
    _save_json(path, meta)


# ====================================================================== PHASE A
def phase_a_extract(cfg: BenchmarkConfig, force: bool = False) -> str:
    """Chat-LLM-only extraction for all three systems. Unloads the LLM on exit."""
    ctx = _prepare(cfg)
    write_manifest(ctx.run_dir, ctx.questions, ctx.titles, cfg.seed, cfg)
    _write_meta(ctx, cfg)

    openie_path = os.path.join(ctx.cdir, "openie.json")
    gdir = gr_index._gdir(ctx.cdir)
    tracker = get_usage_tracker()
    chat_usage = _load_json(os.path.join(ctx.run_dir, "index_chat_usage.json"), {})

    need_prop = force or not os.path.isfile(openie_path)
    need_gr = force or not gr_index._exists(gdir)
    if not need_prop and not need_gr:
        logger.info("Phase A: extraction already complete; not loading the LLM.")
        return ctx.run_dir

    check_backend(ctx.poc_cfg)
    logger.info("Phase A: loading chat model for extraction ...")
    llm = BenchLLMClient(ctx.poc_cfg)
    try:
        if need_prop:
            logger.info("Phase A: PropRAG NER + proposition extraction (%d chunks)",
                        len(ctx.cid2text))
            before = tracker.snapshot("index").as_dict()
            t0 = time.monotonic()
            chunk_props = Extractor(llm).batch_extract(ctx.cid2text)
            _save_json(openie_path, chunk_props)
            d = _delta(before, tracker.snapshot("index").as_dict())
            chat_usage["PropRAG"] = {**_chat_fields(d), "wall_time_s": round(time.monotonic() - t0, 3)}

        if need_gr:
            logger.info("Phase A: GraphRAG extraction + community reports")
            b_idx = tracker.snapshot("index").as_dict()
            b_gr = tracker.snapshot("index::GraphRAG").as_dict()
            t0 = time.monotonic()
            n_failures = _build_graphrag_llm(llm, ctx.cid2text, cfg, gdir, tracker, force)
            d = _add(_delta(b_idx, tracker.snapshot("index").as_dict()),
                     _delta(b_gr, tracker.snapshot("index::GraphRAG").as_dict()))
            chat_usage["GraphRAG"] = {
                **_chat_fields(d),
                "wall_time_s": round(time.monotonic() - t0, 3),
                "parse_failures": n_failures,
            }

        _save_json(os.path.join(ctx.run_dir, "index_chat_usage.json"), chat_usage)
    finally:
        llm.unload()
    logger.info("Phase A complete.")
    return ctx.run_dir


def _build_graphrag_llm(llm, chunk_texts, cfg, gdir, tracker, force) -> int:
    """GraphRAG extraction + communities + reports (no embeddings). Persists JSON.

    Mirrors ``graphrag.index.build_or_load`` minus the entity-embedding step,
    which is deferred to Phase B. Returns the parse-failure count.
    """
    os.makedirs(gdir, exist_ok=True)
    p = gr_index._paths(gdir)

    if os.path.isfile(p["extraction"]) and not force:
        logger.info("Phase A: loading GraphRAG extraction checkpoint")
        with open(p["extraction"], "r", encoding="utf-8") as f:
            ck = json.load(f)
        entities = {k: gr_index._entity_from_dict(v) for k, v in ck["entities"].items()}
        relationships = [gr_index._rel_from_dict(r) for r in ck["relationships"]]
        n_failures = ck.get("n_extract_failures", 0)
    else:
        entities, relationships, n_failures = gr_extract.batch_extract(
            llm, chunk_texts, cfg, tracker
        )
        gr_extract.summarize_entities(llm, entities, cfg)
        with open(p["extraction"], "w", encoding="utf-8") as f:
            json.dump({
                "entities": {k: gr_index._entity_to_dict(v) for k, v in entities.items()},
                "relationships": [gr_index._rel_to_dict(r) for r in relationships],
                "n_extract_failures": n_failures,
            }, f)

    graph = gr_comm.build_graph(entities, relationships)
    community_map = gr_comm.detect_communities(graph)

    members: Dict[int, List[str]] = {}
    for key, cid in community_map.items():
        members.setdefault(cid, []).append(key)

    rels_by_comm: Dict[int, List] = {}
    for r in relationships:
        c1, c2 = community_map.get(r.src_key), community_map.get(r.dst_key)
        if c1 is not None and c1 == c2:
            rels_by_comm.setdefault(c1, []).append(r)

    reports: Dict[int, Dict] = {}
    for cid, keys in members.items():
        if len(keys) < cfg.gr_min_community_size:
            continue
        reports[cid] = gr_comm.community_report(
            llm, keys, entities, rels_by_comm.get(cid, []), cfg
        )
    logger.info("Phase A: built %d GraphRAG community reports", len(reports))

    entity_order = list(entities.keys())
    embed_texts = [gr_index._embed_text(entities[k]) for k in entity_order]
    entity_chunks = {k: sorted(entities[k].chunk_ids) for k in entity_order}
    gr_index._persist(gdir, entities, relationships, community_map, reports,
                      entity_order, embed_texts, entity_chunks, graph, n_failures)
    return n_failures


# ====================================================================== PHASE B
def phase_b_embed_retrieve(cfg: BenchmarkConfig, force: bool = False) -> str:
    """Embedder-only phase: build vector stores + graphs, run all retrieval.

    No chat LLM is loaded. Persists ``retrieval.jsonl`` (retrieved passages, QA
    context, Recall@k) and merges the final ``index_usage.json``. Unloads the
    embedder on exit.
    """
    ctx = _prepare(cfg)
    retr_path = os.path.join(ctx.run_dir, "retrieval.jsonl")
    done = _done_keys(retr_path)
    all_pairs = {(q.qid, s) for q in ctx.questions for s in ctx.systems}

    openie_path = os.path.join(ctx.cdir, "openie.json")
    if not os.path.isfile(openie_path):
        raise RuntimeError("Phase B needs Phase A output (openie.json). Run --phase a first.")

    if all_pairs.issubset(done) and not force:
        logger.info("Phase B: retrieval already complete; not loading the embedder.")
        _merge_index_usage(ctx, {})
        return ctx.run_dir

    logger.info("Phase B: loading embedder ...")
    emb = EmbeddingModel(ctx.poc_cfg)
    tracker = get_usage_tracker()
    embed_usage: Dict[str, Dict] = {}
    try:
        # --- BaseRAG index: shared chunk store ---
        chunk_store = EmbeddingStore(emb, ctx.cdir, "chunk")
        before = tracker.snapshot("index").as_dict()
        t0 = time.monotonic()
        chunk_store.insert_strings([chunk_text(t, x) for t, x in ctx.docs])
        _save_json(os.path.join(ctx.cdir, "title_map.json"), ctx.cid2title)
        embed_usage["BaseRAG"] = {
            "embed_texts": _delta(before, tracker.snapshot("index").as_dict()).get("embed_texts", 0),
            "wall_time_s": round(time.monotonic() - t0, 3),
        }

        # --- PropRAG index: entities + propositions + knowledge graph ---
        corpus = _build_proprag_corpus(ctx, emb, chunk_store, openie_path, tracker, embed_usage, force)

        # --- GraphRAG index: entity embeddings + load structure from Phase A ---
        gr = _load_graphrag_index(ctx, emb, chunk_store, tracker, embed_usage)

        retrievers = {
            "BaseRAG": BaseRAGRetriever(corpus, emb, ctx.poc_cfg),
            "PropRAG": Retriever(corpus, emb, ctx.poc_cfg),
            "GraphRAG": GraphRAGLocalRetriever(gr, emb, ctx.poc_cfg, cfg),
        }

        _retrieve_all(ctx, cfg, retrievers, retr_path, done)
        _save_json(os.path.join(ctx.run_dir, "index_embed_usage.json"), embed_usage)
        _merge_index_usage(ctx, embed_usage)
    finally:
        emb.unload()
    logger.info("Phase B complete.")
    return ctx.run_dir


def _build_proprag_corpus(ctx, emb, chunk_store, openie_path, tracker, embed_usage, force) -> BenchCorpus:
    graph_path = os.path.join(ctx.cdir, "graph.graphml")
    maps_path = os.path.join(ctx.cdir, "maps.json")
    entity_store = EmbeddingStore(emb, ctx.cdir, "entity")
    proposition_store = EmbeddingStore(emb, ctx.cdir, "proposition")

    with open(openie_path, "r", encoding="utf-8") as f:
        chunk_props = json.load(f)

    if os.path.isfile(graph_path) and os.path.isfile(maps_path) and not force:
        graph = ig.Graph.Read_GraphML(graph_path)
        maps = _load_json(maps_path, {})
        embed_usage.setdefault("PropRAG", {"embed_texts": 0, "wall_time_s": 0.0})
        return BenchCorpus(ctx.corpus_ident, chunk_store, entity_store, proposition_store,
                           graph, maps["proposition_to_entities_map"],
                           maps["chunk_propositions"], ctx.cid2title)

    before = tracker.snapshot("index").as_dict()
    t0 = time.monotonic()
    entities, prop_texts, prop_to_entities = set(), [], {}
    for chunk_id, props in chunk_props.items():
        for prop in props:
            entities.update(prop["entities"])
            pkey = compute_mdhash_id(prop["text"], prefix="proposition-")
            prop_texts.append(prop["text"])
            prop_to_entities[pkey] = prop["entities"]

    entity_store.insert_strings(sorted(entities))
    proposition_store.insert_strings(prop_texts)
    graph = GraphBuilder(ctx.poc_cfg).build(
        chunk_store.get_all_ids(), chunk_props, entity_store, chunk_store
    )
    graph.write_graphml(graph_path)
    _save_json(maps_path, {
        "proposition_to_entities_map": prop_to_entities,
        "chunk_propositions": chunk_props,
    })
    embed_usage["PropRAG"] = {
        "embed_texts": _delta(before, tracker.snapshot("index").as_dict()).get("embed_texts", 0),
        "wall_time_s": round(time.monotonic() - t0, 3),
    }
    return BenchCorpus(ctx.corpus_ident, chunk_store, entity_store, proposition_store,
                       graph, prop_to_entities, chunk_props, ctx.cid2title)


def _load_graphrag_index(ctx, emb, chunk_store, tracker, embed_usage):
    gdir = gr_index._gdir(ctx.cdir)
    entity_store = EmbeddingStore(emb, gdir, gr_index._GR_NAMESPACE)
    edata = _load_json(gr_index._paths(gdir)["entities"], {})
    before = tracker.snapshot("index").as_dict()
    t0 = time.monotonic()
    entity_store.insert_strings(edata.get("embed_texts", []))
    embed_usage["GraphRAG"] = {
        "embed_texts": _delta(before, tracker.snapshot("index").as_dict()).get("embed_texts", 0),
        "wall_time_s": round(time.monotonic() - t0, 3),
    }
    return gr_index._load(gdir, entity_store, chunk_store)


def _retrieve_all(ctx, cfg, retrievers, retr_path, done) -> None:
    for qi, q in enumerate(ctx.questions, 1):
        logger.info("Retrieve Q %d/%d [%s] %s", qi, len(ctx.questions), q.qtype, q.question[:70])
        for system in ctx.systems:
            if (q.qid, system) in done:
                continue
            retriever = retrievers[system]
            try:
                t0 = time.monotonic()
                passages = retriever.retrieve(q.question, top_k=ctx.poc_cfg.retrieval_top_k)
                lat = time.monotonic() - t0
                retrieved_titles = [ctx.cid2title.get(p.chunk_id, "") for p in passages]
                recall = recall_at_k(q.gold_titles, retrieved_titles, cfg.recall_ks)
                if system == "GraphRAG":
                    context = retriever.build_qa_context(q.question, passages)
                else:
                    context = [p.text for p in passages[: cfg.qa_top_k]]
                _append_jsonl(retr_path, {
                    "qid": q.qid, "qtype": q.qtype, "system": system,
                    "question": q.question, "gold_answer": q.answer,
                    "gold_titles": q.gold_titles, "gold_answers": q.gold_answers,
                    "retrieved": [
                        {"chunk_id": p.chunk_id, "title": ctx.cid2title.get(p.chunk_id, ""),
                         "score": p.score} for p in passages
                    ],
                    "recall": recall,
                    "context": context,
                    "retrieval_latency_s": round(lat, 3),
                })
            except Exception as e:  # noqa: BLE001 - one failure must not kill the phase
                logger.exception("Retrieval failed for %s / %s", q.qid, system)
                _append_jsonl(retr_path, {
                    "qid": q.qid, "qtype": q.qtype, "system": system, "error": str(e),
                })


def _merge_index_usage(ctx, embed_usage: Dict[str, Dict]) -> None:
    """Combine Phase A chat usage + Phase B embed usage into report's index_usage."""
    chat = _load_json(os.path.join(ctx.run_dir, "index_chat_usage.json"), {})
    embed = embed_usage or _load_json(os.path.join(ctx.run_dir, "index_embed_usage.json"), {})
    out: Dict[str, Dict] = {}
    for system in ("BaseRAG", "PropRAG", "GraphRAG"):
        c = chat.get(system, {})
        e = embed.get(system, {})
        out[system] = {
            "chat_calls": c.get("chat_calls", 0),
            "chat_prompt_tokens": c.get("chat_prompt_tokens", 0),
            "chat_completion_tokens": c.get("chat_completion_tokens", 0),
            "embed_texts": e.get("embed_texts", 0),
            "wall_time_s": round(c.get("wall_time_s", 0.0) + e.get("wall_time_s", 0.0), 3),
            "parse_failures": c.get("parse_failures", 0),
        }
    _save_json(os.path.join(ctx.run_dir, "index_usage.json"), out)


# ====================================================================== PHASE C
def phase_c_qa(cfg: BenchmarkConfig, make_charts: bool = True) -> str:
    """Chat-LLM-only QA over the stored retrieval contexts, then build the report."""
    ctx = _prepare(cfg)
    retr_path = os.path.join(ctx.run_dir, "retrieval.jsonl")
    rows = [r for r in _read_jsonl(retr_path) if not r.get("error")]
    if not rows:
        raise RuntimeError("Phase C needs Phase B output (retrieval.jsonl). Run --phase b first.")

    store = ResultsStore(ctx.run_dir)
    done = store.done_keys()
    todo = [r for r in rows if (r["qid"], r["system"]) not in done]

    if todo:
        check_backend(ctx.poc_cfg)
        logger.info("Phase C: loading chat model for QA (%d answers) ...", len(todo))
        llm = BenchLLMClient(ctx.poc_cfg)
        tracker = get_usage_tracker()
        try:
            for i, r in enumerate(todo, 1):
                scope = f"q::{r['system']}"
                before = tracker.snapshot(scope).as_dict()
                t1 = time.monotonic()
                with tracker.scope(scope):
                    answer, raw = answer_question(llm, r["question"], r["context"], cfg)
                qa_latency = time.monotonic() - t1
                usage = _delta(before, tracker.snapshot(scope).as_dict())
                store.append({
                    "qid": r["qid"], "qtype": r["qtype"], "system": r["system"],
                    "question": r["question"], "gold_answer": r.get("gold_answer"),
                    "gold_titles": r["gold_titles"], "answer": answer, "raw_answer": raw,
                    "retrieved": r["retrieved"], "recall": r["recall"],
                    "em": em_score(r["gold_answers"], answer),
                    "f1": f1_score(r["gold_answers"], answer),
                    "retrieval_latency_s": r["retrieval_latency_s"],
                    "qa_latency_s": round(qa_latency, 3),
                    "usage": usage, "ts": time.time(),
                })
                logger.info("QA %d/%d done [%s/%s]", i, len(todo), r["system"], r["qid"])
        finally:
            llm.unload()

    report_mod.build(ctx.run_dir, make_charts=make_charts)
    logger.info("Phase C complete. Report: %s", os.path.join(ctx.run_dir, "report.md"))
    return ctx.run_dir


# ---------------------------------------------------------------------- driver
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PropRAG vs GraphRAG vs BaseRAG benchmark (memory-safe)")
    p.add_argument("--questions", type=int, default=50)
    p.add_argument("--pilot", type=int, default=None, help="stratified k-of-subset pilot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--systems", type=str, default="BaseRAG,GraphRAG,PropRAG")
    p.add_argument("--phase", choices=["a", "b", "c", "all"], default="all",
                   help="a=extract (LLM), b=embed+retrieve (embedder), c=QA (LLM)")
    p.add_argument("--force-reindex", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--no-charts", action="store_true")
    return p.parse_args(argv)


def _make_cfg(args) -> BenchmarkConfig:
    project_dir = os.environ.get(
        "PROPRAG_PROJECT_DIR",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    cfg = BenchmarkConfig(
        project_dir=project_dir, n_questions=args.questions, seed=args.seed, pilot=args.pilot,
    )
    cfg.systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    return cfg


def main(argv=None) -> None:
    args = _parse_args(argv)
    _setup_logging()
    cfg = _make_cfg(args)

    if args.report_only:
        ctx = _prepare(cfg)
        report_mod.build(ctx.run_dir, make_charts=not args.no_charts)
        print("Report:", os.path.join(ctx.run_dir, "report.md"))
        return

    if args.phase in ("a", "all"):
        phase_a_extract(cfg, force=args.force_reindex)
    if args.phase in ("b", "all"):
        phase_b_embed_retrieve(cfg, force=args.force_reindex)
    if args.phase in ("c", "all"):
        run_dir = phase_c_qa(cfg, make_charts=not args.no_charts)
        print(f"\nReport: {os.path.join(run_dir, 'report.md')}")


if __name__ == "__main__":
    main()
