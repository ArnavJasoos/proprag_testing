"""GraphRAG index: build / persist / load.

Reuses the SHARED chunk embedding store (no re-embedding of passages). Persists
under ``data/corpora/<corpus_id>/graphrag/``:
  extraction.json   - entities + relationships checkpoint (after batch_extract)
  entities.json     - merged/summarized entities (with embed text + provenance)
  relationships.json, communities.json, reports.json
  graph.graphml     - entity graph
  gr_entity_*       - EmbeddingStore for "name: description" vectors
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List

import igraph as ig
import numpy as np

from proprag_poc.core.ids import compute_mdhash_id
from proprag_poc.core.store import EmbeddingStore
from proprag_poc.embedding.encoder import EmbeddingModel

from . import communities as comm
from .extract import EntityRec, RelRec, batch_extract, summarize_entities

logger = logging.getLogger(__name__)

_GR_NAMESPACE = "gr_entity"


def _embed_text(rec: EntityRec) -> str:
    return f"{rec.name}: {rec.merged_description()}".strip()[:512]


# ------------------------------------------------------------- (de)serialize
def _entity_to_dict(rec: EntityRec) -> Dict:
    return {
        "key": rec.key,
        "name": rec.name,
        "type": rec.type,
        "descriptions": rec.descriptions,
        "chunk_ids": sorted(rec.chunk_ids),
        "summary": rec.summary,
    }


def _entity_from_dict(d: Dict) -> EntityRec:
    return EntityRec(
        key=d["key"], name=d["name"], type=d["type"],
        descriptions=list(d.get("descriptions", [])),
        chunk_ids=set(d.get("chunk_ids", [])),
        summary=d.get("summary", ""),
    )


def _rel_to_dict(r: RelRec) -> Dict:
    return {
        "src_key": r.src_key, "dst_key": r.dst_key,
        "descriptions": r.descriptions, "weight": r.weight,
        "chunk_ids": sorted(r.chunk_ids),
    }


def _rel_from_dict(d: Dict) -> RelRec:
    return RelRec(
        src_key=d["src_key"], dst_key=d["dst_key"],
        descriptions=list(d.get("descriptions", [])),
        weight=float(d.get("weight", 0.0)),
        chunk_ids=set(d.get("chunk_ids", [])),
    )


@dataclass
class GraphRAGIndex:
    entities: Dict[str, EntityRec]
    relationships: List[RelRec]
    communities: Dict[str, int]          # entity key -> community id
    reports: Dict[int, Dict]             # community id -> report
    entity_order: List[str]              # entity keys aligned to entity_embeddings
    entity_embeddings: np.ndarray
    entity_store: EmbeddingStore
    graph: ig.Graph
    chunk_store: EmbeddingStore          # SHARED
    entity_chunks: Dict[str, List[str]]  # entity key -> chunk ids
    n_extract_failures: int = 0


def _gdir(corpus_dir: str) -> str:
    return os.path.join(corpus_dir, "graphrag")


def _paths(gdir: str) -> Dict[str, str]:
    return {name: os.path.join(gdir, f"{name}.json") for name in
            ("extraction", "entities", "relationships", "communities", "reports")} | {
        "graph": os.path.join(gdir, "graph.graphml"),
    }


def _exists(gdir: str) -> bool:
    p = _paths(gdir)
    return all(os.path.isfile(p[k]) for k in ("entities", "relationships", "communities", "reports", "graph"))


def _fetch_embeddings(store: EmbeddingStore, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    ids = [compute_mdhash_id(t, prefix=f"{_GR_NAMESPACE}-") for t in texts]
    return store.get_embeddings(ids)


def build_or_load(
    poc_cfg, bench_cfg, corpus_dir: str, chunk_store: EmbeddingStore,
    emb: EmbeddingModel, llm, tracker, force: bool = False,
) -> GraphRAGIndex:
    gdir = _gdir(corpus_dir)
    os.makedirs(gdir, exist_ok=True)
    p = _paths(gdir)
    entity_store = EmbeddingStore(emb, gdir, _GR_NAMESPACE)

    if _exists(gdir) and not force:
        logger.info("Loading existing GraphRAG index")
        return _load(gdir, entity_store, chunk_store)

    # --- extraction (checkpointed) ---
    n_failures = 0
    if os.path.isfile(p["extraction"]) and not force:
        logger.info("Loading GraphRAG extraction checkpoint")
        with open(p["extraction"], "r", encoding="utf-8") as f:
            ck = json.load(f)
        entities = {k: _entity_from_dict(v) for k, v in ck["entities"].items()}
        relationships = [_rel_from_dict(r) for r in ck["relationships"]]
        n_failures = ck.get("n_extract_failures", 0)
    else:
        chunk_texts = {
            cid: row["content"] for cid, row in chunk_store.get_text_for_all_rows().items()
        }
        entities, relationships, n_failures = batch_extract(llm, chunk_texts, bench_cfg, tracker)
        summarize_entities(llm, entities, bench_cfg)
        with open(p["extraction"], "w", encoding="utf-8") as f:
            json.dump({
                "entities": {k: _entity_to_dict(v) for k, v in entities.items()},
                "relationships": [_rel_to_dict(r) for r in relationships],
                "n_extract_failures": n_failures,
            }, f)

    # --- entity graph + communities ---
    graph = comm.build_graph(entities, relationships)
    community_map = comm.detect_communities(graph)

    members: Dict[int, List[str]] = {}
    for key, cid in community_map.items():
        members.setdefault(cid, []).append(key)

    rels_by_comm: Dict[int, List[RelRec]] = {}
    for r in relationships:
        c1, c2 = community_map.get(r.src_key), community_map.get(r.dst_key)
        if c1 is not None and c1 == c2:
            rels_by_comm.setdefault(c1, []).append(r)

    reports: Dict[int, Dict] = {}
    for cid, keys in members.items():
        if len(keys) < bench_cfg.gr_min_community_size:
            continue
        reports[cid] = comm.community_report(
            llm, keys, entities, rels_by_comm.get(cid, []), bench_cfg
        )
    logger.info("GraphRAG built %d community reports", len(reports))

    # --- entity embeddings ("name: description") ---
    entity_order = list(entities.keys())
    embed_texts = [_embed_text(entities[k]) for k in entity_order]
    entity_store.insert_strings(embed_texts)
    entity_embeddings = _fetch_embeddings(entity_store, embed_texts)

    entity_chunks = {k: sorted(entities[k].chunk_ids) for k in entity_order}

    _persist(gdir, entities, relationships, community_map, reports,
             entity_order, embed_texts, entity_chunks, graph, n_failures)

    return GraphRAGIndex(
        entities=entities, relationships=relationships, communities=community_map,
        reports=reports, entity_order=entity_order, entity_embeddings=entity_embeddings,
        entity_store=entity_store, graph=graph, chunk_store=chunk_store,
        entity_chunks=entity_chunks, n_extract_failures=n_failures,
    )


def _persist(gdir, entities, relationships, community_map, reports, entity_order,
             embed_texts, entity_chunks, graph, n_failures) -> None:
    p = _paths(gdir)
    with open(p["entities"], "w", encoding="utf-8") as f:
        json.dump({
            "entities": {k: _entity_to_dict(v) for k, v in entities.items()},
            "entity_order": entity_order,
            "embed_texts": embed_texts,
            "entity_chunks": entity_chunks,
            "n_extract_failures": n_failures,
        }, f)
    with open(p["relationships"], "w", encoding="utf-8") as f:
        json.dump([_rel_to_dict(r) for r in relationships], f)
    with open(p["communities"], "w", encoding="utf-8") as f:
        json.dump(community_map, f)
    with open(p["reports"], "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in reports.items()}, f)
    graph.write_graphml(p["graph"])


def _load(gdir, entity_store, chunk_store) -> GraphRAGIndex:
    p = _paths(gdir)
    with open(p["entities"], "r", encoding="utf-8") as f:
        edata = json.load(f)
    entities = {k: _entity_from_dict(v) for k, v in edata["entities"].items()}
    entity_order = edata["entity_order"]
    embed_texts = edata["embed_texts"]
    entity_chunks = edata["entity_chunks"]
    n_failures = edata.get("n_extract_failures", 0)
    with open(p["relationships"], "r", encoding="utf-8") as f:
        relationships = [_rel_from_dict(r) for r in json.load(f)]
    with open(p["communities"], "r", encoding="utf-8") as f:
        community_map = json.load(f)
    with open(p["reports"], "r", encoding="utf-8") as f:
        reports = {int(k): v for k, v in json.load(f).items()}
    graph = ig.Graph.Read_GraphML(p["graph"])
    entity_embeddings = _fetch_embeddings(entity_store, embed_texts)
    return GraphRAGIndex(
        entities=entities, relationships=relationships, communities=community_map,
        reports=reports, entity_order=entity_order, entity_embeddings=entity_embeddings,
        entity_store=entity_store, graph=graph, chunk_store=chunk_store,
        entity_chunks=entity_chunks, n_extract_failures=n_failures,
    )
