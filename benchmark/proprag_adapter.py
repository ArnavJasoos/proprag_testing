"""Index building over the reused ``proprag_poc`` library.

Three RAG systems share one chunk embedding store and one embedding model:

- BaseRAG cost  = building the shared chunk store (embeddings only).
- PropRAG cost  = NER + proposition extraction + knowledge-graph construction.
- GraphRAG cost = its own extraction/summaries/reports (see ``graphrag/``).

We deliberately do NOT import ``proprag_poc.core.index`` (it top-imports PyMuPDF /
tiktoken via the PDF ingestion path). Instead we replicate the ~50-line
``Indexer.build_from_texts`` body directly from ``EmbeddingStore`` + ``Extractor``
+ ``GraphBuilder``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import igraph as ig

from . import _bootstrap  # noqa: F401
from .dataset import chunk_text
from proprag_poc.config import POCConfig
from proprag_poc.core.baseline_retrievers import BaseRAGRetriever
from proprag_poc.core.extraction import Extractor
from proprag_poc.core.graph_builder import GraphBuilder
from proprag_poc.core.ids import compute_mdhash_id
from proprag_poc.core.retriever import Retriever
from proprag_poc.core.store import EmbeddingStore
from proprag_poc.embedding.encoder import EmbeddingModel

logger = logging.getLogger(__name__)


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


# ------------------------------------------------------------------- paths
def corpus_dir(poc_cfg: POCConfig, corpus_id: str) -> str:
    return os.path.join(poc_cfg.data_dir, "corpora", corpus_id)


def _title_map_path(cdir: str) -> str:
    return os.path.join(cdir, "title_map.json")


def _graph_path(cdir: str) -> str:
    return os.path.join(cdir, "graph.graphml")


def _maps_path(cdir: str) -> str:
    return os.path.join(cdir, "maps.json")


# --------------------------------------------------------- shared chunk store
def build_base_index(
    poc_cfg: POCConfig, corpus_id: str, docs: List[Tuple[str, str]], emb: EmbeddingModel
) -> Tuple[EmbeddingStore, Dict[str, str]]:
    """Build (or load) the shared chunk store. This is the BaseRAG index cost.

    Returns the chunk store and a persisted ``chunk_id -> title`` map used for
    Recall@k scoring.
    """
    cdir = corpus_dir(poc_cfg, corpus_id)
    os.makedirs(cdir, exist_ok=True)

    chunk_store = EmbeddingStore(emb, cdir, "chunk")
    texts = [chunk_text(title, text) for title, text in docs]
    logger.info("BaseRAG index: embedding %d chunks", len(texts))
    chunk_store.insert_strings(texts)

    chunk_id_to_title = {
        compute_mdhash_id(chunk_text(title, text), prefix="chunk-"): title
        for title, text in docs
    }
    with open(_title_map_path(cdir), "w", encoding="utf-8") as f:
        json.dump(chunk_id_to_title, f)

    missing = [cid for cid in chunk_store.get_all_ids() if cid not in chunk_id_to_title]
    if missing:
        raise RuntimeError(f"{len(missing)} chunk ids lack a title mapping")
    return chunk_store, chunk_id_to_title


def load_title_map(cdir: str) -> Dict[str, str]:
    with open(_title_map_path(cdir), "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------- PropRAG index
def proprag_exists(poc_cfg: POCConfig, corpus_id: str) -> bool:
    cdir = corpus_dir(poc_cfg, corpus_id)
    return os.path.isfile(_graph_path(cdir)) and os.path.isfile(_maps_path(cdir))


def build_or_load_proprag(
    poc_cfg: POCConfig,
    corpus_id: str,
    chunk_store: EmbeddingStore,
    chunk_id_to_title: Dict[str, str],
    emb: EmbeddingModel,
    llm,
    force: bool = False,
) -> BenchCorpus:
    """Extraction + graph over the shared chunk store (PropRAG index cost)."""
    cdir = corpus_dir(poc_cfg, corpus_id)
    entity_store = EmbeddingStore(emb, cdir, "entity")
    proposition_store = EmbeddingStore(emb, cdir, "proposition")

    if proprag_exists(poc_cfg, corpus_id) and not force:
        logger.info("Loading existing PropRAG index for %s", corpus_id)
        graph = ig.Graph.Read_GraphML(_graph_path(cdir))
        with open(_maps_path(cdir), "r", encoding="utf-8") as f:
            maps = json.load(f)
        return BenchCorpus(
            corpus_id, chunk_store, entity_store, proposition_store, graph,
            maps["proposition_to_entities_map"], maps["chunk_propositions"],
            chunk_id_to_title,
        )

    # --- replicate Indexer.build_from_texts (chunk store already built) ---
    chunk_id_to_text = {
        cid: row["content"] for cid, row in chunk_store.get_text_for_all_rows().items()
    }
    logger.info("PropRAG index: extracting NER + propositions for %d chunks",
                len(chunk_id_to_text))
    chunk_propositions = Extractor(llm).batch_extract(chunk_id_to_text)

    entities, prop_texts = set(), []
    prop_to_entities: Dict[str, List[str]] = {}
    for chunk_id, props in chunk_propositions.items():
        for prop in props:
            entities.update(prop["entities"])
            pkey = compute_mdhash_id(prop["text"], prefix="proposition-")
            prop_texts.append(prop["text"])
            prop_to_entities[pkey] = prop["entities"]

    logger.info("PropRAG index: embedding %d entities, %d propositions",
                len(entities), len(prop_texts))
    entity_store.insert_strings(sorted(entities))
    proposition_store.insert_strings(prop_texts)

    logger.info("PropRAG index: building knowledge graph")
    graph = GraphBuilder(poc_cfg).build(
        chunk_store.get_all_ids(), chunk_propositions, entity_store, chunk_store
    )
    graph.write_graphml(_graph_path(cdir))
    with open(_maps_path(cdir), "w", encoding="utf-8") as f:
        json.dump(
            {
                "proposition_to_entities_map": prop_to_entities,
                "chunk_propositions": chunk_propositions,
            },
            f,
        )
    logger.info("PropRAG index complete: %d nodes, %d edges",
                graph.vcount(), graph.ecount())
    return BenchCorpus(
        corpus_id, chunk_store, entity_store, proposition_store, graph,
        prop_to_entities, chunk_propositions, chunk_id_to_title,
    )


# ----------------------------------------------------------------- retrievers
def make_proprag_retriever(corpus: BenchCorpus, emb: EmbeddingModel, poc_cfg: POCConfig):
    return Retriever(corpus, emb, poc_cfg)


def make_baserag_retriever(corpus: BenchCorpus, emb: EmbeddingModel, poc_cfg: POCConfig):
    return BaseRAGRetriever(corpus, emb, poc_cfg)
