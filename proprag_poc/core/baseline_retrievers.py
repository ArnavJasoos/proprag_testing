"""Baseline retrievers for head-to-head comparison against PropRAG.

Both reuse the *same* embedding stores and knowledge graph that PropRAG builds, so
the only differences measured are retrieval strategy and its cost — embeddings,
chunks and the graph are shared and built once.

  * ``BaseRAGRetriever`` — classic dense passage retrieval: embed the query, take
    the top-k passages by cosine similarity. No graph, no LLM at query time.

  * ``GraphRAGRetriever`` — a lightweight, summary-free GraphRAG: run NER on the
    query (one LLM call), match the named entities to entity nodes, expand one hop
    to neighbouring entities, gather the passages attached to that entity set, and
    rank them by (graph overlap with query entities) + (dense similarity). Falls
    back to dense retrieval when no query entity matches the graph.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..config import POCConfig
from ..embedding.encoder import EmbeddingModel
from .extraction import Extractor
from .ids import compute_mdhash_id
from .retriever import RetrievedPassage, _min_max

logger = logging.getLogger(__name__)


class BaseRAGRetriever:
    system_name = "BaseRAG"

    def __init__(self, corpus, embedding_model: EmbeddingModel, config: POCConfig):
        self.config = config
        self.embedding_model = embedding_model
        self.chunk_store = corpus.chunk_store
        self.passage_keys = self.chunk_store.get_all_ids()
        self.passage_embeddings = self.chunk_store.get_embeddings(self.passage_keys)

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedPassage]:
        top_k = top_k or self.config.retrieval_top_k
        q = self.embedding_model.batch_encode(
            query, instruction=self.config.embedding_query_instruction, norm=True,
            input_type="query",
        )
        scores = np.atleast_1d((self.passage_embeddings @ q.squeeze().T).squeeze())
        order = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedPassage(
                chunk_id=self.passage_keys[i],
                text=self.chunk_store.get_row(self.passage_keys[i])["content"],
                score=float(scores[i]),
            )
            for i in order
        ]


class GraphRAGRetriever:
    system_name = "GraphRAG"

    # Cosine floor for matching a query NER phrase to an entity node by embedding.
    _ENTITY_MATCH_THRESHOLD = 0.6

    def __init__(self, corpus, embedding_model: EmbeddingModel, config: POCConfig,
                 extractor: Extractor):
        self.config = config
        self.embedding_model = embedding_model
        self.extractor = extractor
        self.graph = corpus.graph
        self.chunk_store = corpus.chunk_store
        self.entity_store = corpus.entity_store

        self.passage_keys = self.chunk_store.get_all_ids()
        self.passage_key_to_row = {k: i for i, k in enumerate(self.passage_keys)}
        self.passage_embeddings = self.chunk_store.get_embeddings(self.passage_keys)

        self.entity_keys = self.entity_store.get_all_ids()
        self.entity_embeddings = (
            self.entity_store.get_embeddings(self.entity_keys)
            if self.entity_keys else np.zeros((0, 1), dtype=np.float32)
        )
        self.name_to_idx = {v["name"]: i for i, v in enumerate(self.graph.vs)}

    # -------------------------------------------------------------- helpers
    def _dense_scores(self, query: str) -> np.ndarray:
        q = self.embedding_model.batch_encode(
            query, instruction=self.config.embedding_query_instruction, norm=True,
            input_type="query",
        )
        return _min_max(np.atleast_1d((self.passage_embeddings @ q.squeeze().T).squeeze()))

    def _match_query_entities(self, phrases: List[str]) -> List[str]:
        """Map query NER phrases to entity node keys (exact id, then embedding NN)."""
        matched, unresolved = set(), []
        for p in phrases:
            key = compute_mdhash_id(p.strip(), prefix="entity-")
            if key in self.name_to_idx:
                matched.add(key)
            else:
                unresolved.append(p)
        if unresolved and len(self.entity_keys) > 0:
            pe = self.embedding_model.batch_encode(
                unresolved, norm=True, input_type="query"
            )
            sims = pe @ self.entity_embeddings.T  # (n_phrases, n_entities)
            for row in sims:
                j = int(np.argmax(row))
                if float(row[j]) >= self._ENTITY_MATCH_THRESHOLD:
                    matched.add(self.entity_keys[j])
        return list(matched)

    # -------------------------------------------------------------- retrieve
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedPassage]:
        top_k = top_k or self.config.retrieval_top_k
        dense = self._dense_scores(query)

        try:
            phrases = self.extractor.ner(query)
        except Exception as e:  # noqa: BLE001 - degrade to dense on NER failure
            logger.warning("GraphRAG query NER failed, using dense: %s", e)
            phrases = []
        seeds = self._match_query_entities(phrases) if phrases else []

        if not seeds:
            return self._dense_top_k(dense, top_k)

        # Expand: seed entities + one hop of neighbouring entities.
        entity_set = set(seeds)
        for ekey in seeds:
            vi = self.name_to_idx.get(ekey)
            if vi is None:
                continue
            for nb in self.graph.neighbors(vi, mode="all"):
                name = self.graph.vs[nb]["name"]
                if name.startswith("entity-"):
                    entity_set.add(name)

        # Passages attached to that entity set, with an overlap count vs the seeds.
        overlap = np.zeros(len(self.passage_keys))
        for ekey in entity_set:
            vi = self.name_to_idx.get(ekey)
            if vi is None:
                continue
            weight = 1.0 if ekey in seeds else 0.4  # seeds count more than hop-neighbours
            for nb in self.graph.neighbors(vi, mode="all"):
                name = self.graph.vs[nb]["name"]
                row = self.passage_key_to_row.get(name)
                if row is not None:
                    overlap[row] += weight

        if overlap.sum() == 0:
            return self._dense_top_k(dense, top_k)

        # Final score: graph overlap (primary) blended with dense similarity.
        final = _min_max(overlap) + 0.5 * dense
        order = np.argsort(final)[::-1][:top_k]
        return [
            RetrievedPassage(
                chunk_id=self.passage_keys[i],
                text=self.chunk_store.get_row(self.passage_keys[i])["content"],
                score=float(final[i]),
            )
            for i in order
            if final[i] > 0
        ]

    def _dense_top_k(self, dense: np.ndarray, top_k: int) -> List[RetrievedPassage]:
        order = np.argsort(dense)[::-1][:top_k]
        return [
            RetrievedPassage(
                chunk_id=self.passage_keys[i],
                text=self.chunk_store.get_row(self.passage_keys[i])["content"],
                score=float(dense[i]),
            )
            for i in order
        ]
