"""GraphRAG LOCAL search: query -> entity match (by embedding) -> chunk scoring.

Retrieval-comparable with the other systems (Recall@k is computed from
``retrieve()`` which returns documents only). ``build_qa_context`` assembles the
faithful local-search context - community reports + relationship lines + chunks -
within the same ``qa_top_k`` budget used by BaseRAG/PropRAG.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from proprag_poc.core.retriever import RetrievedPassage, _min_max

from .index import GraphRAGIndex

logger = logging.getLogger(__name__)


class GraphRAGLocalRetriever:
    system_name = "GraphRAG"

    def __init__(self, index: GraphRAGIndex, embedding_model, poc_cfg, bench_cfg):
        self.index = index
        self.embedding_model = embedding_model
        self.poc_cfg = poc_cfg
        self.cfg = bench_cfg

        self.passage_keys = index.chunk_store.get_all_ids()
        self.passage_row = {k: i for i, k in enumerate(self.passage_keys)}
        self.passage_embeddings = index.chunk_store.get_embeddings(self.passage_keys)

        self.entity_row = {k: i for i, k in enumerate(index.entity_order)}
        self.rel_lookup: Dict[frozenset, object] = {
            frozenset((r.src_key, r.dst_key)): r for r in index.relationships
        }

    # ------------------------------------------------------------- helpers
    def _embed_query(self, query: str) -> np.ndarray:
        return self.embedding_model.batch_encode(
            query, instruction=self.poc_cfg.embedding_query_instruction,
            norm=True, input_type="query",
        ).squeeze()

    def _match_entities(self, q: np.ndarray) -> List[Tuple[str, float]]:
        if self.index.entity_embeddings.shape[0] == 0 or self.index.entity_embeddings.ndim != 2:
            return []
        sims = self.index.entity_embeddings @ q.T
        order = np.argsort(sims)[::-1][: self.cfg.gr_local_top_entities]
        out = []
        for i in order:
            score = float(sims[i])
            if score < self.cfg.gr_entity_sim_floor:
                break
            out.append((self.index.entity_order[i], score))
        return out

    def _dense_scores(self, q: np.ndarray) -> np.ndarray:
        return np.atleast_1d((self.passage_embeddings @ q.T).squeeze())

    # ------------------------------------------------------------- retrieve
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedPassage]:
        top_k = top_k or self.poc_cfg.retrieval_top_k
        q = self._embed_query(query)
        dense = _min_max(self._dense_scores(q))
        matched = self._match_entities(q)

        if not matched:
            order = np.argsort(dense)[::-1][:top_k]
            return [self._passage(i, float(dense[i])) for i in order]

        graph_score = np.zeros(len(self.passage_keys))
        for ekey, sim in matched:
            for cid in self.index.entity_chunks.get(ekey, []):
                row = self.passage_row.get(cid)
                if row is not None:
                    graph_score[row] += sim

        # Bonus for chunks cited by a relationship between two matched entities.
        matched_keys = [k for k, _ in matched]
        for a in range(len(matched_keys)):
            for b in range(a + 1, len(matched_keys)):
                rel = self.rel_lookup.get(frozenset((matched_keys[a], matched_keys[b])))
                if rel is None:
                    continue
                for cid in rel.chunk_ids:
                    row = self.passage_row.get(cid)
                    if row is not None:
                        graph_score[row] += 0.5

        if graph_score.sum() == 0:
            order = np.argsort(dense)[::-1][:top_k]
            return [self._passage(i, float(dense[i])) for i in order]

        final = _min_max(graph_score) + self.cfg.gr_dense_blend * dense
        order = np.argsort(final)[::-1][:top_k]
        return [self._passage(i, float(final[i])) for i in order]

    def _passage(self, row: int, score: float) -> RetrievedPassage:
        key = self.passage_keys[row]
        return RetrievedPassage(
            chunk_id=key,
            text=self.index.chunk_store.get_row(key)["content"],
            score=score,
        )

    # ---------------------------------------------------------- QA context
    def build_qa_context(self, query: str, passages: List[RetrievedPassage]) -> List[str]:
        """Local-search context within the qa_top_k budget: community reports +
        relationship lines + top chunks."""
        budget = self.cfg.qa_top_k
        q = self._embed_query(query)
        matched = self._match_entities(q)
        matched_keys = [k for k, _ in matched]

        context: List[str] = []

        # 1-2 community reports covering the matched entities (highest rating first).
        seen_comm = set()
        comm_reports = []
        for ekey in matched_keys:
            cid = self.index.communities.get(ekey)
            if cid is None or cid in seen_comm:
                continue
            report = self.index.reports.get(int(cid))
            if report:
                seen_comm.add(cid)
                comm_reports.append(report)
        comm_reports.sort(key=lambda r: r.get("rating", 0), reverse=True)
        for report in comm_reports[:2]:
            context.append(f"[Community report] {report.get('title', '')}: {report.get('summary', '')}")

        # Relationship one-liners between matched entities.
        rel_lines = []
        for a in range(len(matched_keys)):
            for b in range(a + 1, len(matched_keys)):
                rel = self.rel_lookup.get(frozenset((matched_keys[a], matched_keys[b])))
                if rel is None:
                    continue
                na = self.index.entities[rel.src_key].name
                nb = self.index.entities[rel.dst_key].name
                desc = " ; ".join(dict.fromkeys(rel.descriptions)) or "related"
                rel_lines.append(f"{na} -> {nb}: {desc}")
        if rel_lines:
            context.append("[Relationships]\n" + "\n".join(rel_lines[:10]))

        # Fill the remaining budget with top retrieved chunks.
        for p in passages:
            if len(context) >= budget:
                break
            context.append(p.text)
        return context[:budget] if context else [p.text for p in passages[:budget]]
