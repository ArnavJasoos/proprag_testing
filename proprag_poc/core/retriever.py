"""Two-stage, LLM-free retrieval (ported from the reference
``graph_search_with_proposition_entities`` + ``run_ppr``):

DPR passage prior -> beam search -> path entities seed PPR #1 (full graph,
d=0.75) -> focused subgraph of top passages -> beam #2 -> PPR #2 (d=0.45) ->
merged passage ranking. Falls back to dense passage retrieval if graph signal
is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..config import POCConfig
from ..embedding.encoder import EmbeddingModel
from .beam_search import BeamSearchPathFinder
from .ids import compute_mdhash_id

logger = logging.getLogger(__name__)


def _min_max(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


@dataclass
class RetrievedPassage:
    chunk_id: str
    text: str
    score: float


class Retriever:
    system_name = "PropRAG"

    def __init__(self, corpus, embedding_model: EmbeddingModel, config: POCConfig):
        self.config = config
        self.embedding_model = embedding_model
        self.chunk_embedding_store = corpus.chunk_store
        self.entity_embedding_store = corpus.entity_store
        self.proposition_embedding_store = corpus.proposition_store
        self.graph = corpus.graph
        self.proposition_to_entities_map = corpus.proposition_to_entities_map
        self.chunk_propositions = corpus.chunk_propositions
        self._prepare()

    # ----------------------------------------------------------------- setup
    def _prepare(self):
        self.entity_node_keys = self.entity_embedding_store.get_all_ids()
        self.passage_node_keys = self.chunk_embedding_store.get_all_ids()
        self.proposition_node_keys = self.proposition_embedding_store.get_all_ids()

        self.node_name_to_vertex_idx = {v["name"]: i for i, v in enumerate(self.graph.vs)}
        self.passage_node_idxs = [
            self.node_name_to_vertex_idx[k]
            for k in self.passage_node_keys
            if k in self.node_name_to_vertex_idx
        ]
        self.passage_key_to_row = {k: i for i, k in enumerate(self.passage_node_keys)}

        self.passage_embeddings = self.chunk_embedding_store.get_embeddings(self.passage_node_keys)
        prop_embs = self.proposition_embedding_store.get_embeddings(self.proposition_node_keys)
        self.prop_key_to_propositions = {
            k: prop_embs[i] for i, k in enumerate(self.proposition_node_keys)
        }
        self.all_proposition_embeddings = prop_embs

        self.beam_search = BeamSearchPathFinder(
            self,
            beam_width=self.config.beam_width,
            max_path_length=self.config.max_path_length,
            embedding_combination=self.config.embedding_combination,
            second_stage_filter_k=self.config.second_stage_filter_k,
        )

    # --------------------------------------------------------- dense passages
    def _dpr_scores(self, query: str) -> np.ndarray:
        q = self.embedding_model.batch_encode(
            query, instruction=self.config.embedding_query_instruction, norm=True,
            input_type="query",
        )
        scores = (self.passage_embeddings @ q.squeeze().T).squeeze()
        return _min_max(np.atleast_1d(scores))  # aligned to passage_node_keys

    # --------------------------------------------------------------- PPR
    def _run_ppr(self, graph, reset_prob, damping):
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        if reset_prob.sum() == 0:
            return None
        return np.array(
            graph.personalized_pagerank(
                vertices=range(graph.vcount()),
                damping=damping,
                directed=False,
                weights="weight",
                reset=list(reset_prob),
                implementation="prpack",
            )
        )

    # --------------------------------------------------------------- retrieve
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedPassage]:
        top_k = top_k or self.config.retrieval_top_k
        n_vert = self.graph.vcount()
        dpr = self._dpr_scores(query)

        # Stage 1: beam over full graph -> entity seeds.
        paths = self.beam_search.find_paths(query)[: self.config.select_top_k_paths]
        top_entities = self.beam_search.get_entities_from_paths(paths)[
            : self.config.select_top_k_entities
        ]

        phrase_weights = np.zeros(n_vert)
        for ekey, _ in top_entities:
            idx = self.node_name_to_vertex_idx.get(ekey)
            if idx is not None:
                phrase_weights[idx] = 1.0
        phrase_weights = _min_max(phrase_weights)

        passage_weights = np.zeros(n_vert)
        for row, pkey in enumerate(self.passage_node_keys):
            idx = self.node_name_to_vertex_idx.get(pkey)
            if idx is not None:
                passage_weights[idx] = dpr[row] * self.config.passage_node_weight

        node_weights = phrase_weights + passage_weights
        ppr1 = self._run_ppr(self.graph, node_weights, self.config.ppr_damping_stage1)
        if ppr1 is None:
            return self._dpr_fallback(dpr, top_k)

        doc_scores = np.array([ppr1[i] for i in self.passage_node_idxs])
        final = self._focused_stage(query, doc_scores, dpr)
        order = np.argsort(final)[::-1][:top_k]
        return [
            RetrievedPassage(
                chunk_id=self.passage_node_keys[i],
                text=self.chunk_embedding_store.get_row(self.passage_node_keys[i])["content"],
                score=float(final[i]),
            )
            for i in order
        ]

    # ---------------------------------------------------- focused 2nd stage
    def _focused_stage(self, query, first_doc_scores, dpr):
        focus = self.config.focus_top_k
        top_doc_rows = np.argsort(first_doc_scores)[::-1][:focus]
        top_doc_keys = [self.passage_node_keys[r] for r in top_doc_rows]
        valid = [k for k in top_doc_keys if k in self.node_name_to_vertex_idx]
        if not valid:
            return first_doc_scores

        # Propositions belonging to the focused passages.
        focus_props = []
        for k in valid:
            for prop in self.chunk_propositions.get(k, []):
                focus_props.append(compute_mdhash_id(prop["text"], prefix="proposition-"))
        focus_props = [p for p in focus_props if p in self.prop_key_to_propositions]
        if not focus_props:
            return first_doc_scores

        # Subgraph: focused passages + their entity neighbors.
        doc_vertices = [self.node_name_to_vertex_idx[k] for k in valid]
        include = set(doc_vertices)
        for dv in doc_vertices:
            for nb in self.graph.neighbors(dv, mode="all"):
                if self.graph.vs[nb]["name"].startswith("entity-"):
                    include.add(nb)
        sub = self.graph.induced_subgraph(list(include))
        if sub.vcount() == 0:
            return first_doc_scores
        sub_name_to_idx = {v["name"]: i for i, v in enumerate(sub.vs)}

        # Re-target beam search onto the subgraph + focused proposition set.
        bs = self.beam_search
        orig_graph, orig_map = bs.active_graph, bs.node_name_to_vertex_idx
        bs.active_graph = sub
        bs.set_node_name_to_vertex_idx(sub_name_to_idx)
        bs.clear_caches()
        try:
            paths2 = bs.find_paths(query, prop_set=focus_props)[:5]
            ents2 = bs.get_entities_from_paths(paths2)[:5]
        finally:
            bs.active_graph = orig_graph
            bs.set_node_name_to_vertex_idx(orig_map)
            bs.clear_caches()

        sub_phrase = np.zeros(sub.vcount())
        for ekey, scores in ents2:
            si = sub_name_to_idx.get(ekey)
            if si is not None:
                sub_phrase[si] = max(scores) if scores else 0.0
        if sub_phrase.sum() > 0:
            sub_phrase = _min_max(sub_phrase)

        sub_passage = np.zeros(sub.vcount())
        for row, pkey in enumerate(self.passage_node_keys):
            si = sub_name_to_idx.get(pkey)
            if si is not None:
                sub_passage[si] = dpr[row] * self.config.passage_node_weight

        sub_weights = sub_phrase + sub_passage
        sub_ppr = self._run_ppr(sub, sub_weights, self.config.ppr_damping_stage2)
        if sub_ppr is None:
            return first_doc_scores

        # Merge: focused PPR overrides scores for passages in the subgraph;
        # scale the rest below the focused floor.
        floor = float(np.min(sub_ppr)) * 0.5
        final = first_doc_scores * floor
        for si, score in enumerate(sub_ppr):
            name = sub.vs[si]["name"]
            if name.startswith("chunk-") and name in self.passage_key_to_row:
                final[self.passage_key_to_row[name]] = score
        return final

    def _dpr_fallback(self, dpr, top_k):
        order = np.argsort(dpr)[::-1][:top_k]
        return [
            RetrievedPassage(
                chunk_id=self.passage_node_keys[i],
                text=self.chunk_embedding_store.get_row(self.passage_node_keys[i])["content"],
                score=float(dpr[i]),
            )
            for i in order
        ]
