"""Build the undirected PropRAG knowledge graph with igraph.

Node types: entity-<md5>, chunk-<md5> (propositions are NOT graph nodes; they
live in the embedding store + maps). Edge weights:
  - entity-entity co-occurrence: INTEGER count (entities sharing a proposition).
  - entity-passage: 1.0.
  - entity-entity synonymy: FLOAT similarity in [threshold, 1).
The integer-vs-float distinction is how beam search later identifies synonymy
edges, so the invariant must hold.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import igraph as ig
import numpy as np

from ..config import POCConfig
from .ids import compute_mdhash_id
from .store import EmbeddingStore

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, config: POCConfig):
        self.config = config

    def build(
        self,
        chunk_ids: List[str],
        chunk_propositions: Dict[str, List[Dict]],
        entity_store: EmbeddingStore,
        chunk_store: EmbeddingStore,
    ) -> ig.Graph:
        stats: Dict[Tuple[str, str], float] = {}

        # 1) entity-entity co-occurrence (fully connect entities within a prop).
        for chunk_id in chunk_ids:
            for prop in chunk_propositions.get(chunk_id, []):
                ekeys = [
                    compute_mdhash_id(e, prefix="entity-") for e in prop["entities"]
                ]
                for i in range(len(ekeys)):
                    for j in range(i + 1, len(ekeys)):
                        a, b = sorted((ekeys[i], ekeys[j]))
                        if a == b:
                            continue
                        stats[(a, b)] = stats.get((a, b), 0.0) + 1.0

        # 2) entity-passage edges.
        for chunk_id in chunk_ids:
            ents = set()
            for prop in chunk_propositions.get(chunk_id, []):
                for e in prop["entities"]:
                    ents.add(compute_mdhash_id(e, prefix="entity-"))
            for ekey in ents:
                stats[(chunk_id, ekey)] = 1.0

        # 3) entity-entity synonymy (KNN, float weights; only where no edge yet).
        self._add_synonymy(stats, entity_store)

        return self._assemble(stats, entity_store, chunk_store)

    # ------------------------------------------------------------- synonymy
    def _add_synonymy(self, stats, entity_store: EmbeddingStore):
        ids = entity_store.get_all_ids()
        if len(ids) < 2:
            return
        embs = entity_store.get_embeddings(ids)  # normalized
        sims = embs @ embs.T
        thr = self.config.synonymy_sim_threshold
        topk = self.config.synonymy_top_k
        np.fill_diagonal(sims, -1.0)
        for i, id_i in enumerate(ids):
            row = sims[i]
            cand = np.argsort(row)[::-1][:topk]
            for j in cand:
                score = float(row[j])
                if score < thr:
                    break
                a, b = sorted((id_i, ids[j]))
                if (a, b) in stats:  # keep integer co-occurrence invariant
                    continue
                # Ensure strictly non-integer so beam search reads it as synonymy.
                stats[(a, b)] = min(0.999999, max(thr, score))

    # ------------------------------------------------------------- assemble
    def _assemble(self, stats, entity_store, chunk_store) -> ig.Graph:
        names = list(entity_store.get_all_ids()) + list(chunk_store.get_all_ids())
        name_to_idx = {n: i for i, n in enumerate(names)}
        g = ig.Graph(directed=self.config.is_directed_graph)
        g.add_vertices(len(names))
        g.vs["name"] = names

        edges, weights = [], []
        for (a, b), w in stats.items():
            if a in name_to_idx and b in name_to_idx and a != b:
                edges.append((name_to_idx[a], name_to_idx[b]))
                weights.append(float(w))
        g.add_edges(edges)
        g.es["weight"] = weights
        logger.info("Graph built: %d nodes, %d edges", g.vcount(), g.ecount())
        return g
