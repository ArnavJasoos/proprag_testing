"""LLM-free beam search over proposition paths (ported from the reference
BeamSearchPathFinder; the trained-predictor combination mode is removed).

Paths grow by hopping to propositions that share an entity (or a synonymous
entity) with the path's last proposition. Each path is scored by a combined
embedding of its propositions against the query.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from .ids import compute_mdhash_id

logger = logging.getLogger(__name__)


class BeamSearchPathFinder:
    def __init__(self, rag, beam_width=4, max_path_length=3,
                 embedding_combination="concatenate", sim_threshold=0.75,
                 second_stage_filter_k=0):
        self.rag = rag
        self.beam_width = beam_width
        self.max_path_length = max_path_length
        self.embedding_combination = embedding_combination
        self.sim_threshold = sim_threshold
        self.second_stage_filter_k = second_stage_filter_k
        self.active_graph = rag.graph
        self.node_name_to_vertex_idx = rag.node_name_to_vertex_idx

        # Inverse map: entity-key -> [proposition-key].
        self.entity_to_propositions_map = defaultdict(list)
        for prop_key, entities in rag.proposition_to_entities_map.items():
            for entity in entities:
                ekey = compute_mdhash_id(entity, prefix="entity-")
                self.entity_to_propositions_map[ekey].append(prop_key)
        self.clear_caches()

    # ----------------------------------------------------------------- setup
    def set_node_name_to_vertex_idx(self, mapping):
        self.node_name_to_vertex_idx = mapping

    def clear_caches(self):
        self.synonymous_entities_cache = {}
        self.connected_propositions_cache = {}

    # --------------------------------------------------------------- lookups
    def get_proposition_text(self, prop_key: str) -> str:
        row = self.rag.proposition_embedding_store.get_row(prop_key)
        return row["content"] if row else ""

    def get_entity_text(self, entity_key: str) -> str:
        row = self.rag.entity_embedding_store.get_row(entity_key)
        return row["content"] if row else ""

    def get_proposition_embeddings(self, prop_keys: List[str]) -> np.ndarray:
        return np.array([self.rag.prop_key_to_propositions[k] for k in prop_keys])

    def find_entities_in_proposition(self, prop_key: str) -> List[str]:
        ents = self.rag.proposition_to_entities_map.get(prop_key, [])
        return [compute_mdhash_id(e, prefix="entity-") for e in ents]

    def find_synonymous_entities(self, entity_key: str) -> List[Tuple[str, float]]:
        if entity_key in self.synonymous_entities_cache:
            return self.synonymous_entities_cache[entity_key]
        synonyms = []
        idx = self.node_name_to_vertex_idx.get(entity_key)
        if idx is not None:
            try:
                for nb in self.active_graph.neighbors(idx, mode="all"):
                    name = self.active_graph.vs[nb]["name"]
                    if not name.startswith("entity-"):
                        continue
                    eid = self.active_graph.get_eid(idx, nb, error=False)
                    if eid == -1:
                        continue
                    w = self.active_graph.es[eid]["weight"]
                    if isinstance(w, float) and not float(w).is_integer() and w >= self.sim_threshold:
                        synonyms.append((name, float(w)))
            except Exception as e:  # noqa: BLE001
                logger.debug("synonym lookup failed: %s", e)
        self.synonymous_entities_cache[entity_key] = synonyms
        return synonyms

    def find_connected_propositions(self, prop_key: str, prop_set=None):
        if prop_key in self.connected_propositions_cache:
            cached, processed = self.connected_propositions_cache[prop_key]
            return cached, processed
        connected, processed = [], set()
        for ekey in self.find_entities_in_proposition(prop_key):
            for other in self.entity_to_propositions_map.get(ekey, []):
                if other != prop_key and other not in processed and (
                    prop_set is None or other in prop_set
                ):
                    connected.append((other, [{"type": "exact", "entity1": ekey,
                                               "entity2": ekey, "similarity": 1.0}]))
                processed.add(other)
            for syn, sim in self.find_synonymous_entities(ekey):
                for other in self.entity_to_propositions_map.get(syn, []):
                    if other != prop_key and other not in processed and (
                        prop_set is None or other in prop_set
                    ):
                        connected.append((other, [{"type": "synonym", "entity1": ekey,
                                                   "entity2": syn, "similarity": sim}]))
                    processed.add(other)
        self.connected_propositions_cache[prop_key] = (connected, processed)
        return connected, processed

    # --------------------------------------------------------------- scoring
    def batch_score_paths(self, paths: List[Dict], query_embedding: np.ndarray) -> List[float]:
        if not paths:
            return []
        prop_lists = [p["propositions"] for p in paths]
        if self.embedding_combination == "concatenate":
            texts = [" ".join(self.get_proposition_text(k) for k in props) for props in prop_lists]
            combined = self.rag.embedding_model.batch_encode(texts, norm=True)
        else:
            batch = np.array([list(self.get_proposition_embeddings(props)) for props in prop_lists])
            if self.embedding_combination == "average":
                combined = batch.mean(axis=1)
            elif self.embedding_combination == "weighted_average":
                w = np.linspace(0.5, 1.0, batch.shape[1])
                w = w / w.sum()
                combined = (batch * w.reshape(1, -1, 1)).sum(axis=1)
            elif self.embedding_combination == "max_pool":
                idx = np.argmax(np.abs(batch), axis=1)
                combined = np.take_along_axis(batch, idx[:, None, :], axis=1).squeeze(1)
            elif self.embedding_combination == "attention":
                qv = query_embedding.squeeze()
                att = np.exp(batch @ qv)
                att = att / att.sum(axis=1, keepdims=True)
                combined = (batch * att[:, :, None]).sum(axis=1)
            else:
                combined = batch.mean(axis=1)
            combined = combined / np.clip(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12, None)
        return (combined @ query_embedding.squeeze()).tolist()

    # --------------------------------------------------------------- search
    def find_paths(self, query: str, prop_set: List[str] = None):
        q = self.rag.embedding_model.batch_encode(
            query, instruction=self.rag.config.embedding_query_instruction, norm=True,
            input_type="query",
        )
        if prop_set is None:
            all_keys = self.rag.proposition_embedding_store.get_all_ids()
            scores = (self.rag.all_proposition_embeddings @ q.T).squeeze()
        else:
            all_keys = prop_set
            scores = (self.get_proposition_embeddings(all_keys) @ q.T).squeeze()
        scores = np.atleast_1d(scores)

        top_k = min(self.rag.config.initial_proposition_seeds, len(scores))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        beam = [{"propositions": [all_keys[i]], "connections": [], "score": float(scores[i])}
                for i in top_idx]

        initial_props = [(p["propositions"][0], []) for p in beam[:3]]
        beam = beam[:self.beam_width]
        all_paths, seen = [], set()
        for p in beam:
            fs = frozenset(p["propositions"])
            if fs not in seen:
                seen.add(fs)
                all_paths.append(p.copy())

        for depth in range(2, self.max_path_length + 1):
            new_candidates = []
            by_last = defaultdict(list)
            for p in beam:
                by_last[p["propositions"][-1]].append(p)
            for last_prop, paths_here in by_last.items():
                connected, conn_set = self.find_connected_propositions(last_prop, prop_set)
                connected = list(connected)
                for ip in initial_props:
                    if last_prop != ip[0] and ip[0] not in conn_set:
                        connected.append(ip)
                for path in paths_here:
                    for next_prop, conns in connected:
                        if next_prop in path["propositions"]:
                            continue
                        new_path = {
                            "propositions": path["propositions"] + [next_prop],
                            "connections": path["connections"] + [
                                {"from_prop": last_prop, "to_prop": next_prop,
                                 "entity_connections": conns}],
                        }
                        fs = frozenset(new_path["propositions"])
                        if fs in seen:
                            continue
                        seen.add(fs)
                        new_path["score"] = 0.0
                        new_candidates.append(new_path)
            if not new_candidates:
                break
            for i, s in enumerate(self.batch_score_paths(new_candidates, q)):
                new_candidates[i]["score"] = float(s)
            new_candidates.sort(key=lambda x: x["score"], reverse=True)

            if self.second_stage_filter_k > 0 and len(new_candidates) > self.beam_width:
                first = new_candidates[: self.second_stage_filter_k]
                # The second-stage re-rank refines with "concatenate" (re-embeds joined
                # path text). That means one fresh embedding API call per candidate;
                # under an online rate-limited embedder it dominates latency/cost, so
                # only do it for a local embedder. Online keeps the pooled scores.
                if not self.rag.config.embedding_is_online:
                    saved = self.embedding_combination
                    self.embedding_combination = "concatenate"
                    for i, s in enumerate(self.batch_score_paths(first, q)):
                        first[i]["score"] = float(s)
                    self.embedding_combination = saved
                first.sort(key=lambda x: x["score"], reverse=True)
                new_candidates = first

            beam = new_candidates[: self.beam_width]
            all_paths.extend(p.copy() for p in beam)

        all_paths.sort(key=lambda x: x["score"], reverse=True)
        return self._post_process(all_paths)

    def _post_process(self, all_paths):
        out = []
        for path in all_paths:
            out.append({
                "proposition_keys": path["propositions"],
                "proposition_texts": [self.get_proposition_text(k) for k in path["propositions"]],
                "connections": [{
                    "entity_connections": [{
                        "entity1": self.get_entity_text(c["entity1"]),
                        "entity2": self.get_entity_text(c["entity2"]),
                        "type": c["type"], "similarity": c.get("similarity", 1.0),
                    } for c in conn["entity_connections"]],
                } for conn in path["connections"]],
                "score": path.get("score", 0.0),
            })
        return out

    # --------------------------------------------------------- path → entities
    def get_entities_from_paths(self, paths) -> List[Tuple[str, List[float]]]:
        entity_scores = defaultdict(list)
        for path in paths:
            ps = path["score"]
            for conn in path.get("connections", []):
                for c in conn["entity_connections"]:
                    if c["entity1"] != c["entity2"]:
                        entity_scores[compute_mdhash_id(c["entity2"], prefix="entity-")].append(ps)
            for prop_key in path["proposition_keys"]:
                for ekey in self.find_entities_in_proposition(prop_key):
                    entity_scores[ekey].append(ps)
        return sorted(entity_scores.items(), key=lambda x: np.sum(x[1]), reverse=True)
