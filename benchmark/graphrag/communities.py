"""Leiden community detection (igraph built-in, no leidenalg) + LLM reports."""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

import igraph as ig

from .extract import EntityRec, RelRec
from . import prompts

logger = logging.getLogger(__name__)


def build_graph(entities: Dict[str, EntityRec], rels: List[RelRec]) -> ig.Graph:
    names = list(entities.keys())
    idx = {n: i for i, n in enumerate(names)}
    g = ig.Graph()
    g.add_vertices(len(names))
    g.vs["name"] = names

    edges, weights = [], []
    for rel in rels:
        if rel.src_key in idx and rel.dst_key in idx and rel.src_key != rel.dst_key:
            edges.append((idx[rel.src_key], idx[rel.dst_key]))
            weights.append(float(rel.weight) if rel.weight > 0 else 1.0)
    g.add_edges(edges)
    g.es["weight"] = weights
    logger.info("GraphRAG entity graph: %d nodes, %d edges", g.vcount(), g.ecount())
    return g


def detect_communities(g: ig.Graph) -> Dict[str, int]:
    """entity key -> community id via weighted-modularity Leiden."""
    if g.vcount() == 0:
        return {}
    if g.ecount() == 0:
        return {g.vs[i]["name"]: i for i in range(g.vcount())}
    clustering = g.community_leiden(
        objective_function="modularity", weights=g.es["weight"], n_iterations=5
    )
    membership = clustering.membership
    n = len(set(membership))
    logger.info("GraphRAG detected %d communities over %d entities", n, g.vcount())
    return {g.vs[i]["name"]: membership[i] for i in range(g.vcount())}


def _loads_lenient(raw: str):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    return json.loads(s)


def community_report(
    llm,
    member_keys: List[str],
    entities: Dict[str, EntityRec],
    intra_rels: List[RelRec],
    cfg,
) -> Dict:
    """Generate a report for one community; degrade to a minimal report on failure."""
    ent_lines = []
    for k in member_keys:
        rec = entities.get(k)
        if rec is None:
            continue
        ent_lines.append(f"{rec.name} ({rec.type}): {rec.merged_description()}")
    entities_block = "\n".join(ent_lines)

    rel_lines = [
        f"{entities[r.src_key].name} -> {entities[r.dst_key].name}: "
        f"{' ; '.join(dict.fromkeys(r.descriptions)) or 'related'} (w={r.weight:.0f})"
        for r in sorted(intra_rels, key=lambda x: x.weight, reverse=True)
        if r.src_key in entities and r.dst_key in entities
    ]
    rels_block = "\n".join(rel_lines)

    # Truncate the combined context to the configured character budget.
    budget = cfg.gr_report_max_input_chars
    if len(entities_block) + len(rels_block) > budget:
        entities_block = entities_block[: budget // 2]
        rels_block = rels_block[: budget // 2]

    try:
        raw, _, _ = llm.infer(
            prompts.community_report_messages(entities_block, rels_block),
            json_mode=True,
            max_completion_tokens=cfg.report_max_tokens,
        )
        report = _loads_lenient(raw)
        if not isinstance(report, dict):
            raise ValueError("report is not a JSON object")
        report.setdefault("title", f"Community of {member_keys[0] if member_keys else '?'}")
        report.setdefault("summary", "")
        report.setdefault("rating", 0)
        report.setdefault("findings", [])
        return report
    except Exception as e:  # noqa: BLE001
        logger.warning("community report failed, using stub: %s", e)
        titles = ", ".join(entities[k].name for k in member_keys[:3] if k in entities)
        return {
            "title": titles or "Community",
            "summary": entities_block[:400],
            "rating": 0,
            "findings": [],
            "parse_failed": True,
        }
