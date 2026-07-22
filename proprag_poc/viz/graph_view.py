"""PyVis ego-graph visualization around an extracted entity."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pyvis.network import Network

from ..core.ids import compute_mdhash_id


def find_entity_key(corpus, query: str) -> Optional[str]:
    """Resolve a free-text entity query to an entity node key (exact, then substring)."""
    rows = corpus.entity_store.get_text_for_all_rows()
    q = query.strip().lower()
    exact = compute_mdhash_id(query.strip(), prefix="entity-")
    if exact in rows:
        return exact
    for key, row in rows.items():
        if q == row["content"].lower():
            return key
    for key, row in rows.items():
        if q in row["content"].lower():
            return key
    return None


def list_entities(corpus, limit: int = 500) -> List[str]:
    rows = corpus.entity_store.get_text_for_all_rows()
    return sorted(r["content"] for r in rows.values())[:limit]


def ego_graph_html(corpus, entity_key: str, hops: int = 1, max_nodes: int = 60) -> str:
    g = corpus.graph
    name_to_idx = {v["name"]: i for i, v in enumerate(g.vs)}
    if entity_key not in name_to_idx:
        return "<p>Entity not found in graph.</p>"

    center = name_to_idx[entity_key]
    # BFS neighborhood up to `hops`, capped at max_nodes.
    frontier = {center}
    visited = {center}
    for _ in range(hops):
        nxt = set()
        for v in frontier:
            for nb in g.neighbors(v, mode="all"):
                if nb not in visited:
                    nxt.add(nb)
        visited |= nxt
        frontier = nxt
        if len(visited) >= max_nodes:
            break
    nodes = list(visited)[:max_nodes]
    node_set = set(nodes)

    net = Network(height="600px", width="100%", bgcolor="#ffffff", directed=False)
    net.barnes_hut(spring_length=120)
    for v in nodes:
        name = g.vs[v]["name"]
        is_entity = name.startswith("entity-")
        label = _node_label(corpus, name, is_entity)
        net.add_node(
            v,
            label=label if is_entity else "[passage]",
            title=label,
            color="#e8743b" if v == center else ("#1f9e89" if is_entity else "#6c6c6c"),
            shape="dot" if is_entity else "square",
            size=22 if v == center else (14 if is_entity else 10),
        )
    for e in g.es:
        s, t = e.tuple
        if s in node_set and t in node_set:
            w = e["weight"]
            syn = isinstance(w, float) and not float(w).is_integer()
            net.add_edge(s, t, color="#b59ad6" if syn else "#c7c7c7",
                         title=("synonym %.3f" % w) if syn else ("co-occur %g" % w))
    return net.generate_html(notebook=False)


def _node_label(corpus, name: str, is_entity: bool) -> str:
    store = corpus.entity_store if is_entity else corpus.chunk_store
    row = store.get_row(name)
    text = row["content"] if row else name
    return text if len(text) <= 60 else text[:57] + "..."
