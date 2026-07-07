"""Typed entity + relationship extraction with a multi-tier JSON-repair chain.

Mirrors the repair strategy proven in ``proprag_poc/core/extraction.py``:
json_mode -> lenient parse -> ``fix_json`` repair call -> regex object salvage ->
empty result with a counted warning. Extraction workers set the
``index::GraphRAG`` usage scope so token attribution survives the thread pool.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from proprag_poc.llm import prompts as poc_prompts
from . import prompts

logger = logging.getLogger(__name__)

_ENTITY_OBJ = re.compile(
    r'\{\s*"name"\s*:\s*"(?P<name>(?:\\.|[^"\\])*)"\s*,\s*"type"\s*:\s*'
    r'"(?P<type>(?:\\.|[^"\\])*)"\s*,\s*"description"\s*:\s*'
    r'"(?P<desc>(?:\\.|[^"\\])*)"',
    re.DOTALL,
)
_REL_OBJ = re.compile(
    r'\{\s*"source"\s*:\s*"(?P<src>(?:\\.|[^"\\])*)"\s*,\s*"target"\s*:\s*'
    r'"(?P<dst>(?:\\.|[^"\\])*)"\s*,\s*"description"\s*:\s*'
    r'"(?P<desc>(?:\\.|[^"\\])*)"(?:\s*,\s*"strength"\s*:\s*(?P<strength>\d+))?',
    re.DOTALL,
)


@dataclass
class EntityRec:
    key: str
    name: str
    type: str
    descriptions: List[str] = field(default_factory=list)
    chunk_ids: Set[str] = field(default_factory=set)
    summary: str = ""

    def merged_description(self) -> str:
        return self.summary or " ".join(dict.fromkeys(self.descriptions))


@dataclass
class RelRec:
    src_key: str
    dst_key: str
    descriptions: List[str] = field(default_factory=list)
    weight: float = 0.0
    chunk_ids: Set[str] = field(default_factory=set)


def _loads_lenient(raw: str):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    return json.loads(s)


def _normalize(obj) -> Dict[str, list]:
    entities, rels = [], []
    if isinstance(obj, dict):
        for e in obj.get("entities", []) or []:
            if isinstance(e, dict) and e.get("name"):
                entities.append(e)
        for r in obj.get("relationships", []) or []:
            if isinstance(r, dict) and r.get("source") and r.get("target"):
                rels.append(r)
    return {"entities": entities, "relationships": rels}


def parse_extraction(raw: str) -> Dict[str, list]:
    """Lenient JSON parse; fall back to per-object regex salvage."""
    try:
        parsed = _normalize(_loads_lenient(raw))
        if parsed["entities"] or parsed["relationships"]:
            return parsed
    except Exception:  # noqa: BLE001
        pass

    entities = [
        {"name": m.group("name"), "type": m.group("type"), "description": m.group("desc")}
        for m in _ENTITY_OBJ.finditer(raw)
    ]
    rels = []
    for m in _REL_OBJ.finditer(raw):
        rels.append({
            "source": m.group("src"),
            "target": m.group("dst"),
            "description": m.group("desc"),
            "strength": int(m.group("strength")) if m.group("strength") else 5,
        })
    return {"entities": entities, "relationships": rels}


def extract_chunk(llm, text: str, cfg) -> Tuple[Dict[str, list], bool]:
    """Return (parsed, failed). ``failed`` is True only when nothing was salvaged."""
    raw, _, _ = llm.infer(
        prompts.extraction_messages(text, list(cfg.gr_entity_types)),
        json_mode=True,
        max_completion_tokens=cfg.extract_max_tokens,
    )
    parsed = parse_extraction(raw)
    if not parsed["entities"] and not parsed["relationships"]:
        fixed, _, _ = llm.infer(poc_prompts.fix_json_messages(raw), json_mode=True)
        parsed = parse_extraction(fixed)

    failed = not parsed["entities"] and not parsed["relationships"]

    for _ in range(max(0, cfg.gr_max_gleanings)):
        prev = json.dumps(parsed)
        graw, _, _ = llm.infer(
            prompts.gleaning_messages(text, prev, list(cfg.gr_entity_types)),
            json_mode=True,
            max_completion_tokens=cfg.extract_max_tokens,
        )
        extra = parse_extraction(graw)
        parsed["entities"].extend(extra["entities"])
        parsed["relationships"].extend(extra["relationships"])

    return parsed, failed


def _ekey(name: str) -> str:
    return name.strip().lower()


def batch_extract(
    llm, chunk_texts: Dict[str, str], cfg, tracker
) -> Tuple[Dict[str, EntityRec], List[RelRec], int]:
    """Parallel extraction + merge. Returns (entities, relationships, n_failures)."""
    keys = list(chunk_texts.keys())
    max_workers = llm.config.llm_max_workers
    logger.info("GraphRAG extract: %d chunks, %d workers", len(keys), max_workers)

    def _work(cid: str) -> Tuple[str, Dict[str, list], bool]:
        with tracker.scope("index::GraphRAG"):
            parsed, failed = extract_chunk(llm, chunk_texts[cid], cfg)
        return cid, parsed, failed

    per_chunk: Dict[str, Dict[str, list]] = {}
    n_failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_work, cid): cid for cid in keys}
        for done, fut in enumerate(as_completed(futs), 1):
            cid, parsed, failed = fut.result()
            per_chunk[cid] = parsed
            n_failures += int(failed)
            logger.info("GraphRAG extract %d/%d chunks done", done, len(keys))

    entities: Dict[str, EntityRec] = {}
    relationships: Dict[Tuple[str, str], RelRec] = {}

    def _ensure_entity(name: str, etype: str = "other") -> EntityRec:
        k = _ekey(name)
        rec = entities.get(k)
        if rec is None:
            rec = EntityRec(key=k, name=name.strip(), type=etype or "other")
            entities[k] = rec
        elif not rec.type or rec.type == "other":
            rec.type = etype or rec.type
        return rec

    for cid, parsed in per_chunk.items():
        for e in parsed["entities"]:
            rec = _ensure_entity(e["name"], e.get("type", "other"))
            desc = (e.get("description") or "").strip()
            if desc:
                rec.descriptions.append(desc)
            rec.chunk_ids.add(cid)
        for r in parsed["relationships"]:
            src = _ensure_entity(r["source"])          # stub endpoints (Microsoft behavior)
            dst = _ensure_entity(r["target"])
            if src.key == dst.key:
                continue
            pair = tuple(sorted((src.key, dst.key)))
            rel = relationships.get(pair)
            if rel is None:
                rel = RelRec(src_key=pair[0], dst_key=pair[1])
                relationships[pair] = rel
            desc = (r.get("description") or "").strip()
            if desc:
                rel.descriptions.append(desc)
            try:
                rel.weight += float(r.get("strength", 5) or 5)
            except (TypeError, ValueError):
                rel.weight += 5.0
            rel.chunk_ids.add(cid)

    logger.info("GraphRAG merged: %d entities, %d relationships (%d parse failures)",
                len(entities), len(relationships), n_failures)
    return entities, list(relationships.values()), n_failures


def summarize_entities(llm, entities: Dict[str, EntityRec], cfg) -> None:
    """LLM-merge descriptions only when their joined length exceeds the threshold."""
    n = 0
    for rec in entities.values():
        joined = " ".join(dict.fromkeys(rec.descriptions))
        if len(rec.descriptions) > 1 and len(joined) > cfg.gr_summarize_desc_threshold:
            raw, _, _ = llm.infer(
                prompts.summarize_descriptions_messages(rec.name, rec.descriptions),
                json_mode=True,
                max_completion_tokens=cfg.summarize_max_tokens,
            )
            try:
                rec.summary = (_loads_lenient(raw).get("description") or "").strip()
            except Exception:  # noqa: BLE001
                rec.summary = joined[: cfg.gr_summarize_desc_threshold]
            n += 1
    logger.info("GraphRAG summarized %d oversized entity descriptions", n)
