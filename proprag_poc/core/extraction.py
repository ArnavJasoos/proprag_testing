"""NER + proposition extraction (ported from reference EnhancedOpenIE +
PropositionExtractor). Parallel LLM calls, constrained propositions, and a
multi-tier JSON-repair fallback chain.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from ..llm.client import LLMClient
from ..llm import prompts

logger = logging.getLogger(__name__)

# Matches {"text": "...", "entities": [ ... ]} proposition objects.
_PROP_OBJ = re.compile(
    r'\{\s*"text"\s*:\s*"(?P<text>(?:\\.|[^"\\])*)"\s*,\s*"entities"\s*:\s*'
    r'\[(?P<ents>[^\]]*)\]\s*\}',
    re.DOTALL,
)
_STR = re.compile(r'"((?:\\.|[^"\\])*)"')


def _loads_lenient(raw: str):
    """Best-effort parse; strips code fences before json.loads."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    return json.loads(s)


def _parse_entities(raw: str) -> List[str]:
    try:
        obj = _loads_lenient(raw)
        ents = obj.get("entities", []) if isinstance(obj, dict) else []
        return [str(e) for e in ents if str(e).strip()]
    except Exception:
        # Fallback: pull the entities array via regex.
        m = re.search(r'"entities"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        if not m:
            return []
        return [s.strip() for s in _STR.findall(m.group(1)) if s.strip()]


def _parse_propositions(raw: str) -> List[Dict]:
    """Parse proposition objects, tolerating malformed surrounding JSON."""
    try:
        obj = _loads_lenient(raw)
        props = obj.get("propositions", []) if isinstance(obj, dict) else []
        out = [
            {"text": p["text"], "entities": list(p.get("entities", []))}
            for p in props
            if isinstance(p, dict) and p.get("text")
        ]
        if out:
            return out
    except Exception:
        pass
    # Regex fallback: extract each {"text","entities"} object independently.
    out = []
    for m in _PROP_OBJ.finditer(raw):
        ents = [s for s in _STR.findall(m.group("ents"))]
        out.append({"text": m.group("text"), "entities": ents})
    return out


class Extractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ------------------------------------------------------------------- NER
    def ner(self, passage: str) -> List[str]:
        raw, _, _ = self.llm.infer(prompts.ner_messages(passage), json_mode=True)
        entities = _parse_entities(raw)
        if not entities:
            fixed, _, _ = self.llm.infer(prompts.fix_json_messages(raw), json_mode=True)
            entities = _parse_entities(fixed)
        # Dedupe, keep order.
        return list(dict.fromkeys(entities))

    # --------------------------------------------------------- propositions
    def propositions(self, passage: str, named_entities: List[str]) -> List[Dict]:
        msgs = prompts.proposition_messages(passage, json.dumps(named_entities))
        raw, _, _ = self.llm.infer(msgs, json_mode=True)
        props = _parse_propositions(raw)
        if not props:
            fixed, _, _ = self.llm.infer(prompts.fix_json_messages(raw), json_mode=True)
            props = _parse_propositions(fixed)
        return self._constrain(props, named_entities)

    def _constrain(self, props: List[Dict], allowed: List[str]) -> List[Dict]:
        """Drop entities not in the NER set (hard constraint from the reference)."""
        allowed_set = set(allowed)
        cleaned = []
        for p in props:
            ents = [e for e in p["entities"] if e in allowed_set]
            text = (p.get("text") or "").strip()
            if text:
                cleaned.append({"text": text, "entities": ents})
        return cleaned

    # --------------------------------------------------------------- batch
    def batch_extract(self, chunk_texts: Dict[str, str]) -> Dict[str, List[Dict]]:
        """chunk_id -> [{text, entities}]. Parallel NER then parallel propositions."""
        keys = list(chunk_texts.keys())
        n = len(keys)
        max_workers = self.llm.config.llm_max_workers
        logger.info("extraction: %d chunks (NER then propositions), %d workers", n, max_workers)

        ner_results: Dict[str, List[str]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.ner, chunk_texts[k]): k for k in keys}
            for done, fut in enumerate(as_completed(futs), 1):
                k = futs[fut]
                ner_results[k] = fut.result()
                logger.info("NER %d/%d chunks done", done, n)

        prop_results: Dict[str, List[Dict]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.propositions, chunk_texts[k], ner_results[k]): k for k in keys}
            for done, fut in enumerate(as_completed(futs), 1):
                k = futs[fut]
                prop_results[k] = fut.result()
                logger.info("propositions %d/%d chunks done", done, n)
        logger.info("extraction complete: %d chunks", n)
        return prop_results
