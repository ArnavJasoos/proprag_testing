"""GraphRAG prompts: entity/relationship extraction, description summarization,
and community reports. All emit JSON (``json_mode=True``), kept short with one
worked example so a CPU-served 20B model stays parseable.
"""

from __future__ import annotations

from typing import Dict, List

# ------------------------------------------------------------------ extraction
_EXTRACT_SYSTEM = """You are a knowledge-graph extractor. From the text, extract
named entities and the relationships between them.

Return ONLY a JSON object with this exact shape:
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
 "relationships": [{{"source": "...", "target": "...", "description": "...", "strength": 1-10}}]}}

Rules:
- "type" must be one of: {entity_types}.
- "name" is the entity as it appears (proper noun / phrase).
- "description" is a short factual phrase grounded in the text.
- "source" and "target" of every relationship must be entity names you also list.
- "strength" is an integer 1-10 rating how strongly the text ties the two entities.
Extract nothing that is not supported by the text."""

_EXTRACT_EXAMPLE = """Example.
Text: "Marie Curie, a physicist, was born in Warsaw and won the Nobel Prize."
JSON:
{"entities": [
  {"name": "Marie Curie", "type": "person", "description": "physicist born in Warsaw"},
  {"name": "Warsaw", "type": "location", "description": "birthplace of Marie Curie"},
  {"name": "Nobel Prize", "type": "work", "description": "award won by Marie Curie"}],
 "relationships": [
  {"source": "Marie Curie", "target": "Warsaw", "description": "was born in", "strength": 8},
  {"source": "Marie Curie", "target": "Nobel Prize", "description": "won the", "strength": 9}]}"""

_EXTRACT_USER = """{example}

Now extract from this text.
Text:
```
{text}
```
JSON:"""


def extraction_messages(text: str, entity_types: List[str]) -> List[Dict[str, str]]:
    system = _EXTRACT_SYSTEM.format(entity_types=", ".join(entity_types))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _EXTRACT_USER.format(example=_EXTRACT_EXAMPLE, text=text)},
    ]


_GLEAN_USER = """Some entities and relationships were missed in the previous
extraction of the same text. Add ONLY new ones, using the exact same JSON shape
{{"entities": [...], "relationships": [...]}}. Do not repeat anything already
listed below.

Text:
```
{text}
```

Already extracted:
{prev_json}

Additional JSON:"""


def gleaning_messages(text: str, prev_json: str, entity_types: List[str]) -> List[Dict[str, str]]:
    system = _EXTRACT_SYSTEM.format(entity_types=", ".join(entity_types))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _GLEAN_USER.format(text=text, prev_json=prev_json)},
    ]


# ------------------------------------------------------- description summaries
_SUMMARIZE_SYSTEM = """You merge several descriptions of the same entity into one
concise, factual description. Return ONLY JSON: {"description": "..."}."""

_SUMMARIZE_USER = """Entity: {name}
Descriptions:
{descriptions}

Merged description as JSON:"""


def summarize_descriptions_messages(name: str, descriptions: List[str]) -> List[Dict[str, str]]:
    body = "\n".join(f"- {d}" for d in descriptions)
    return [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": _SUMMARIZE_USER.format(name=name, descriptions=body)},
    ]


# ----------------------------------------------------------- community reports
_REPORT_SYSTEM = """You write a concise analytical report about a community of
related entities. Return ONLY JSON:
{"title": "...", "summary": "...", "rating": 0-10,
 "findings": [{"summary": "...", "explanation": "..."}]}
"rating" is the community's overall importance. Keep the summary under 120 words
and base everything strictly on the provided entities and relationships."""

_REPORT_USER = """Entities:
{entities_block}

Relationships:
{rels_block}

Write the community report as JSON:"""


def community_report_messages(entities_block: str, rels_block: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _REPORT_SYSTEM},
        {"role": "user", "content": _REPORT_USER.format(
            entities_block=entities_block, rels_block=rels_block)},
    ]
