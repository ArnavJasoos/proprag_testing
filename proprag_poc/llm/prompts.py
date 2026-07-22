"""Prompt templates: NER, proposition extraction, JSON repair, RAG QA, query
contextualization. NER + proposition prompts are ported from the reference
PropRAG templates; the QA and contextualization prompts are POC additions.
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------------- NER
_NER_SYSTEM = """You are an expert at named-entity recognition. Extract every named
entity (people, organizations, locations, dates, products, events, works,
quantities and other proper noun phrases) from the passage. Be comprehensive.
Respond ONLY with a JSON object of the form {"entities": ["...", "..."]}."""

_NER_USER = """Passage:
```
{passage}
```
Extract the named entities as JSON."""


def ner_messages(passage: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _NER_SYSTEM},
        {"role": "user", "content": _NER_USER.format(passage=passage)},
    ]


# ----------------------------------------------------------------- propositions
_PROPOSITION_SYSTEM = """Your task is to break a passage into precise, atomic propositions
using ONLY a provided list of named entities. A proposition is a fully
contextualized statement expressing a single unit of meaning.

For each proposition:
1. Extract a complete, standalone statement that preserves full context.
2. Use ONLY entities from the named_entities list - do not introduce new entities.
3. Ensure each proposition contains only ONE claim or relationship.
4. Be specific about which entities are involved in each relationship.
5. Preserve temporal and causal relationships.

Respond ONLY with a JSON object:
{"propositions": [{"text": "...", "entities": ["...", "..."]}, ...]}
where each "entities" array contains only entities from the named_entities list."""

_PROPOSITION_USER = """Passage:
```
{passage}
```

Named entities: {named_entities}

Break the passage into atomic propositions as JSON."""


def proposition_messages(passage: str, named_entities_json: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _PROPOSITION_SYSTEM},
        {
            "role": "user",
            "content": _PROPOSITION_USER.format(
                passage=passage, named_entities=named_entities_json
            ),
        },
    ]


# ------------------------------------------------------------------- JSON repair
_FIX_JSON_SYSTEM = """You repair malformed JSON. Return ONLY valid JSON that preserves
the intended structure and content. Do not add commentary."""


def fix_json_messages(broken: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _FIX_JSON_SYSTEM},
        {"role": "user", "content": f"Fix this JSON:\n```\n{broken}\n```"},
    ]


# ------------------------------------------------------------------------ RAG QA
_QA_SYSTEM = """You answer questions using ONLY the provided context passages and the
prior conversation. Think briefly, then give a final answer. If the answer is not
in the context, say you don't know. End your reply with a line starting with
"Answer:" followed by the concise final answer."""


def qa_messages(
    question: str,
    passages: List[str],
    history: List[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": _QA_SYSTEM}]
    if history:
        for turn in history:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": turn["content"]})
    context = "\n\n".join(f"[Passage {i + 1}]\n{p}" for i, p in enumerate(passages))
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nThought:",
        }
    )
    return messages


# -------------------------------------------------------- query contextualization
_CONTEXT_SYSTEM = """Given a conversation history and a follow-up question, rewrite the
follow-up into a standalone, self-contained search query that resolves all
pronouns and references using the history. Respond ONLY with a JSON object:
{"query": "<standalone query>"}. Do not answer the question."""


def contextualize_messages(
    question: str, history: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    hist = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    return [
        {"role": "system", "content": _CONTEXT_SYSTEM},
        {
            "role": "user",
            "content": f"Conversation:\n{hist}\n\nFollow-up question: {question}\n\n"
            "Rewrite as a standalone query (JSON).",
        },
    ]
