"""Follow-up query contextualization for conversational RAG.

PropRAG embeds the raw query for proposition/passage scoring, so a bare pronoun
("what about its sequel?") retrieves nothing. When a session has prior turns, an
LLM rewrites the follow-up into a standalone retrieval query before retrieval.
The rewritten query feeds beam search/PPR; the original question + history feed
the QA prompt.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from ..llm.client import LLMClient
from ..llm import prompts

logger = logging.getLogger(__name__)


class QueryContextualizer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def contextualize(self, question: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return question
        try:
            raw, _, _ = self.llm.infer(
                prompts.contextualize_messages(question, history), json_mode=True
            )
            return self._parse(raw) or question
        except Exception as e:  # noqa: BLE001 - never block retrieval on this
            logger.warning("contextualization failed, using raw question: %s", e)
            return question

    @staticmethod
    def _parse(raw: str) -> str:
        try:
            return json.loads(re.sub(r"```(?:json)?|```", "", raw).strip()).get("query", "")
        except Exception:
            m = re.search(r'"query"\s*:\s*"((?:\\.|[^"\\])*)"', raw)
            return m.group(1) if m else ""
