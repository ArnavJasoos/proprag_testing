"""History-aware QA over retrieved passages (ported/adapted from reference ``qa``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..config import POCConfig
from ..llm.client import LLMClient
from ..llm import prompts
from .retriever import RetrievedPassage


@dataclass
class QAResult:
    answer: str
    raw: str
    passages: List[RetrievedPassage] = field(default_factory=list)


class QAEngine:
    def __init__(self, llm: LLMClient, config: POCConfig):
        self.llm = llm
        self.config = config

    def answer(
        self,
        question: str,
        passages: List[RetrievedPassage],
        history: List[Dict[str, str]] | None = None,
    ) -> QAResult:
        top = passages[: self.config.qa_top_k]
        msgs = prompts.qa_messages(
            question,
            [p.text for p in top],
            (history or [])[-self.config.history_max_turns :],
        )
        raw, _, _ = self.llm.infer(msgs, json_mode=False)
        answer = raw.split("Answer:", 1)[1].strip() if "Answer:" in raw else raw.strip()
        return QAResult(answer=answer, raw=raw, passages=top)
