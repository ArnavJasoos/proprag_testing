"""Shared QA prompt + answer extraction for all three systems.

One identical prompt keeps the generation step fair; only the retrieved context
differs per system. Tuned for 2wiki short spans: brief reasoning, then a final
``Answer:`` line carrying the shortest span, which we parse back out for EM/F1.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

_QA_SYSTEM = (
    "You answer questions using ONLY the provided context passages. Reason "
    "briefly, then finish with a line formatted exactly as 'Answer: <answer>' "
    "where <answer> is the shortest possible span (a name, date, or short "
    "phrase). If the answer is not in the context, write 'Answer: unknown'."
)


def qa_messages(question: str, context_texts: List[str]) -> List[Dict[str, str]]:
    context = "\n\n".join(f"[Passage {i + 1}]\n{p}" for i, p in enumerate(context_texts))
    return [
        {"role": "system", "content": _QA_SYSTEM},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nReasoning:",
        },
    ]


def extract_answer(raw: str) -> str:
    """Pull the span after the last ``Answer:`` marker; first line only."""
    if not raw:
        return "unknown"
    tail = raw.rsplit("Answer:", 1)[1] if "Answer:" in raw else raw
    first_line = next((ln for ln in tail.strip().splitlines() if ln.strip()), "")
    return first_line.strip().strip('"').strip("'").strip(". ").strip() or "unknown"


def answer_question(llm, question: str, context_texts: List[str], cfg) -> Tuple[str, str]:
    """Return ``(short_answer, raw_response)``."""
    raw, _, _ = llm.infer(
        qa_messages(question, context_texts),
        json_mode=False,
        max_completion_tokens=cfg.qa_max_answer_tokens,
    )
    return extract_answer(raw), raw
