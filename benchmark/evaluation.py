"""Answer and retrieval metrics.

``normalize_answer`` / EM / F1 are ported verbatim from PropRAG-main
(``utils/eval_utils.normalize_answer`` and ``evaluation/qa_eval``), stripped of
the ``BaseMetric`` scaffolding. Recall@k is the standard supporting-title recall.
Run ``python -m benchmark.evaluation`` to exercise the inline hand cases.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence


def normalize_answer(answer: str) -> str:
    """Lowercase, drop punctuation, drop a/an/the, collapse whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def em_score(golds: Sequence[str], pred: str) -> float:
    """Exact match, max over the gold list (MRQA-style)."""
    npred = normalize_answer(pred)
    return max((1.0 if normalize_answer(g) == npred else 0.0) for g in golds) if golds else 0.0


def _f1(gold: str, pred: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(golds: Sequence[str], pred: str) -> float:
    """Token-overlap F1, max over the gold list."""
    return max((_f1(g, pred) for g in golds), default=0.0)


def recall_at_k(
    gold_titles: Sequence[str],
    retrieved_titles: Sequence[str],
    ks: Sequence[int] = (2, 5, 10),
) -> Dict[str, float]:
    """For each k: |gold ∩ top-k retrieved| / |gold|."""
    gold = set(gold_titles)
    out: Dict[str, float] = {}
    for k in ks:
        if not gold:
            out[f"recall@{k}"] = 0.0
            continue
        topk = set(retrieved_titles[:k])
        out[f"recall@{k}"] = len(gold & topk) / len(gold)
    return out


def _main() -> None:
    assert normalize_answer("The  A.B.C.!") == "abc"
    assert em_score(["20 March 851"], "20 march 851.") == 1.0
    assert em_score(["Paris"], "London") == 0.0
    assert abs(f1_score(["barack obama"], "obama") - (2 * 1.0 * 0.5 / 1.5)) < 1e-9
    assert f1_score(["cat dog"], "cat dog") == 1.0
    r = recall_at_k(["A", "B"], ["A", "X", "B", "Y"], ks=(1, 2, 3))
    assert r == {"recall@1": 0.5, "recall@2": 0.5, "recall@3": 1.0}, r
    assert recall_at_k([], ["A"], ks=(2,)) == {"recall@2": 0.0}
    print("evaluation sanity OK")


if __name__ == "__main__":
    _main()
