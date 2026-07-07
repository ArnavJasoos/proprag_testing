"""2WikiMultiHopQA loading, stratified subsetting, and corpus assembly.

Pure Python (no LLM, no embeddings) so it can be sanity-checked offline via
``python -m benchmark.dataset``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Canonical type order: fixes largest-remainder tie-breaks and output sorting.
TYPE_ORDER = ("compositional", "comparison", "bridge_comparison", "inference")


@dataclass
class BenchQuestion:
    qid: str
    qtype: str
    question: str
    answer: str
    gold_titles: List[str]
    context_titles: List[str]

    @property
    def gold_answers(self) -> List[str]:
        """EM/F1 gold list. 2wiki gives one canonical answer per question."""
        return [self.answer]


# --------------------------------------------------------------------- loading
def load_questions(path: str) -> List[BenchQuestion]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[BenchQuestion] = []
    for q in raw:
        gold = list(dict.fromkeys(sf[0] for sf in q.get("supporting_facts", [])))
        ctx = list(dict.fromkeys(c[0] for c in q.get("context", [])))
        out.append(
            BenchQuestion(
                qid=q["_id"],
                qtype=q["type"],
                question=q["question"],
                answer=q["answer"],
                gold_titles=gold,
                context_titles=ctx,
            )
        )
    return out


# ------------------------------------------------------------- allocation math
def _largest_remainder(counts: Dict[str, int], n: int) -> Dict[str, int]:
    """Proportionally allocate ``n`` slots across types; tie-break by TYPE_ORDER.

    Clamps each allocation to what the type actually has and redistributes any
    shortfall to types with spare capacity (largest remainder first).
    """
    types = [t for t in TYPE_ORDER if t in counts] + [
        t for t in counts if t not in TYPE_ORDER
    ]
    total = sum(counts.values())
    if total == 0 or n <= 0:
        return {t: 0 for t in types}

    raw = {t: counts[t] * n / total for t in types}
    alloc = {t: int(raw[t]) for t in types}
    order_index = {t: i for i, t in enumerate(types)}

    def frac_key(t: str) -> Tuple[float, int]:
        return (-(raw[t] - int(raw[t])), order_index[t])

    remainder = n - sum(alloc.values())
    for t in sorted(types, key=frac_key)[:remainder]:
        alloc[t] += 1

    # Clamp to availability, then redistribute any shortfall.
    for t in types:
        alloc[t] = min(alloc[t], counts[t])
    shortfall = n - sum(alloc.values())
    while shortfall > 0:
        spare = [t for t in types if alloc[t] < counts[t]]
        if not spare:
            break
        for t in sorted(spare, key=frac_key):
            if shortfall == 0:
                break
            alloc[t] += 1
            shortfall -= 1
    return alloc


def _stratified_pick(
    qs: List[BenchQuestion], n: int, seed: int
) -> List[BenchQuestion]:
    by_type: Dict[str, List[BenchQuestion]] = {}
    for q in qs:
        by_type.setdefault(q.qtype, []).append(q)
    counts = {t: len(v) for t, v in by_type.items()}
    alloc = _largest_remainder(counts, n)

    picked: List[BenchQuestion] = []
    rng = random.Random(seed)
    for t in sorted(by_type, key=lambda x: TYPE_ORDER.index(x) if x in TYPE_ORDER else 99):
        pool = sorted(by_type[t], key=lambda q: q.qid)  # deterministic pool order
        picked.extend(rng.sample(pool, alloc[t]))
    picked.sort(key=lambda q: (TYPE_ORDER.index(q.qtype) if q.qtype in TYPE_ORDER else 99, q.qid))
    return picked


def stratified_subset(qs: List[BenchQuestion], n: int, seed: int) -> List[BenchQuestion]:
    return _stratified_pick(qs, n, seed)


def pilot_subset(qs_subset: List[BenchQuestion], k: int, seed: int) -> List[BenchQuestion]:
    """Stratified ``k``-of-subset. Its questions are a subset of ``qs_subset`` so
    every pilot extraction becomes a cache hit during the full run."""
    return _stratified_pick(qs_subset, k, seed)


# ----------------------------------------------------------------- corpus build
def chunk_text(title: str, text: str) -> str:
    """One searchable chunk per document: the title on its own line, then body."""
    return f"{title}\n{text}"


def build_corpus(qs: List[BenchQuestion], corpus_path: str) -> Dict[str, str]:
    """title -> text for the union of the subset's context titles.

    Asserts every gold and context title exists in the corpus so Recall@k and QA
    context are never silently missing a gold passage.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    full: Dict[str, str] = {}
    for doc in raw:
        title = doc["title"]
        if title not in full:  # dedupe by title, first wins
            full[title] = doc["text"]

    needed = set()
    for q in qs:
        needed.update(q.context_titles)
        needed.update(q.gold_titles)

    missing = sorted(t for t in needed if t not in full)
    if missing:
        raise ValueError(
            f"{len(missing)} needed titles absent from corpus, e.g. {missing[:3]}"
        )
    return {t: full[t] for t in sorted(needed)}


# -------------------------------------------------------------------- identity
def subset_hash(qs: List[BenchQuestion]) -> str:
    joined = "|".join(sorted(q.qid for q in qs))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def corpus_id(qs: List[BenchQuestion], tag: str) -> str:
    return f"2wiki_{tag}_{subset_hash(qs)[:10]}"


def type_counts(qs: List[BenchQuestion]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for q in qs:
        counts[q.qtype] = counts.get(q.qtype, 0) + 1
    return counts


def write_manifest(
    run_dir: str, qs: List[BenchQuestion], titles: Dict[str, str], seed: int, cfg
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "manifest.json")
    manifest = {
        "seed": seed,
        "n_questions": len(qs),
        "type_counts": type_counts(qs),
        "subset_hash": subset_hash(qs),
        "corpus_size": len(titles),
        "qids": [q.qid for q in qs],
        "dataset_path": cfg.dataset_path,
        "corpus_path": cfg.corpus_path,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


# ---------------------------------------------------------------------- sanity
def _main() -> None:
    from .bench_config import BenchmarkConfig

    cfg = BenchmarkConfig(project_dir=os.path.dirname(os.path.dirname(__file__)))
    qs = load_questions(cfg.dataset_path)
    print(f"loaded {len(qs)} questions; full type counts: {type_counts(qs)}")

    subset = stratified_subset(qs, cfg.n_questions, cfg.seed)
    print(f"subset n={len(subset)} type counts: {type_counts(subset)}")

    pilot = pilot_subset(subset, 10, cfg.seed)
    print(f"pilot n={len(pilot)} type counts: {type_counts(pilot)}")
    assert set(p.qid for p in pilot).issubset(set(s.qid for s in subset)), "pilot not a subset"

    titles = build_corpus(subset, cfg.corpus_path)
    print(f"corpus subset size: {len(titles)} docs (expect ~380-420)")

    # Gold-title coverage.
    all_gold = set()
    for q in subset:
        all_gold.update(q.gold_titles)
    covered = sum(1 for t in all_gold if t in titles)
    print(f"gold titles: {len(all_gold)}, covered by corpus: {covered}")
    assert covered == len(all_gold), "some gold titles are missing from the corpus"
    print(f"corpus_id: {corpus_id(subset, 'n50')}")
    print("dataset sanity OK")


if __name__ == "__main__":
    _main()
