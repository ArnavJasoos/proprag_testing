"""Aggregate results.jsonl + index_usage.json into metrics.json + report.md.

Charts are optional: a guarded matplotlib import degrades to "no charts" if the
package is missing.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)

_SYSTEMS = ("BaseRAG", "GraphRAG", "PropRAG")
_RECALL_KS = (2, 5, 10)


# ------------------------------------------------------------------- loading
def _load_rows(run_dir: str) -> List[Dict]:
    path = os.path.join(run_dir, "results.jsonl")
    latest: Dict = {}
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error") or "qid" not in row or "system" not in row:
                continue
            latest[(row["qid"], row["system"])] = row  # last wins (dedupe)
    return list(latest.values())


def _load_json(path: str) -> Dict:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------- aggregation
def _aggregate(rows: List[Dict]) -> Dict:
    by_system: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)

    metrics: Dict[str, Dict] = {}
    for system, srows in by_system.items():
        em = _mean([r.get("em", 0.0) for r in srows])
        f1 = _mean([r.get("f1", 0.0) for r in srows])
        recalls = {
            f"recall@{k}": _mean([r.get("recall", {}).get(f"recall@{k}", 0.0) for r in srows])
            for k in _RECALL_KS
        }
        usages = [r.get("usage", {}) for r in srows]
        metrics[system] = {
            "n": len(srows),
            "em": em,
            "f1": f1,
            **recalls,
            "mean_retrieval_latency_s": _mean([r.get("retrieval_latency_s", 0.0) for r in srows]),
            "mean_qa_latency_s": _mean([r.get("qa_latency_s", 0.0) for r in srows]),
            "mean_query_prompt_tokens": _mean([u.get("chat_prompt_tokens", 0) for u in usages]),
            "mean_query_completion_tokens": _mean([u.get("chat_completion_tokens", 0) for u in usages]),
            "mean_chat_calls_per_q": _mean([u.get("chat_calls", 0) for u in usages]),
            "by_type": _by_type(srows),
        }
    return metrics


def _by_type(srows: List[Dict]) -> Dict[str, Dict]:
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for r in srows:
        by_type[r.get("qtype", "?")].append(r)
    return {
        t: {
            "n": len(rs),
            "em": _mean([r.get("em", 0.0) for r in rs]),
            "f1": _mean([r.get("f1", 0.0) for r in rs]),
            "recall@5": _mean([r.get("recall", {}).get("recall@5", 0.0) for r in rs]),
        }
        for t, rs in sorted(by_type.items())
    }


# ------------------------------------------------------------------- markdown
def _fmt(x: float, pct: bool = False) -> str:
    return f"{x * 100:.1f}%" if pct else f"{x:.1f}"


def _render_md(metrics: Dict, index_usage: Dict, manifest: Dict, meta: Dict) -> str:
    systems = [s for s in _SYSTEMS if s in metrics] or list(metrics)
    lines: List[str] = ["# PropRAG vs GraphRAG vs BaseRAG — 2WikiMultiHopQA", ""]
    lines.append(f"Questions: {manifest.get('n_questions', '?')} "
                 f"(seed {manifest.get('seed', '?')}, subset {manifest.get('subset_hash', '?')[:10]}); "
                 f"corpus {manifest.get('corpus_size', '?')} docs.")
    lines.append("")

    # Answer + retrieval quality.
    lines.append("## Answer & retrieval quality")
    lines.append("")
    header = "| System | EM | F1 | R@2 | R@5 | R@10 |"
    lines += [header, "|---|---|---|---|---|---|"]
    for s in systems:
        m = metrics[s]
        lines.append(
            f"| {s} | {_fmt(m['em'], True)} | {_fmt(m['f1'], True)} | "
            f"{_fmt(m['recall@2'], True)} | {_fmt(m['recall@5'], True)} | {_fmt(m['recall@10'], True)} |"
        )
    lines.append("")

    # Per-question-type.
    lines.append("## By question type (EM / F1 / R@5)")
    lines.append("")
    lines += ["| Type | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    all_types = sorted({t for s in systems for t in metrics[s]["by_type"]})
    for t in all_types:
        cells = []
        for s in systems:
            bt = metrics[s]["by_type"].get(t)
            cells.append(
                f"{_fmt(bt['em'], True)} / {_fmt(bt['f1'], True)} / {_fmt(bt['recall@5'], True)}"
                if bt else "-"
            )
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append("")

    # Query cost.
    lines.append("## Query cost (per question)")
    lines.append("")
    lines += ["| System | chat calls | prompt tok | completion tok | retr latency (s) | QA latency (s) |",
              "|---|---|---|---|---|---|"]
    for s in systems:
        m = metrics[s]
        lines.append(
            f"| {s} | {m['mean_chat_calls_per_q']:.2f} | {m['mean_query_prompt_tokens']:.0f} | "
            f"{m['mean_query_completion_tokens']:.0f} | {m['mean_retrieval_latency_s']:.2f} | "
            f"{m['mean_qa_latency_s']:.2f} |"
        )
    lines.append("")

    # Index cost.
    lines.append("## Index cost (one-time)")
    lines.append("")
    lines += ["| System | chat calls | prompt tok | completion tok | embed texts | wall (s) | parse fails |",
              "|---|---|---|---|---|---|---|"]
    for s in systems:
        u = index_usage.get(s, {})
        lines.append(
            f"| {s} | {u.get('chat_calls', 0):.0f} | {u.get('chat_prompt_tokens', 0):.0f} | "
            f"{u.get('chat_completion_tokens', 0):.0f} | {u.get('embed_texts', 0):.0f} | "
            f"{u.get('wall_time_s', 0):.1f} | {u.get('parse_failures', 0)} |"
        )
    lines.append("")

    # Fairness footer.
    lines.append("## Fairness & configuration")
    lines.append("")
    lines.append(f"- Shared LLM: `{meta.get('llm_model', '?')}` @ `{meta.get('llm_backend', '?')}`; "
                 f"shared embedder: `{meta.get('embedding_model', '?')}`.")
    lines.append(f"- Shared chunk embeddings; qa_top_k={meta.get('qa_top_k', '?')}, "
                 f"retrieval_top_k={meta.get('retrieval_top_k', '?')}, seed={manifest.get('seed', '?')}.")
    lines.append(f"- GraphRAG gleanings={meta.get('gr_max_gleanings', '?')} "
                 "(Microsoft default is 1; 0 lowers index cost).")
    lines.append("- Recall@k uses retrieved documents; GraphRAG QA additionally uses community "
                 "reports + relationships via local-search context assembly.")
    total_wall = sum(index_usage.get(s, {}).get("wall_time_s", 0) for s in systems)
    lines.append(f"- Total index wall time: {total_wall:.1f}s.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- charts
def _charts(metrics: Dict, index_usage: Dict, run_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        logger.info("matplotlib unavailable; skipping charts")
        return

    systems = [s for s in _SYSTEMS if s in metrics]
    if not systems:
        return
    charts_dir = os.path.join(run_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Quality bars.
    fig, ax = plt.subplots(figsize=(7, 4))
    import numpy as np
    x = np.arange(len(systems))
    width = 0.25
    for i, key in enumerate(("em", "f1", "recall@5")):
        ax.bar(x + (i - 1) * width, [metrics[s][key] for s in systems], width, label=key.upper())
    ax.set_xticks(x); ax.set_xticklabels(systems); ax.set_ylabel("score"); ax.legend()
    ax.set_title("Answer quality & Recall@5")
    fig.tight_layout(); fig.savefig(os.path.join(charts_dir, "quality.png"), dpi=120); plt.close(fig)

    # Index tokens.
    fig, ax = plt.subplots(figsize=(7, 4))
    prompt = [index_usage.get(s, {}).get("chat_prompt_tokens", 0) for s in systems]
    compl = [index_usage.get(s, {}).get("chat_completion_tokens", 0) for s in systems]
    ax.bar(x, prompt, width * 2, label="prompt")
    ax.bar(x, compl, width * 2, bottom=prompt, label="completion")
    ax.set_xticks(x); ax.set_xticklabels(systems); ax.set_ylabel("tokens"); ax.legend()
    ax.set_title("Index chat tokens per system")
    fig.tight_layout(); fig.savefig(os.path.join(charts_dir, "index_tokens.png"), dpi=120); plt.close(fig)
    logger.info("charts written to %s", charts_dir)


# ---------------------------------------------------------------------- build
def build(run_dir: str, make_charts: bool = True) -> Dict:
    rows = _load_rows(run_dir)
    index_usage = _load_json(os.path.join(run_dir, "index_usage.json"))
    manifest = _load_json(os.path.join(run_dir, "manifest.json"))
    meta = _load_json(os.path.join(run_dir, "run_meta.json"))

    metrics = _aggregate(rows)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    md = _render_md(metrics, index_usage, manifest, meta)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    if make_charts:
        _charts(metrics, index_usage, run_dir)
    logger.info("report written to %s", os.path.join(run_dir, "report.md"))
    return metrics
