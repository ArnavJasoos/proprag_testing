"""Benchmark configuration wrapping the reused ``POCConfig``.

One ``BenchmarkConfig`` object drives the whole run. It carries benchmark-only
knobs (subset size, GraphRAG parameters, per-call token caps) and knows how to
build the ``POCConfig`` the reused ``proprag_poc`` library expects, pinned to the
local Koboldcpp backend and the shared sentence-transformers embedder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

from . import _bootstrap  # noqa: F401 - side effect: proprag_poc on sys.path
from proprag_poc.config import POCConfig

# Default dataset location (sibling PropRAG-main checkout).
_DEFAULT_DATASET = os.path.join(
    _bootstrap._PROPRAG_PARENT,
    "PropRAG-main",
    "reproduce",
    "dataset",
    "2wikimultihopqa.json",
)
_DEFAULT_CORPUS = os.path.join(
    _bootstrap._PROPRAG_PARENT,
    "PropRAG-main",
    "reproduce",
    "dataset",
    "2wikimultihopqa_corpus.json",
)


@dataclass
class BenchmarkConfig:
    project_dir: str
    dataset_path: str = _DEFAULT_DATASET
    corpus_path: str = _DEFAULT_CORPUS

    # ----- Subset selection -----
    n_questions: int = 50
    seed: int = 42
    pilot: Optional[int] = None

    # ----- Shared retrieval / QA -----
    recall_ks: Tuple[int, ...] = (2, 5, 10)
    retrieval_top_k: int = 10
    qa_top_k: int = 5                 # passages fed to the generator (all systems)
    qa_max_answer_tokens: int = 512

    # ----- Per-call completion caps (runtime control on a CPU-served 20B) -----
    extract_max_tokens: int = 1500
    summarize_max_tokens: int = 512
    report_max_tokens: int = 1000

    # ----- GraphRAG knobs -----
    gr_entity_types: Tuple[str, ...] = (
        "person", "organization", "location", "event", "work", "date", "other",
    )
    gr_max_gleanings: int = 0         # Microsoft default is 1; 0 halves index cost (flagged in report)
    gr_summarize_desc_threshold: int = 800   # merged-description chars before LLM summarization
    gr_min_community_size: int = 2
    gr_report_max_input_chars: int = 12000   # ~3000-token cap on community-report context
    gr_local_top_entities: int = 10
    gr_entity_sim_floor: float = 0.5
    gr_dense_blend: float = 0.5

    systems: Tuple[str, ...] = ("BaseRAG", "GraphRAG", "PropRAG")

    def __post_init__(self):
        self.project_dir = os.path.abspath(self.project_dir)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.project_dir, "data")

    def make_poc_config(self) -> POCConfig:
        """Build the library config, pinned to the local Koboldcpp + bge stack.

        ``PROPRAG_*`` env vars still override backend/model/embedder (e.g. drop to
        ``BAAI/bge-base-en-v1.5`` under RAM pressure) via ``POCConfig`` itself.
        """
        return POCConfig(
            data_dir=self.data_dir,
            llm_backend="koboldcpp",   # -> http://localhost:5001/v1, gpt-oss-20b-Q4_K_M.gguf
            temperature=0.0,
            seed=self.seed,
            retrieval_top_k=self.retrieval_top_k,
            qa_top_k=self.qa_top_k,
            llm_max_workers=1,         # Koboldcpp serves one request at a time
            strip_reasoning=True,      # honoured by BenchLLMClient (POC leaves it unimplemented)
        )
