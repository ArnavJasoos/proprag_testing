"""Benchmark configuration wrapping the reused POCConfig.

Desktop runs can still use Koboldcpp. Colab sets PROPRAG_LLM_BACKEND=llama_cpp
and PROPRAG_GGUF_MODEL_PATH so inference uses a downloaded GGUF model directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from . import _bootstrap  # noqa: F401 - side effect: proprag_poc on sys.path
from proprag_poc.config import POCConfig

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

    n_questions: int = 50
    seed: int = 42
    pilot: Optional[int] = None

    recall_ks: Tuple[int, ...] = (2, 5, 10)
    retrieval_top_k: int = 10
    qa_top_k: int = 5
    qa_max_answer_tokens: int = 512

    extract_max_tokens: int = 1500
    summarize_max_tokens: int = 512
    report_max_tokens: int = 1000

    gr_entity_types: Tuple[str, ...] = (
        "person", "organization", "location", "event", "work", "date", "other",
    )
    gr_max_gleanings: int = 0
    gr_summarize_desc_threshold: int = 800
    gr_min_community_size: int = 2
    gr_report_max_input_chars: int = 12000
    gr_local_top_entities: int = 10
    gr_entity_sim_floor: float = 0.5
    gr_dense_blend: float = 0.5

    systems: Tuple[str, ...] = ("BaseRAG", "GraphRAG", "PropRAG")

    def __post_init__(self):
        self.project_dir = os.path.abspath(self.project_dir)
        self.dataset_path = os.environ.get("PROPRAG_DATASET_PATH", self.dataset_path)
        self.corpus_path = os.environ.get("PROPRAG_CORPUS_PATH", self.corpus_path)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.project_dir, "data")

    def make_poc_config(self) -> POCConfig:
        backend = os.environ.get("PROPRAG_LLM_BACKEND", "koboldcpp")
        return POCConfig(
            data_dir=self.data_dir,
            llm_backend=backend,
            temperature=0.0,
            seed=self.seed,
            retrieval_top_k=self.retrieval_top_k,
            qa_top_k=self.qa_top_k,
            llm_max_workers=1,
            strip_reasoning=True,
        )
