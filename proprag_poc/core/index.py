"""Indexing orchestration + persistence, and a thin Engine wiring retrieval,
QA, contextualization and sessions for the GUI.

A *corpus* is one indexed set of PDFs, stored under ``data/corpora/<corpus_id>``:
embedding stores (parquet+npy), graph (GraphML), and extraction/maps (JSON).
Re-opening a corpus loads artifacts instead of recomputing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import igraph as ig

from ..config import POCConfig
from ..embedding.encoder import EmbeddingModel
from ..ingestion.chunker import Chunker
from ..ingestion.pdf_loader import PDFLoader
from ..llm.client import LLMClient
from .baseline_retrievers import BaseRAGRetriever, GraphRAGRetriever
from .compare import ComparisonResult, run_comparison
from .conversation import QueryContextualizer
from .extraction import Extractor
from .graph_builder import GraphBuilder
from .ids import compute_mdhash_id
from .metrics import get_usage_tracker
from .qa import QAEngine, QAResult
from .retriever import Retriever
from .store import EmbeddingStore

logger = logging.getLogger(__name__)


@dataclass
class IndexedCorpus:
    corpus_id: str
    chunk_store: EmbeddingStore
    entity_store: EmbeddingStore
    proposition_store: EmbeddingStore
    graph: ig.Graph
    proposition_to_entities_map: Dict[str, List[str]]
    proposition_to_passages: Dict[str, List[str]]
    chunk_propositions: Dict[str, List[Dict]]


class Indexer:
    def __init__(self, config: POCConfig, embedding_model: EmbeddingModel, llm: LLMClient):
        self.config = config
        self.embedding_model = embedding_model
        self.extractor = Extractor(llm)
        self.graph_builder = GraphBuilder(config)
        self.loader = PDFLoader(config)
        self.chunker = Chunker(config)

    # ------------------------------------------------------------- paths
    def _corpus_dir(self, corpus_id: str) -> str:
        return os.path.join(self.config.data_dir, "corpora", corpus_id)

    def _maps_path(self, corpus_id: str) -> str:
        return os.path.join(self._corpus_dir(corpus_id), "maps.json")

    def _graph_path(self, corpus_id: str) -> str:
        return os.path.join(self._corpus_dir(corpus_id), "graph.graphml")

    def exists(self, corpus_id: str) -> bool:
        return os.path.isfile(self._graph_path(corpus_id)) and not self.config.force_index_from_scratch

    # ------------------------------------------------------------- build
    def build_from_pdfs(self, corpus_id: str, pdf_paths: List[str]) -> IndexedCorpus:
        cdir = self._corpus_dir(corpus_id)
        os.makedirs(cdir, exist_ok=True)

        chunk_texts = []
        for path in pdf_paths:
            doc = self.loader.load(path)
            for chunk in self.chunker.chunk_doc(doc):
                chunk_texts.append(chunk.text)
        return self.build_from_texts(corpus_id, chunk_texts)

    def build_from_texts(self, corpus_id: str, chunk_texts: List[str]) -> IndexedCorpus:
        cdir = self._corpus_dir(corpus_id)
        os.makedirs(cdir, exist_ok=True)

        chunk_store = EmbeddingStore(self.embedding_model, cdir, "chunk")
        entity_store = EmbeddingStore(self.embedding_model, cdir, "entity")
        proposition_store = EmbeddingStore(self.embedding_model, cdir, "proposition")

        logger.info("indexing '%s': embedding %d chunks", corpus_id, len(chunk_texts))
        chunk_store.insert_strings(chunk_texts)
        chunk_rows = chunk_store.get_text_for_all_rows()
        chunk_id_to_text = {cid: row["content"] for cid, row in chunk_rows.items()}

        logger.info("Extracting NER + propositions for %d chunks", len(chunk_id_to_text))
        chunk_propositions = self.extractor.batch_extract(chunk_id_to_text)

        # Collect entities, propositions, and maps.
        entities, prop_texts = set(), []
        prop_to_entities: Dict[str, List[str]] = {}
        prop_to_passages: Dict[str, List[str]] = {}
        for chunk_id, props in chunk_propositions.items():
            for prop in props:
                entities.update(prop["entities"])
                pkey = compute_mdhash_id(prop["text"], prefix="proposition-")
                prop_texts.append(prop["text"])
                prop_to_entities[pkey] = prop["entities"]
                prop_to_passages.setdefault(pkey, [])
                if chunk_id not in prop_to_passages[pkey]:
                    prop_to_passages[pkey].append(chunk_id)

        logger.info("embedding %d entities, %d propositions", len(entities), len(prop_texts))
        entity_store.insert_strings(sorted(entities))
        proposition_store.insert_strings(prop_texts)

        logger.info("building knowledge graph")
        chunk_ids = chunk_store.get_all_ids()
        graph = self.graph_builder.build(
            chunk_ids, chunk_propositions, entity_store, chunk_store
        )
        graph.write_graphml(self._graph_path(corpus_id))
        logger.info("index '%s' complete (%d nodes, %d edges)", corpus_id,
                    graph.vcount(), graph.ecount())
        with open(self._maps_path(corpus_id), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "proposition_to_entities_map": prop_to_entities,
                    "proposition_to_passages": prop_to_passages,
                    "chunk_propositions": chunk_propositions,
                },
                f,
            )

        return IndexedCorpus(
            corpus_id, chunk_store, entity_store, proposition_store, graph,
            prop_to_entities, prop_to_passages, chunk_propositions,
        )

    # ------------------------------------------------------------- load
    def load(self, corpus_id: str) -> IndexedCorpus:
        cdir = self._corpus_dir(corpus_id)
        chunk_store = EmbeddingStore(self.embedding_model, cdir, "chunk")
        entity_store = EmbeddingStore(self.embedding_model, cdir, "entity")
        proposition_store = EmbeddingStore(self.embedding_model, cdir, "proposition")
        graph = ig.Graph.Read_GraphML(self._graph_path(corpus_id))
        with open(self._maps_path(corpus_id), "r", encoding="utf-8") as f:
            maps = json.load(f)
        return IndexedCorpus(
            corpus_id, chunk_store, entity_store, proposition_store, graph,
            maps["proposition_to_entities_map"],
            maps["proposition_to_passages"],
            maps["chunk_propositions"],
        )

    def get_or_build(self, corpus_id: str, pdf_paths: List[str]) -> IndexedCorpus:
        if self.exists(corpus_id):
            logger.info("Loading existing corpus %s", corpus_id)
            return self.load(corpus_id)
        return self.build_from_pdfs(corpus_id, pdf_paths)


class Engine:
    """High-level wiring used by the GUI / smoke test."""

    def __init__(self, config: POCConfig):
        self.config = config
        self.embedding_model = EmbeddingModel(config)
        self.llm = LLMClient(config)
        self.indexer = Indexer(config, self.embedding_model, self.llm)
        self.contextualizer = QueryContextualizer(self.llm)
        self.qa_engine = QAEngine(self.llm, config)
        self.usage = get_usage_tracker()
        self._retrievers: Dict[str, Retriever] = {}
        self._compare_retrievers: Dict[str, "OrderedDict[str, object]"] = {}
        self._corpora: Dict[str, IndexedCorpus] = {}

    def corpus(self, corpus_id: str, pdf_paths: Optional[List[str]] = None) -> IndexedCorpus:
        if corpus_id not in self._corpora:
            self._corpora[corpus_id] = self.indexer.get_or_build(corpus_id, pdf_paths or [])
        return self._corpora[corpus_id]

    def retriever(self, corpus_id: str) -> Retriever:
        if corpus_id not in self._retrievers:
            self._retrievers[corpus_id] = Retriever(
                self.corpus(corpus_id), self.embedding_model, self.config
            )
        return self._retrievers[corpus_id]

    def comparison_retrievers(self, corpus_id: str):
        """Ordered {system_name -> retriever} for BaseRAG, GraphRAG, PropRAG."""
        from collections import OrderedDict

        if corpus_id not in self._compare_retrievers:
            corpus = self.corpus(corpus_id)
            self._compare_retrievers[corpus_id] = OrderedDict(
                [
                    ("BaseRAG", BaseRAGRetriever(corpus, self.embedding_model, self.config)),
                    ("GraphRAG", GraphRAGRetriever(
                        corpus, self.embedding_model, self.config, self.indexer.extractor)),
                    ("PropRAG", self.retriever(corpus_id)),
                ]
            )
        return self._compare_retrievers[corpus_id]

    def ablation_retrievers(self, corpus_id: str):
        """Ablation: all three labels map to one chunk-vector dense retriever.

        Strips the index differences that define GraphRAG/PropRAG so the systems
        converge — an algorithm-only view, NOT a fair framework comparison.
        """
        from collections import OrderedDict

        key = f"__ablation__{corpus_id}"
        if key not in self._compare_retrievers:
            base = BaseRAGRetriever(self.corpus(corpus_id), self.embedding_model, self.config)
            self._compare_retrievers[key] = OrderedDict(
                [("BaseRAG", base), ("GraphRAG", base), ("PropRAG", base)]
            )
        return self._compare_retrievers[key]

    def compare(self, corpus_id: str, question: str,
                history: Optional[List[Dict[str, str]]] = None,
                ablation: bool = False) -> ComparisonResult:
        return run_comparison(self, corpus_id, question, history, ablation=ablation)

    def ask(self, corpus_id: str, question: str,
            history: Optional[List[Dict[str, str]]] = None) -> QAResult:
        history = history or []
        search_query = self.contextualizer.contextualize(question, history)
        passages = self.retriever(corpus_id).retrieve(search_query)
        return self.qa_engine.answer(question, passages, history)
