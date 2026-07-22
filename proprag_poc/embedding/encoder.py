"""Pluggable embedding encoder with an on-disk vector cache, a shared request
rate limiter, and token-usage instrumentation.

Backends:
  * ``sentence_transformers`` (default) -> local model (e.g. BAAI/bge-large-en-v1.5),
    loaded once via HuggingFace ``transformers``. Runs on GPU if available, else
    CPU. Offline: no rate limit, no token cost, no API key.
  * ``nvidia`` -> the NVIDIA NeMo Retriever embedding API (OpenAI-compatible), the
    same online provider the chat LLM can use. nv-embedqa models require an
    ``input_type`` ("query" vs "passage") sent via ``extra_body``; handled here.
  * ``openai`` / ``ollama`` -> any OpenAI-compatible embedding server.

Only the online backends (``nvidia``/``openai``/``ollama`` pointed at a remote
server) acquire a slot from the shared ``RateLimiter``. All backends report token
usage + latency to the ``UsageTracker``. Online requests are batched (one HTTP
request = one rate-limit slot) regardless of how many texts they carry.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from typing import List, Optional

import numpy as np

from ..config import POCConfig
from ..core.metrics import CallRecord, get_usage_tracker
from ..llm.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, config: POCConfig):
        self.config = config
        self._backend = config.embedding_backend
        self._dim: Optional[int] = None
        self._model = None
        self._client = None
        self._tracker = get_usage_tracker()
        self._limiter = (
            get_rate_limiter(config.rpm_limit) if config.embedding_is_online else None
        )
        self._cache_path = os.path.join(config.data_dir, "embedding_cache.sqlite")
        self._lock = threading.Lock()
        self._init_cache()
        self._init_backend()

    # ----------------------------------------------------------------- setup
    def _init_backend(self):
        if self._backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model)
            self._dim = self._model.get_sentence_embedding_dimension()
        elif self._backend in ("nvidia", "ollama", "openai"):
            from openai import OpenAI

            base = self.config.embedding_base_url or (
                "http://localhost:11434/v1" if self._backend == "ollama" else None
            )
            self._client = OpenAI(
                base_url=base,
                api_key=self.config.embedding_api_key or "not-needed",
                timeout=self.config.request_timeout,
            )
        else:
            raise ValueError(f"Unknown embedding backend: {self._backend}")

    def _init_cache(self):
        with sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, dim INT, data BLOB)"
            )

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = self.batch_encode(["dimension probe"]).shape[1]
        return self._dim

    # ----------------------------------------------------------------- cache
    def _key(self, text: str, instruction: str, input_type: str) -> str:
        # input_type changes nv-embedqa output, so it is part of the cache key.
        raw = f"{self.config.embedding_model}|{input_type}|{instruction}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_get_many(self, keys):
        out = {}
        with self._lock, sqlite3.connect(self._cache_path) as conn:
            conn.row_factory = None
            qmarks = ",".join("?" * len(keys))
            for k, dim, blob in conn.execute(
                f"SELECT key, dim, data FROM vec WHERE key IN ({qmarks})", keys
            ):
                out[k] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        return out

    def _cache_put_many(self, items):
        with self._lock, sqlite3.connect(self._cache_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO vec (key, dim, data) VALUES (?, ?, ?)",
                [(k, v.shape[0], v.astype(np.float32).tobytes()) for k, v in items],
            )

    # ---------------------------------------------------------------- encode
    def batch_encode(
        self,
        texts,
        instruction: str = "",
        norm: Optional[bool] = None,
        use_cache: bool = True,
        input_type: str = "passage",
    ) -> np.ndarray:
        """Encode texts. ``input_type`` is "passage" for stored docs, "query" for
        search queries (only meaningful for nv-embedqa models)."""
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        norm = self.config.embedding_normalize if norm is None else norm

        keys = [self._key(t, instruction, input_type) for t in texts]
        cached = self._cache_get_many(keys) if use_cache else {}

        if use_cache and cached:
            self._tracker.record(
                CallRecord(
                    kind="embed",
                    model=self.config.embedding_model,
                    cache_hit=True,
                    batch_size=len(cached),
                )
            )

        todo_idx = [i for i, k in enumerate(keys) if k not in cached]
        if todo_idx:
            todo_texts = [instruction + texts[i] for i in todo_idx]
            fresh = self._encode_raw(todo_texts, input_type)
            new_items = []
            for j, i in enumerate(todo_idx):
                cached[keys[i]] = fresh[j]
                new_items.append((keys[i], fresh[j]))
            if use_cache:
                self._cache_put_many(new_items)

        vecs = np.stack([cached[k] for k in keys]).astype(np.float32)
        if norm:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-12, None)
        return vecs

    def _encode_raw(self, texts: List[str], input_type: str) -> np.ndarray:
        if self._backend == "sentence_transformers":
            vecs = np.asarray(
                self._model.encode(
                    texts,
                    batch_size=self.config.embedding_batch_size,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            # No API cost, but record the call so per-system counts stay consistent.
            self._tracker.record(
                CallRecord(
                    kind="embed", model=self.config.embedding_model, batch_size=len(texts)
                )
            )
            return vecs

        # OpenAI-compatible / NVIDIA: send in capped batches, one rate-limit slot each.
        out: List[np.ndarray] = []
        bs = max(1, self.config.embedding_batch_size)
        extra = {}
        if self.config.embedding_needs_input_type:
            extra = {"input_type": input_type, "truncate": "END"}
        for start in range(0, len(texts), bs):
            chunk = texts[start : start + bs]
            rate_wait = self._limiter.acquire() if self._limiter is not None else 0.0
            t0 = time.monotonic()
            resp = self._client.embeddings.create(
                model=self.config.embedding_model,
                input=chunk,
                **({"extra_body": extra} if extra else {}),
            )
            latency = time.monotonic() - t0
            usage = getattr(resp, "usage", None)
            tokens = int(getattr(usage, "total_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
            logger.info(
                "embed %d texts (%s) in %.1fs (%d tok)%s",
                len(chunk), input_type, latency, tokens,
                f", waited {rate_wait:.1f}s" if rate_wait > 0.05 else "",
            )
            self._tracker.record(
                CallRecord(
                    kind="embed",
                    model=self.config.embedding_model,
                    prompt_tokens=tokens,
                    latency_s=latency,
                    rate_wait_s=rate_wait,
                    batch_size=len(chunk),
                )
            )
            out.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
        return np.vstack(out)
