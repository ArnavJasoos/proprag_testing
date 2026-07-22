"""Pluggable embedding encoder with an on-disk vector cache and usage
instrumentation.

Colab build: the ``sentence_transformers`` backend loads a local HuggingFace
model (default ``nvidia/NV-Embed-v2``). NV-Embed-v2 is a 7.85B-parameter model
that does NOT fit a free-tier T4 (15GB) in fp16, so it is loaded in 8-bit via
``bitsandbytes`` (``PROPRAG_EMBED_8BIT=1``, the default). A CPU fp32 fallback is
used if the 8-bit load fails. ``unload()`` frees the model from VRAM so the GGUF
chat model can take the whole GPU in the next phase.

The cache + ``batch_encode`` logic is byte-for-byte identical to the desktop
version so cache keys (model|input_type|instruction|text) stay stable across
phases: Phase B fills the cache while embedding; nothing re-encodes later.
"""

from __future__ import annotations

import gc
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


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "")


class EmbeddingModel:
    def __init__(self, config: POCConfig):
        self.config = config
        self._backend = config.embedding_backend
        self._dim: Optional[int] = None
        self._model = None
        self._client = None
        self._append_eos = False  # NV-Embed-v2 (sentence-transformers path) needs a trailing EOS
        self._nv_native = False   # NV-Embed-v2 loaded via transformers (its own .encode)
        self._max_len = 512
        self._nv_batch = 4
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
            self._load_sentence_transformer()
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

    def _load_sentence_transformer(self):
        import torch

        name = self.config.embedding_model
        is_nv_embed = "nv-embed" in name.lower()
        want_8bit = _env_flag("PROPRAG_EMBED_8BIT", True) and torch.cuda.is_available()
        self._nv_native = False          # NV-Embed-v2 loaded via transformers (its own .encode)
        self._max_len = int(os.environ.get("PROPRAG_EMBED_MAX_LEN", "512"))
        self._nv_batch = int(os.environ.get("PROPRAG_EMBED_BATCH", "4"))

        # Preferred path for NV-Embed-v2 on a T4: load in 8-bit straight through
        # transformers (device_map handles placement; no SentenceTransformer .to()
        # which rejects 8-bit models) and use the model's native encode().
        if is_nv_embed and want_8bit:
            try:
                from transformers import AutoModel, BitsAndBytesConfig

                bnb = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Loading NV-Embed-v2 in 8-bit (bitsandbytes) via transformers")
                model = AutoModel.from_pretrained(
                    name,
                    trust_remote_code=True,
                    quantization_config=bnb,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                model.eval()
                self._model = model
                self._nv_native = True
                self._dim = int(getattr(model.config, "hidden_size", 4096))
                return
            except Exception as e:  # noqa: BLE001 - fall through to sentence-transformers/CPU
                logger.warning(
                    "8-bit NV-Embed load failed (%s); falling back to CPU sentence-transformers.", e
                )

        from sentence_transformers import SentenceTransformer

        # If 8-bit was wanted but unavailable, use CPU so the (large) model still
        # fits without competing with the GGUF model for T4 VRAM.
        device = "cpu" if want_8bit else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading embedder %s via sentence-transformers on %s", name, device)
        model = SentenceTransformer(name, trust_remote_code=True, device=device)
        if is_nv_embed:
            model.max_seq_length = self._max_len
            try:
                model.tokenizer.padding_side = "right"
            except Exception:  # noqa: BLE001
                pass
            self._append_eos = True
        self._model = model
        self._dim = model.get_sentence_embedding_dimension()

    def unload(self):
        """Free the local model from (V)RAM so the next phase's model can load."""
        import torch

        self._model = None
        self._client = None
        self._dim = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Embedding model unloaded; VRAM freed.")

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
            if self._model is None:
                raise RuntimeError(
                    "Embedding model is unloaded but a cache miss occurred. In the "
                    "3-phase Colab flow every text must be encoded during the "
                    "embedding phase; a miss here means a text was not pre-encoded."
                )
            if self._nv_native:
                import torch

                out = []
                bs = max(1, self._nv_batch)
                for start in range(0, len(texts), bs):
                    with torch.no_grad():
                        # NV-Embed-v2's own encode() handles EOS + pooling; the
                        # per-text instruction is already prepended by batch_encode,
                        # so pass instruction="" here.
                        emb = self._model.encode(
                            texts[start : start + bs], instruction="", max_length=self._max_len
                        )
                    out.append(np.asarray(emb.detach().to(torch.float32).cpu().numpy(), dtype=np.float32))
                vecs = np.vstack(out) if out else np.zeros((0, self.embedding_dim), dtype=np.float32)
            else:
                enc_texts = texts
                if self._append_eos:
                    eos = self._model.tokenizer.eos_token or ""
                    enc_texts = [t + eos for t in texts]
                vecs = np.asarray(
                    self._model.encode(
                        enc_texts,
                        batch_size=self.config.embedding_batch_size,
                        normalize_embeddings=False,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )
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
