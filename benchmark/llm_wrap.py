"""LLM client wrapper for gpt-oss inference.

Desktop runs can still use the original Koboldcpp/OpenAI-compatible path. Colab
uses PROPRAG_LLM_BACKEND=llama_cpp plus PROPRAG_GGUF_MODEL_PATH to load a
pre-quantized GGUF file directly with llama-cpp-python. That avoids a localhost
inference server and avoids quantizing while loading the model.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List

from . import _bootstrap  # noqa: F401
from proprag_poc.config import POCConfig
from proprag_poc.core.metrics import CallRecord, get_usage_tracker
from proprag_poc.llm.client import LLMClient

_LLAMA_CPP_MODEL_LABEL = "gpt-oss-20b-gguf"

logger = logging.getLogger(__name__)

_FINAL_MARKER = "<|channel|>final<|message|>"
_TERMINATORS = ("<|end|>", "<|return|>", "<|start|>")


def strip_gpt_oss_reasoning(content: str) -> str:
    """Remove the gpt-oss Harmony reasoning channel, keep the final answer text."""
    if not content:
        return content
    if _FINAL_MARKER in content:
        content = content.split(_FINAL_MARKER)[-1]
        for term in _TERMINATORS:
            content = content.split(term)[0]
        return content.strip()
    stripped = content.lstrip()
    if stripped.startswith("analysis") and "assistantfinal" in stripped:
        return stripped.split("assistantfinal")[-1].strip()
    return content


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class BenchLLMClient(LLMClient):
    """LLMClient with gpt-oss post-processing and optional direct GGUF mode."""

    def __init__(self, config: POCConfig):
        self._direct_llama = config.llm_backend == "llama_cpp"
        if self._direct_llama:
            # Skip LLMClient.__init__ (it would build an OpenAI HTTP client for a
            # preset we're not using), but still wire up the SQLite response cache
            # and UsageTracker plumbing LLMClient.infer() normally provides -
            # otherwise every direct call would be uncached and untracked, which
            # would silently zero out this benchmark's token-consumption numbers
            # and break resume-on-interrupt for Colab runs.
            self.config = config
            self._backend = "llama_cpp"
            self._tracker = get_usage_tracker()
            self._lock = threading.Lock()
            self._cache_path = os.path.join(config.data_dir, "llm_cache.sqlite")
            self._init_cache()
        else:
            super().__init__(config)
        self._json_response_format_ok = False
        self._llama = None
        self._llama_lock = threading.Lock()
        if self._direct_llama:
            self._llama = self._load_llama_cpp()

    def _load_llama_cpp(self):
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover - depends on Colab install
            raise RuntimeError(
                "PROPRAG_LLM_BACKEND=llama_cpp requires llama-cpp-python. "
                "Install it in Colab before running the benchmark."
            ) from e

        model_path = os.environ.get("PROPRAG_GGUF_MODEL_PATH", "")
        if not model_path or not os.path.isfile(model_path):
            raise RuntimeError(
                "PROPRAG_GGUF_MODEL_PATH must point to the downloaded GGUF file."
            )

        n_gpu_layers = _env_int("PROPRAG_LLAMA_N_GPU_LAYERS", -1)
        n_ctx = _env_int("PROPRAG_LLAMA_N_CTX", 8192)
        n_batch = _env_int("PROPRAG_LLAMA_N_BATCH", 512)
        n_threads = _env_int("PROPRAG_LLAMA_N_THREADS", 2)
        logger.info("Loading GGUF with llama.cpp: %s", model_path)
        return Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            verbose=False,
        )

    def infer(self, messages: List[Dict[str, str]], *args, **kwargs):
        if not self._direct_llama:
            content, meta, cache_hit = super().infer(messages, *args, **kwargs)
            if self.config.strip_reasoning:
                content = strip_gpt_oss_reasoning(content)
            return content, meta, cache_hit
        return self._infer_direct(messages, *args, **kwargs)

    def _infer_direct(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        json_mode: bool = False,
        use_cache: bool = True,
        max_completion_tokens: int = None,
        response_checker=None,
    ):
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = max_completion_tokens or self.config.max_completion_tokens
        # `system=None, turns=messages`: the exact split doesn't matter here, this
        # is purely a cache-key input, and "backend" already disambiguates it from
        # the HTTP cache path's keys.
        key = self._cache_key(None, messages, temperature, max_tokens, json_mode)

        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                self._tracker.record(
                    CallRecord(kind="chat", model=_LLAMA_CPP_MODEL_LABEL, cache_hit=True)
                )
                content = strip_gpt_oss_reasoning(cached) if self.config.strip_reasoning else cached
                return content, {"finish_reason": "cached"}, True

        t0 = time.monotonic()
        with self._llama_lock:
            response = self._llama.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self.config.seed if self.config.seed is not None else -1,
            )
        latency = time.monotonic() - t0
        choice = (response.get("choices") or [{}])[0]
        raw_content = (choice.get("message") or {}).get("content") or choice.get("text") or ""
        if not raw_content:
            raise RuntimeError("empty response from local llama.cpp model")

        usage = response.get("usage") or {}
        self._tracker.record(
            CallRecord(
                kind="chat",
                model=_LLAMA_CPP_MODEL_LABEL,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_s=latency,
            )
        )
        if use_cache:
            self._cache_put(key, raw_content)

        content = strip_gpt_oss_reasoning(raw_content) if self.config.strip_reasoning else raw_content
        return content, {"finish_reason": choice.get("finish_reason", "stop")}, False


def check_backend(poc_cfg: POCConfig, timeout: float = 5.0) -> None:
    """Validate whichever LLM backend is configured."""
    if poc_cfg.llm_backend == "llama_cpp":
        model_path = os.environ.get("PROPRAG_GGUF_MODEL_PATH", "")
        if not model_path or not os.path.isfile(model_path):
            raise RuntimeError(
                "Cannot find the GGUF model. Set PROPRAG_GGUF_MODEL_PATH after "
                "downloading gpt-oss-20b-Q4_K_M.gguf from Hugging Face."
            )
        logger.info("Direct llama.cpp backend ready: %s", model_path)
        return

    base = (poc_cfg.llm_base_url or "").rstrip("/")
    url = f"{base}/models"
    hint = (
        "Cannot reach the LLM backend at "
        f"{poc_cfg.llm_base_url!r}. Start Koboldcpp first:\n"
        "  koboldcpp.exe --model gpt-oss-20b-Q4_K_M.gguf --port 5001 --contextsize 8192"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"{url} returned HTTP {resp.status}")
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        raise RuntimeError(f"{hint}\n(underlying error: {e})") from e
    logger.info("LLM backend reachable at %s", poc_cfg.llm_base_url)
