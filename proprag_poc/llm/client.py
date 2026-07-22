"""Chat LLM client with a SQLite response cache, retry, parallel inference,
a shared request rate limiter, and token-usage instrumentation.

Two backends are dispatched transparently:
  * ``google``  -> the native google-genai SDK (Gemini): system prompt via
    ``system_instruction``, roles ``user`` / ``model``.
  * everything else (``nvidia``, ``openrouter``, ``koboldcpp``, ``ollama``,
    ``vllm``) -> the OpenAI-compatible Chat Completions API via the ``openai`` SDK.

Online backends (NVIDIA / Gemini / OpenRouter) acquire a slot from the shared
``RateLimiter`` before each network call and report token usage + latency to the
``UsageTracker`` so the Compare view can attribute cost per RAG system.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from ..config import POCConfig
from ..core.metrics import CallRecord, get_usage_tracker
from .rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


def _split_system(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Separate system messages from the turn list and remap roles for Gemini."""
    system_parts, turns = [], []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_parts.append(m["content"])
        else:
            gem_role = "model" if role == "assistant" else "user"
            turns.append({"role": gem_role, "content": m["content"]})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, turns


class LLMClient:
    """Cached chat client (Gemini or OpenAI-compatible) with rate limiting + metrics."""

    def __init__(self, config: POCConfig):
        self.config = config
        self._backend = config.llm_backend
        self._tracker = get_usage_tracker()
        self._limiter = get_rate_limiter(config.rpm_limit) if config.llm_is_online else None
        # Some OpenAI-compatible servers reject response_format; drop it after a failure.
        self._json_response_format_ok = True

        if self._backend == "google":
            from google import genai  # lazy: keep package optional at import time

            self._genai = genai
            from google.genai import types

            self._types = types
            self.client = genai.Client(api_key=config.llm_api_key)
        else:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=config.llm_base_url,
                api_key=config.llm_api_key or "not-needed",
                timeout=config.request_timeout,
            )

        self._cache_path = os.path.join(config.data_dir, "llm_cache.sqlite")
        self._lock = threading.Lock()
        self._call_n = 0  # completed network calls, for progress logging
        self._init_cache()

    # ----------------------------------------------------------------- cache
    def _init_cache(self):
        with sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
            )

    def _cache_key(self, system, turns, temperature, max_tokens, json_mode) -> str:
        payload = json.dumps(
            {
                "system": system,
                "messages": turns,
                "backend": self._backend,
                "model": self.config.llm_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json": json_mode,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        with self._lock, sqlite3.connect(self._cache_path) as conn:
            row = conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _cache_put(self, key: str, value: str):
        with self._lock, sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
            )

    # ----------------------------------------------------------------- infer
    def infer(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        json_mode: bool = False,
        use_cache: bool = True,
        max_completion_tokens: Optional[int] = None,
        response_checker: Optional[Callable[[str], bool]] = None,
    ) -> Tuple[str, Dict, bool]:
        """Return ``(content, metadata, cache_hit)``."""
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = max_completion_tokens or self.config.max_completion_tokens
        system, turns = _split_system(messages)
        key = self._cache_key(system, turns, temperature, max_tokens, json_mode)

        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                self._tracker.record(
                    CallRecord(kind="chat", model=self.config.llm_model, cache_hit=True)
                )
                return cached, {"finish_reason": "cached"}, True

        last_err: Optional[Exception] = None
        attempt_temp = temperature
        rate_wait_total = 0.0
        for attempt in range(self.config.max_retry_attempts):
            if self._limiter is not None:
                rate_wait_total += self._limiter.acquire()
            with self._lock:
                call_no = self._call_n + 1
            logger.info(
                "chat #%d: request sent (attempt %d/%d)...",
                call_no, attempt + 1, self.config.max_retry_attempts,
            )
            t0 = time.monotonic()
            try:
                content, prompt_tok, compl_tok = self._call_backend(
                    system, turns, attempt_temp, max_tokens, json_mode
                )
                latency = time.monotonic() - t0
                if not content:
                    raise RuntimeError("empty response (possibly blocked or truncated)")
                if response_checker is not None and not response_checker(content):
                    attempt_temp = min(1.0, attempt_temp + 0.1)
                    last_err = ValueError("response_checker rejected output")
                    continue
                self._tracker.record(
                    CallRecord(
                        kind="chat",
                        model=self.config.llm_model,
                        prompt_tokens=prompt_tok,
                        completion_tokens=compl_tok,
                        latency_s=latency,
                        rate_wait_s=rate_wait_total,
                    )
                )
                with self._lock:
                    self._call_n += 1
                    n = self._call_n
                logger.info(
                    "chat #%d done in %.1fs (%d+%d tok)%s",
                    n, latency, prompt_tok, compl_tok,
                    f", waited {rate_wait_total:.1f}s" if rate_wait_total > 0.05 else "",
                )
                if use_cache:
                    self._cache_put(key, content)
                return content, {"finish_reason": "stop"}, False
            except Exception as e:  # noqa: BLE001 - retried below
                last_err = e
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
        raise RuntimeError(f"LLM inference failed after retries: {last_err}")

    # ------------------------------------------------------ backend dispatch
    def _call_backend(self, system, turns, temperature, max_tokens, json_mode):
        if self._backend == "google":
            return self._call_google(system, turns, temperature, max_tokens, json_mode)
        return self._call_openai(system, turns, temperature, max_tokens, json_mode)

    def _call_google(self, system, turns, temperature, max_tokens, json_mode):
        types = self._types
        contents = [
            types.Content(role=t["role"], parts=[types.Part.from_text(text=t["content"])])
            for t in turns
        ]
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = self.client.models.generate_content(
            model=self.config.llm_model, contents=contents, config=cfg
        )
        content = (resp.text or "").strip()
        usage = getattr(resp, "usage_metadata", None)
        prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        compl_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        return content, prompt_tok, compl_tok

    def _call_openai(self, system, turns, temperature, max_tokens, json_mode):
        # OpenAI-compatible expects roles user/assistant (turns use user/model).
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for t in turns:
            role = "assistant" if t["role"] == "model" else "user"
            msgs.append({"role": role, "content": t["content"]})

        kwargs = dict(
            model=self.config.llm_model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        use_rf = json_mode and self._json_response_format_ok
        if use_rf:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            # Some NIM endpoints reject response_format; retry once without it and
            # disable it for the rest of the session (prompt + lenient parsing cope).
            if use_rf and "response_format" in str(e).lower():
                self._json_response_format_ok = False
                kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        prompt_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        compl_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        return content, prompt_tok, compl_tok

    # --------------------------------------------------------------- parallel
    def infer_many(
        self,
        message_list: List[List[Dict[str, str]]],
        json_mode: bool = False,
        max_completion_tokens: Optional[int] = None,
        desc: str = "LLM",
    ) -> List[str]:
        """Run many chat calls in parallel; preserves input order."""
        results: List[Optional[str]] = [None] * len(message_list)

        def _work(i: int):
            content, _, _ = self.infer(
                message_list[i],
                json_mode=json_mode,
                max_completion_tokens=max_completion_tokens,
            )
            results[i] = content

        with ThreadPoolExecutor(max_workers=self.config.llm_max_workers) as ex:
            list(ex.map(_work, range(len(message_list))))
        return [r if r is not None else "" for r in results]
