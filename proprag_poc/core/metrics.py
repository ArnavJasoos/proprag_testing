"""Usage / cost / latency instrumentation shared across RAG systems.

A process-wide ``UsageTracker`` records every chat and embedding call (token
counts, latency, cache hits). Query-time calls are attributed to a *scope* (one
per RAG system) so the Compare view can show token consumption, call counts and
latency side-by-side. Index-time calls land in the default scope and are reported
once (the corpus/stores/graph are shared by all three systems).

The tracker is a singleton so the chat client and embedding client can report
into it without threading an object through every constructor.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_DEFAULT_SCOPE = "index"


@dataclass
class CallRecord:
    kind: str            # "chat" | "embed"
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    rate_wait_s: float = 0.0
    cache_hit: bool = False
    batch_size: int = 1  # number of texts in an embedding request


@dataclass
class Snapshot:
    """Aggregated usage for one scope."""

    scope: str
    chat_calls: int = 0
    chat_cache_hits: int = 0
    chat_prompt_tokens: int = 0
    chat_completion_tokens: int = 0
    embed_calls: int = 0
    embed_cache_hits: int = 0
    embed_texts: int = 0
    embed_tokens: int = 0
    chat_latency_s: float = 0.0
    embed_latency_s: float = 0.0
    rate_wait_s: float = 0.0

    @property
    def chat_total_tokens(self) -> int:
        return self.chat_prompt_tokens + self.chat_completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.chat_total_tokens + self.embed_tokens

    @property
    def total_calls(self) -> int:
        return self.chat_calls + self.embed_calls

    def as_dict(self) -> Dict[str, float]:
        return {
            "chat_calls": self.chat_calls,
            "chat_cache_hits": self.chat_cache_hits,
            "chat_prompt_tokens": self.chat_prompt_tokens,
            "chat_completion_tokens": self.chat_completion_tokens,
            "chat_total_tokens": self.chat_total_tokens,
            "embed_calls": self.embed_calls,
            "embed_cache_hits": self.embed_cache_hits,
            "embed_texts": self.embed_texts,
            "embed_tokens": self.embed_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "chat_latency_s": round(self.chat_latency_s, 3),
            "embed_latency_s": round(self.embed_latency_s, 3),
            "rate_wait_s": round(self.rate_wait_s, 3),
        }


class UsageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        # scope name -> list of CallRecord. Thread-local current scope.
        self._records: Dict[str, List[CallRecord]] = {}
        self._tl = threading.local()

    # ------------------------------------------------------------- scoping
    def _current_scope(self) -> str:
        return getattr(self._tl, "scope", _DEFAULT_SCOPE)

    @contextmanager
    def scope(self, name: str):
        """Attribute calls made on this thread to ``name`` for the block's duration."""
        prev = getattr(self._tl, "scope", None)
        self._tl.scope = name
        try:
            yield
        finally:
            if prev is None:
                del self._tl.scope
            else:
                self._tl.scope = prev

    # ------------------------------------------------------------- record
    def record(self, rec: CallRecord, scope: Optional[str] = None):
        scope = scope or self._current_scope()
        with self._lock:
            self._records.setdefault(scope, []).append(rec)

    # ------------------------------------------------------------- read
    def snapshot(self, scope: str) -> Snapshot:
        with self._lock:
            recs = list(self._records.get(scope, []))
        snap = Snapshot(scope=scope)
        for r in recs:
            if r.kind == "chat":
                snap.chat_calls += 1
                snap.chat_cache_hits += int(r.cache_hit)
                snap.chat_prompt_tokens += r.prompt_tokens
                snap.chat_completion_tokens += r.completion_tokens
                snap.chat_latency_s += r.latency_s
            else:
                snap.embed_calls += 1
                snap.embed_cache_hits += int(r.cache_hit)
                snap.embed_texts += r.batch_size
                snap.embed_tokens += r.prompt_tokens
                snap.embed_latency_s += r.latency_s
            snap.rate_wait_s += r.rate_wait_s
        return snap

    def reset(self, scope: Optional[str] = None):
        with self._lock:
            if scope is None:
                self._records.clear()
            else:
                self._records.pop(scope, None)

    def scopes(self) -> List[str]:
        with self._lock:
            return list(self._records.keys())


# ----------------------------------------------------------------- singleton
_TRACKER: Optional[UsageTracker] = None
_TRACKER_LOCK = threading.Lock()


def get_usage_tracker() -> UsageTracker:
    global _TRACKER
    with _TRACKER_LOCK:
        if _TRACKER is None:
            _TRACKER = UsageTracker()
        return _TRACKER
