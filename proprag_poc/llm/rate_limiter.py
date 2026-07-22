"""Process-wide request rate limiter (sliding 60s window).

A single shared limiter caps the request rate against the online provider
(NVIDIA) so it is not hit faster than its free-tier allowance. Only *real
network calls* should acquire a slot — cache hits and local backends must skip
it (pass through ``acquire`` only when needed).

By default this only guards the chat LLM client: embeddings run locally
(``sentence_transformers``) and never acquire a slot. If the embedding backend
is switched to an online one (``nvidia``/``openai``), it shares this same
limiter with chat, keyed by ``config.rpm_limit``, and batches count as a single
slot regardless of how many texts they carry.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Thread-safe sliding-window limiter: at most ``rpm`` acquisitions per 60s."""

    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a slot is free in the trailing 60s window.

        Returns the seconds spent waiting (0.0 if no wait was needed).
        """
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps older than the window.
                while self._times and now - self._times[0] >= _WINDOW_SECONDS:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return waited
                # Otherwise wait until the oldest timestamp exits the window.
                sleep_for = _WINDOW_SECONDS - (now - self._times[0]) + 0.01
            logger.info(
                "rate limit reached (%d/%d in 60s) - waiting %.1fs",
                self.rpm, self.rpm, max(0.0, sleep_for),
            )
            time.sleep(max(0.0, sleep_for))
            waited += max(0.0, sleep_for)


# ----------------------------------------------------------------- singleton
_INSTANCES: Dict[int, RateLimiter] = {}
_INSTANCES_LOCK = threading.Lock()


def get_rate_limiter(rpm: int) -> RateLimiter:
    """Return the shared limiter for a given RPM (one instance per rpm value).

    Chat and embedding clients call this with the same ``config.rpm_limit`` so they
    share one window.
    """
    with _INSTANCES_LOCK:
        inst = _INSTANCES.get(rpm)
        if inst is None:
            inst = RateLimiter(rpm)
            _INSTANCES[rpm] = inst
        return inst
