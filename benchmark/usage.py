"""Per-phase token/latency attribution over the shared ``UsageTracker``.

The tracker's ``scope()`` is thread-local, and the reused ``Extractor.batch_extract``
runs in a thread pool, so index-time extraction lands in the default ``"index"``
scope regardless of any active scope. Index builds run sequentially, so each
phase's cost is the *delta* of the relevant scopes across the phase.

- BaseRAG / PropRAG index: their calls land in ``"index"`` -> delta of ``"index"``.
- GraphRAG index: worker calls set ``"index::GraphRAG"`` explicitly, main-thread
  calls land in ``"index"``; summing both scopes' deltas covers both paths.
"""

from __future__ import annotations

import time
from typing import Dict, List


def delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    """Field-wise ``after - before`` over ``Snapshot.as_dict()`` outputs."""
    keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys}


def _add(into: Dict[str, float], other: Dict[str, float]) -> None:
    for k, v in other.items():
        into[k] = into.get(k, 0) + v


class IndexPhase:
    """Context manager capturing the usage delta of one index phase.

    ``self.usage`` is populated on exit: summed deltas of ``"index"`` plus any
    ``extra_scopes`` (e.g. ``"index::GraphRAG"``), and ``wall_time_s``.
    """

    def __init__(self, tracker, name: str, extra_scopes: List[str] = None):
        self._tracker = tracker
        self.name = name
        self._scopes = ["index"] + list(extra_scopes or [])
        self._before: Dict[str, Dict[str, float]] = {}
        self._t0 = 0.0
        self.usage: Dict[str, float] = {}

    def __enter__(self) -> "IndexPhase":
        self._before = {s: self._tracker.snapshot(s).as_dict() for s in self._scopes}
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc) -> bool:
        wall = time.monotonic() - self._t0
        combined: Dict[str, float] = {}
        for s in self._scopes:
            _add(combined, delta(self._before[s], self._tracker.snapshot(s).as_dict()))
        combined["wall_time_s"] = round(wall, 3)
        self.usage = combined
        return False
