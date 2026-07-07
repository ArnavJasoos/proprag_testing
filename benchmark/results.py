"""Append-only, resume-safe results store (one JSON line per (question, system))."""

from __future__ import annotations

import json
import os
from typing import Dict, Set, Tuple


class ResultsStore:
    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, "results.jsonl")

    def done_keys(self) -> Set[Tuple[str, str]]:
        """(qid, system) pairs already recorded *successfully*.

        Error rows are excluded so a rerun retries transient failures; the report
        deduplicates by (qid, system) keeping the last row.
        """
        done: Set[Tuple[str, str]] = set()
        if not os.path.isfile(self.path):
            return done
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("error"):
                    continue
                if "qid" in row and "system" in row:
                    done.add((row["qid"], row["system"]))
        return done

    def append(self, row: Dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
