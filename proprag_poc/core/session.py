"""Persisted ChatGPT-style chat sessions (SQLite).

Each session has a title, a bound corpus id, and an ordered message list. Used by
the Streamlit GUI for the sidebar session list and multi-turn history.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str
    refs: List[str] = field(default_factory=list)  # cited chunk ids


@dataclass
class Session:
    id: str
    title: str
    corpus_id: Optional[str]
    created_at: float
    messages: List[Message]


class SessionStore:
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "sessions.sqlite")
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "id TEXT PRIMARY KEY, title TEXT, corpus_id TEXT, "
                "created_at REAL, messages TEXT)"
            )

    # ------------------------------------------------------------- mutate
    def create(self, title: str = "New chat", corpus_id: Optional[str] = None) -> Session:
        s = Session(uuid.uuid4().hex[:12], title, corpus_id, time.time(), [])
        self._save(s)
        return s

    def rename(self, session_id: str, title: str):
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))

    def set_corpus(self, session_id: str, corpus_id: str):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE sessions SET corpus_id=? WHERE id=?", (corpus_id, session_id)
            )

    def append(self, session_id: str, message: Message):
        s = self.get(session_id)
        s.messages.append(message)
        self._save(s)

    def delete(self, session_id: str):
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    # ------------------------------------------------------------- query
    def get(self, session_id: str) -> Session:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT id, title, corpus_id, created_at, messages FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        return self._row_to_session(row)

    def list(self) -> List[Session]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT id, title, corpus_id, created_at, messages FROM sessions "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    # ------------------------------------------------------------- helpers
    def history_for_llm(self, session_id: str) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.get(session_id).messages]

    def _save(self, s: Session):
        payload = json.dumps([{"role": m.role, "content": m.content, "refs": m.refs}
                              for m in s.messages])
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (id, title, corpus_id, created_at, messages) "
                "VALUES (?, ?, ?, ?, ?)",
                (s.id, s.title, s.corpus_id, s.created_at, payload),
            )

    @staticmethod
    def _row_to_session(row) -> Session:
        msgs = [Message(**m) for m in json.loads(row[4])]
        return Session(id=row[0], title=row[1], corpus_id=row[2],
                       created_at=row[3], messages=msgs)
