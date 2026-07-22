"""Parquet-backed text store + aligned vector store (slim port of the reference
EmbeddingStore). One store per namespace: chunk / entity / proposition.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from ..embedding.encoder import EmbeddingModel
from .ids import compute_mdhash_id


class EmbeddingStore:
    def __init__(self, embedding_model: EmbeddingModel, directory: str, namespace: str):
        self.embedding_model = embedding_model
        self.namespace = namespace
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._text_path = os.path.join(directory, f"{namespace}_text.parquet")
        self._vec_path = os.path.join(directory, f"{namespace}_vectors.npy")
        self._ids: List[str] = []
        self._content: List[str] = []
        self._id_to_row: Dict[str, int] = {}
        self._vectors: np.ndarray | None = None
        self._load()

    # ----------------------------------------------------------------- io
    def _load(self):
        if os.path.isfile(self._text_path):
            df = pd.read_parquet(self._text_path)
            self._ids = df["hash_id"].tolist()
            self._content = df["content"].tolist()
            self._id_to_row = {h: i for i, h in enumerate(self._ids)}
        if os.path.isfile(self._vec_path):
            self._vectors = np.load(self._vec_path)

    def _persist(self):
        pd.DataFrame({"hash_id": self._ids, "content": self._content}).to_parquet(
            self._text_path, index=False
        )
        if self._vectors is not None:
            np.save(self._vec_path, self._vectors)

    # ------------------------------------------------------------- mutation
    def insert_strings(self, texts: List[str]):
        """Hash, dedupe against existing, encode new, append. Idempotent."""
        new_ids, new_texts = [], []
        seen = set()
        for t in texts:
            hid = compute_mdhash_id(t, prefix=f"{self.namespace}-")
            if hid in self._id_to_row or hid in seen:
                continue
            seen.add(hid)
            new_ids.append(hid)
            new_texts.append(t)
        if not new_ids:
            return
        new_vecs = self.embedding_model.batch_encode(new_texts, norm=True)
        start = len(self._ids)
        self._ids.extend(new_ids)
        self._content.extend(new_texts)
        for j, hid in enumerate(new_ids):
            self._id_to_row[hid] = start + j
        self._vectors = (
            new_vecs if self._vectors is None else np.vstack([self._vectors, new_vecs])
        )
        self._persist()

    # -------------------------------------------------------------- queries
    def get_row(self, hash_id: str) -> Dict[str, str] | None:
        idx = self._id_to_row.get(hash_id)
        return None if idx is None else {"content": self._content[idx]}

    def get_embedding(self, hash_id: str) -> np.ndarray:
        return self._vectors[self._id_to_row[hash_id]]

    def get_embeddings(self, hash_ids: List[str]) -> np.ndarray:
        if not hash_ids:
            return np.zeros((0, self._vectors.shape[1]), dtype=np.float32)
        return np.stack([self._vectors[self._id_to_row[h]] for h in hash_ids])

    def get_all_ids(self) -> List[str]:
        return list(self._ids)

    def get_text_for_all_rows(self) -> Dict[str, Dict[str, str]]:
        return {h: {"content": self._content[i]} for i, h in enumerate(self._ids)}

    def __len__(self):
        return len(self._ids)
