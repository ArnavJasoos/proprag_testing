"""Token-based chunking with overlap (tiktoken), paragraph-aware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken

from ..config import POCConfig
from .pdf_loader import LoadedDoc


@dataclass
class Chunk:
    text: str           # "title\n<chunk text>" — matches reference doc format
    title: str
    source: str
    index: int


class Chunker:
    def __init__(self, config: POCConfig):
        self.config = config
        self.enc = tiktoken.get_encoding(config.tiktoken_encoding)

    def chunk_doc(self, doc: LoadedDoc) -> List[Chunk]:
        paragraphs = [p.strip() for p in doc.text.split("\n") if p.strip()]
        max_tok = self.config.chunk_max_tokens
        overlap = self.config.chunk_overlap_tokens

        # Pack paragraphs greedily into token-bounded windows.
        windows: List[str] = []
        cur: List[str] = []
        cur_tok = 0
        for para in paragraphs:
            ptok = len(self.enc.encode(para))
            if ptok > max_tok:
                # Split an oversized paragraph by tokens.
                for piece in self._split_tokens(para, max_tok, overlap):
                    windows.append(piece)
                continue
            if cur_tok + ptok > max_tok and cur:
                windows.append("\n".join(cur))
                cur, cur_tok = self._carry_overlap(cur, overlap)
            cur.append(para)
            cur_tok += ptok
        if cur:
            windows.append("\n".join(cur))

        chunks = []
        for i, w in enumerate(windows):
            chunks.append(
                Chunk(text=f"{doc.title}\n{w}", title=doc.title, source=doc.source, index=i)
            )
        return chunks

    def _split_tokens(self, text: str, max_tok: int, overlap: int) -> List[str]:
        toks = self.enc.encode(text)
        out = []
        step = max(1, max_tok - overlap)
        for start in range(0, len(toks), step):
            out.append(self.enc.decode(toks[start : start + max_tok]))
            if start + max_tok >= len(toks):
                break
        return out

    def _carry_overlap(self, paragraphs: List[str], overlap: int):
        """Keep trailing paragraphs up to ``overlap`` tokens for the next window."""
        kept: List[str] = []
        tok = 0
        for para in reversed(paragraphs):
            ptok = len(self.enc.encode(para))
            if tok + ptok > overlap:
                break
            kept.insert(0, para)
            tok += ptok
        return kept, tok
