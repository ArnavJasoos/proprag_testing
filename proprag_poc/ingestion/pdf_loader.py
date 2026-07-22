"""PDF text extraction with noise filtering (page numbers, repeated
headers/footers, image-only blocks). Digital-text PDFs; no OCR.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from ..config import POCConfig

logger = logging.getLogger(__name__)

_PAGE_NUM = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_ROMAN = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE)


@dataclass
class LoadedDoc:
    title: str
    text: str            # cleaned full text
    source: str          # file path / name
    num_pages: int


class PDFLoader:
    def __init__(self, config: POCConfig):
        self.config = config

    def load(self, path: str) -> LoadedDoc:
        import fitz  # PyMuPDF; lazy so retrieval/QA paths don't require it

        doc = fitz.open(path)
        title = self._title(doc, path)
        page_lines: List[List[str]] = []
        for page in doc:
            text = page.get_text("text") or ""
            lines = [ln.rstrip() for ln in text.splitlines()]
            page_lines.append(lines)
        doc.close()

        noise = self._header_footer_lines(page_lines) if self.config.drop_repeated_headers else set()
        cleaned_pages = []
        for lines in page_lines:
            kept = [ln for ln in lines if not self._is_noise(ln, noise)]
            page_text = self._join_lines(kept)
            if page_text.strip():
                cleaned_pages.append(page_text)

        full = "\n\n".join(cleaned_pages).strip()
        return LoadedDoc(title=title, text=full, source=os.path.basename(path),
                         num_pages=len(page_lines))

    # ------------------------------------------------------------- helpers
    def _title(self, doc, path: str) -> str:
        meta_title = (doc.metadata or {}).get("title") or ""
        meta_title = meta_title.strip()
        if meta_title:
            return meta_title
        return os.path.splitext(os.path.basename(path))[0]

    def _header_footer_lines(self, page_lines) -> set:
        """Lines appearing on a large fraction of pages = running header/footer."""
        n_pages = len(page_lines)
        if n_pages < 4:
            return set()
        counter = Counter()
        for lines in page_lines:
            for ln in set(l.strip() for l in lines if l.strip()):
                counter[ln] += 1
        threshold = max(2, int(self.config.header_repeat_fraction * n_pages))
        return {ln for ln, c in counter.items() if c >= threshold and len(ln) < 120}

    def _is_noise(self, line: str, noise: set) -> bool:
        s = line.strip()
        if not s:
            return False  # keep blanks for paragraph joining; dropped later
        if self.config.drop_page_numbers and (_PAGE_NUM.match(s) or _ROMAN.match(s)):
            return True
        if s in noise:
            return True
        return False

    def _join_lines(self, lines: List[str]) -> str:
        """Merge hard-wrapped lines into paragraphs; collapse blank runs."""
        out: List[str] = []
        buf: List[str] = []
        for ln in lines:
            if ln.strip():
                buf.append(ln.strip())
            else:
                if buf:
                    out.append(" ".join(buf))
                    buf = []
        if buf:
            out.append(" ".join(buf))
        return "\n".join(out)
