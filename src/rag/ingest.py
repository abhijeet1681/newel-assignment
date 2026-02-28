from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import re

from pypdf import PdfReader


@dataclass
class PageDoc:
    page: int
    text: str


def load_pdf_pages(pdf_path: str) -> List[PageDoc]:
    """Load PDF and return list of PageDoc with page numbers (1-indexed)."""
    reader = PdfReader(pdf_path)
    pages: List[PageDoc] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageDoc(page=i, text=text))
    return pages


def clean_text(text: str) -> str:
    """Basic cleanup: normalize whitespace and remove repeated headers/footers-like artifacts."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove very common page number-only lines
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    return text.strip()
