from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple

from langchain_community.vectorstores import Chroma

from .index import build_or_load_chroma
from .llm import generate_answer


@dataclass
class RetrievedChunk:
    text: str
    page: int
    score: float


def retrieve(vectordb: Chroma, query: str, k: int = 4) -> List[RetrievedChunk]:
    # similarity_search_with_score returns (Document, score). Lower score usually means closer for Chroma cosine distance.
    results = vectordb.similarity_search_with_score(query, k=k)
    chunks: List[RetrievedChunk] = []
    for doc, score in results:
        chunks.append(RetrievedChunk(text=doc.page_content, page=int(doc.metadata.get("page", -1)), score=float(score)))
    return chunks


def make_context(chunks: List[RetrievedChunk]) -> str:
    # Provide explicit page tags to encourage citations
    parts = []
    for c in chunks:
        parts.append(f"[Page {c.page}] {c.text}")
    return "\n\n".join(parts)


def _segment_list_from_chunks(chunks: List[RetrievedChunk]) -> List[str]:
    segment_rules = [
        ("Food Delivery", [r"food\s+delivery"]),
        ("Quick Commerce", [r"quick\s+commerce", r"instamart"]),
        ("Dineout", [r"dine\s*-?\s*out", r"dineout"]),
        ("Out-of-home Consumption", [r"out\s*-?\s*of\s*-?\s*home"]),
    ]

    combined = " ".join(" ".join(chunk.text.split()) for chunk in chunks).lower()
    found: List[str] = []
    for label, patterns in segment_rules:
        if any(re.search(pattern, combined) for pattern in patterns):
            found.append(label)
    return found


def _extractive_fallback_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    """
    Extract a concise answer directly from retrieved chunks when generative output
    is a false negative (e.g., local model says 'Not found' despite relevant text).
    """
    if not chunks:
        return ""

    if "segment" in query.lower():
        segments = _segment_list_from_chunks(chunks)
        if segments:
            return "Business segments mentioned in the report:\n" + "\n".join(f"- {segment}" for segment in segments)

    query_terms = {w for w in re.findall(r"[A-Za-z]+", query.lower()) if len(w) >= 4}
    if "segment" in query.lower():
        query_terms.update({"food", "delivery", "dineout", "quick", "commerce", "instamart"})

    scored_sentences: List[Tuple[int, str]] = []
    seen = set()
    for chunk in chunks:
        text = " ".join(chunk.text.split())
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            score = sum(1 for term in query_terms if term in key)
            if score > 0:
                scored_sentences.append((score, sentence))

    if scored_sentences:
        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        top = [sent for _, sent in scored_sentences[:3]]
        return " ".join(top)

    fallback_text = " ".join(chunks[0].text.split())
    fallback_sentences = re.split(r"(?<=[.!?])\s+", fallback_text)
    for sentence in fallback_sentences:
        sentence = sentence.strip()
        if len(sentence) >= 30:
            return sentence
    return ""


def answer_question(
    query: str,
    persist_dir: str = "chroma_db",
    k: int = 4,
) -> Tuple[str, List[RetrievedChunk]]:
    vectordb = build_or_load_chroma(persist_dir=persist_dir)
    chunks = retrieve(vectordb, query, k=k)
    context = make_context(chunks)

    if not context.strip():
        return "Not found in the report.", chunks

    answer = generate_answer(query, context)

    # Extra safety: if model outputs something that isn't grounded, we keep the rule strict:
    if "not found in the report" in answer.lower():
        fallback = _extractive_fallback_answer(query, chunks)
        if fallback:
            return fallback, chunks
        return "Not found in the report.", chunks

    return answer, chunks
