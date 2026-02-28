from __future__ import annotations

import os
import re
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

load_dotenv()


class LLMResult(dict):
    """Small helper type."""


def _openai_compatible_chat(messages: List[Dict[str, str]]) -> str:
    """
    Calls an OpenAI-compatible /chat/completions endpoint.
    Uses only stdlib urllib to avoid extra dependencies.
    """
    import json
    import urllib.request

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode("utf-8"))
        return out["choices"][0]["message"]["content"]


def _local_flan_t5(prompt: str) -> str:
    """
    Offline fallback generator.
    Note: quality is lower than API models, but good enough for assignment demos.
    """
    from transformers import pipeline

    # Use small model to keep downloads manageable
    gen = pipeline("text2text-generation", model="google/flan-t5-base")
    out = gen(prompt, do_sample=False, max_new_tokens=180, truncation=True)
    return out[0]["generated_text"].strip()


def _compact_context_for_local(question: str, context: str, max_chars: int = 1800) -> str:
    """
    Keep local-model prompt short and focused so Flan-T5 doesn't lose key facts.
    """
    normalized = " ".join(context.split())
    if len(normalized) <= max_chars:
        return normalized

    keywords = [k for k in re.findall(r"[A-Za-z]+", question.lower()) if len(k) >= 4]
    if not keywords:
        return normalized[:max_chars]

    chunks = re.split(r"(?=\[Page\s+\d+\])", context)
    matched: List[str] = []
    for chunk in chunks:
        lower = chunk.lower()
        if any(keyword in lower for keyword in keywords):
            matched.append(" ".join(chunk.split()))

    focused = " ".join(matched) if matched else normalized
    return focused[:max_chars]


def generate_answer(question: str, context: str) -> str:
    """
    Generate answer from context only.
    Provider:
      - openai_compatible (best, if env vars set)
      - local (default)
    """
    provider = os.getenv("LLM_PROVIDER", "local").strip().lower()

    system = (
        "You are a QA assistant. Answer ONLY from the provided context.\n"
        "If the answer is not present in the context, reply exactly: Not found in the report.\n"
        "Be concise. When possible, mention page numbers if given in the context."
    )
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:"
    if provider == "openai_compatible":
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return _openai_compatible_chat(messages).strip()

    # local fallback
    compact_context = _compact_context_for_local(question, context)
    compact_user = f"QUESTION:\n{question}\n\nCONTEXT:\n{compact_context}\n\nANSWER:"
    prompt = system + "\n\n" + compact_user
    return _local_flan_t5(prompt).strip()
