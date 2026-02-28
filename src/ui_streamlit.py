from __future__ import annotations

import re

import streamlit as st
from src.rag.rag_pipeline import answer_question


def _format_answer_for_display(answer: str) -> str:
    cleaned = " ".join(answer.split()).strip()
    if not cleaned:
        return answer

    if cleaned.lower() == "not found in the report.":
        return "Not found in the report."

    if "\n" in answer:
        return answer.strip()

    pair_pattern = re.compile(
        r"([A-Za-z][A-Za-z/&(),:+*\-\s]{2,}?)\s+(\(?-?\d[\d,]*\.?\d*\)?)\s+(\(?-?\d[\d,]*\.?\d*\)?)"
    )
    rows = [(m.group(1).strip(), m.group(2), m.group(3)) for m in pair_pattern.finditer(cleaned)]

    if len(rows) >= 3:
        lines = ["| Metric | Value 1 | Value 2 |", "|---|---:|---:|"]
        for metric, val1, val2 in rows:
            lines.append(f"| {metric} | {val1} | {val2} |")
        return "\n".join(lines)

    sentence_lines = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if len(sentence_lines) >= 2:
        return "\n".join(f"- {line}" for line in sentence_lines)

    return cleaned

st.set_page_config(page_title="Swiggy Annual Report QA", page_icon="📄", layout="centered")

with st.container(border=True):
    st.markdown("## 📄 Swiggy Annual Report RAG QA")
    st.caption("Ask a question and get an answer based on the report.")

with st.container(border=True):
    st.markdown("### 💬 Ask a Question")

    with st.form("qa_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            placeholder=(
                "Examples:\n"
                "• What were the key highlights for FY 2023–24?\n"
                "• Which business segments are described in the report?"
            ),
            height=120,
        )
        submitted = st.form_submit_button("Get Answer", use_container_width=True)

if submitted:
    user_query = question.strip()

    if not user_query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            try:
                answer, _ = answer_question(user_query, k=4)
            except Exception as err:
                st.error(f"Could not answer right now: {err}")
                st.info("If index is missing, run: `python scripts/build_index.py --pdf data/Swiggy_Annual_Report.pdf`")
                st.stop()

        with st.container(border=True):
            st.markdown("### ✅ Final Answer")
            st.success(_format_answer_for_display(answer))
