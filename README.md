# Swiggy Annual Report RAG QA (Assignment)

This project builds a **Retrieval-Augmented Generation (RAG)** Question Answering system that answers questions **only from the Swiggy Annual Report PDF**.

## Source PDF (required by assignment)
Download the latest Swiggy Annual Report PDF from Swiggy's official website:
- https://www.swiggy.com/about-us/

Place the PDF inside:
- `data/Swiggy_Annual_Report.pdf`

> Note: A sample PDF may already be provided to you by the recruiter/college. If so, just copy it into `data/Swiggy_Annual_Report.pdf`.

---

## Features
- PDF text extraction (PyPDF)
- Cleaning + chunking with metadata (page number)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: **Chroma** (persistent on disk)
- RAG: top-k retrieval + grounded answer generation
- Interfaces:
  - CLI: `python -m src.app`
  - Optional UI (Streamlit): `streamlit run src/ui_streamlit.py`

---

## 1) Setup (Windows / macOS / Linux)

### Step 1 — Create & activate virtual environment
**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add the PDF
Copy your PDF to:
```
data/Swiggy_Annual_Report.pdf
```

### Step 4 — Build the vector store (one-time)
```bash
python scripts/build_index.py --pdf data/Swiggy_Annual_Report.pdf
```

### Step 5 — Ask questions (CLI)
```bash
python -m src.app
```

### (Optional) Step 6 — Run the Streamlit UI
```bash
streamlit run src/ui_streamlit.py
```

---

## 2) Environment variables (optional, for better answers)

By default, the project uses a small local model (Flan-T5) for generation.
For **best quality**, you can use any OpenAI-compatible chat API.

Create a `.env` file in the project root:
```ini
LLM_PROVIDER=openai_compatible
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

If you don't set these, the system will still work using the local fallback generator.

---

## 3) How it prevents hallucination
1. It retrieves top relevant chunks from the vector DB.
2. The generator is forced to answer **ONLY using retrieved context**.
3. If the context is insufficient, it replies: **"Not found in the report."**

---

## 4) Project structure
```
swiggy_rag_assignment/
  data/
  scripts/
  src/
    app.py
    ui_streamlit.py
    rag/
      ingest.py
      index.py
      llm.py
      rag_pipeline.py
  requirements.txt
```

---

## 5) Evaluation tips
- Try factual questions (financial summary, business segments, key dates).
- Confirm that answers cite page numbers and don't invent details.

---

## 6) Assignment Requirement Coverage

1. **Document Processing**
  - PDF loading: `src/rag/ingest.py`
  - Cleaning + chunking + metadata: `src/rag/ingest.py`, `src/rag/index.py`

2. **Embedding & Vector Store**
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - Vector DB: Chroma (persistent) at `chroma_db/`
  - Similarity search: `similarity_search_with_score` in `src/rag/rag_pipeline.py`

3. **Retrieval-Augmented Generation (RAG)**
  - Retrieval + context construction: `src/rag/rag_pipeline.py`
  - LLM generation (OpenAI-compatible / local fallback): `src/rag/llm.py`
  - Grounding rule (`Not found in the report` when context is insufficient): enforced in pipeline/generator

4. **Question Answering Interface**
  - CLI: `python -m src.app`
  - Simple UI: `streamlit run src/ui_streamlit.py`
  - Output includes final answer (supporting context is optional per assignment)

---

## 7) Windows quick run commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH='.'
python scripts/build_index.py --pdf data/Swiggy_Annual_Report.pdf
python -m streamlit run src/ui_streamlit.py --server.port 8501
```

---

## License
For assignment/demo use.
