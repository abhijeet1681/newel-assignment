# Swiggy Annual Report RAG QA

## Dataset / Document
- Document Name: Swiggy Annual Report (Latest available)
- Format: PDF
- Source: Publicly available Swiggy Annual Report
- Source link: https://www.swiggy.com/about-us/

Place the file at:
- `data/Swiggy_Annual_Report.pdf`

## Problem Statement
Create a RAG-based Question Answering system that allows users to ask natural language questions related to the Swiggy Annual Report and receive accurate, context-grounded answers.

The system should answer strictly based on document content.

## Functional Requirements
1. Document Processing
   - Load PDF
   - Split into meaningful chunks
   - Text preprocessing (cleaning, chunking, metadata)

2. Embedding & Vector Store
   - Generate embeddings
   - Store embeddings in vector DB
   - Support semantic similarity search

3. Retrieval-Augmented Generation (RAG)
   - Retrieve relevant chunks for query
   - Pass context to LLM
   - Generate answer from retrieved context

4. Question Answering Interface
   - Accept user query (CLI or simple UI)
   - Display final answer
   - Supporting context is optional

## Run (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH='.'
python scripts/build_index.py --pdf data/Swiggy_Annual_Report.pdf
python -m streamlit run src/ui_streamlit.py --server.port 8501
```
