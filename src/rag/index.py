from __future__ import annotations

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .ingest import load_pdf_pages, clean_text


def chunk_pdf(pdf_path: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[Document]:
    pages = load_pdf_pages(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: List[Document] = []
    for p in pages:
        text = clean_text(p.text)
        if not text:
            continue
        chunks = splitter.split_text(text)
        for j, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": p.page,
                        "chunk": j,
                        "source": pdf_path,
                    },
                )
            )
    return docs


def build_or_load_chroma(
    persist_dir: str,
    collection_name: str = "swiggy_annual_report",
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return vectordb


def build_index(pdf_path: str, persist_dir: str = "chroma_db", recreate: bool = False) -> int:
    vectordb = build_or_load_chroma(persist_dir=persist_dir)
    if recreate:
        # Drop and recreate collection
        try:
            vectordb._collection.delete(where={})
        except Exception:
            pass

    docs = chunk_pdf(pdf_path)
    if not docs:
        raise RuntimeError("No text extracted from PDF. Check if the PDF is scanned; use OCR if needed.")
    vectordb.add_documents(docs)
    vectordb.persist()
    return len(docs)
