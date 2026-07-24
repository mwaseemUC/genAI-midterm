"""Corpus loading and vector store construction.

The store is built in memory at startup from data/chunks.jsonl. Container
filesystems are ephemeral, so persistence buys nothing, and an in-memory
store avoids the Windows file-lock problem the notebook had to work around.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks.jsonl"

REQUIRED_FIELDS = ("chunk_id", "page_title", "heading_path", "url", "text", "n_tokens")


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """Read chunks.jsonl, validating that every record carries the fields
    the retrieval and citation code depends on."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"{path.name} line {lineno} is missing fields: {', '.join(missing)}"
                )
            chunks.append(record)

    if not chunks:
        raise ValueError(f"{path.name} contains no chunks")
    return chunks


def to_documents(chunks: list[dict]) -> list[Document]:
    """Convert chunk records into LangChain Documents.

    Metadata mirrors the notebook exactly so citations render identically.
    """
    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "page_title": chunk["page_title"],
                "heading_path": chunk["heading_path"],
                "url": chunk["url"],
                "n_tokens": chunk["n_tokens"],
                "source": f"{chunk['page_title']} ({chunk['url']})",
            },
        )
        for chunk in chunks
    ]


def build_store(chunks: list[dict]) -> Chroma:
    """Embed the corpus into an in-memory Chroma store.

    No persist_directory: the store lives for the process's lifetime and is
    held by Streamlit's cache_resource.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, max_retries=5)
    return Chroma.from_documents(documents=to_documents(chunks), embedding=embeddings)
