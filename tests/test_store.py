import json
from pathlib import Path

import pytest

from rag.store import load_chunks, to_documents

REQUIRED = ("chunk_id", "page_title", "heading_path", "url", "text", "n_tokens")


def write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def make_record(chunk_id: str = "chunk-00001") -> dict:
    return {
        "chunk_id": chunk_id,
        "page_title": "Tuition, Fees, & Aid",
        "heading_path": "Program Tuition",
        "url": "https://datascience.uchicago.edu/education/tuition-fees-aid/",
        "text": "Tuition is $6,384 per course.",
        "n_tokens": 12,
    }


def test_load_chunks_parses_every_line(tmp_path):
    path = write_jsonl(tmp_path, [make_record("chunk-1"), make_record("chunk-2")])
    chunks = load_chunks(path)
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "chunk-1"


def test_load_chunks_skips_blank_lines(tmp_path):
    path = write_jsonl(tmp_path, [make_record()])
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n")
    assert len(load_chunks(path)) == 1


def test_load_chunks_rejects_missing_field(tmp_path):
    bad = make_record()
    del bad["url"]
    path = write_jsonl(tmp_path, [bad])
    with pytest.raises(ValueError, match="url"):
        load_chunks(path)


def test_load_chunks_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "absent.jsonl")


def test_to_documents_preserves_metadata(tmp_path):
    docs = to_documents([make_record("chunk-42")])
    assert len(docs) == 1
    doc = docs[0]
    assert doc.page_content == "Tuition is $6,384 per course."
    assert doc.metadata["chunk_id"] == "chunk-42"
    assert doc.metadata["page_title"] == "Tuition, Fees, & Aid"
    assert doc.metadata["url"].endswith("/tuition-fees-aid/")
    assert doc.metadata["source"] == (
        "Tuition, Fees, & Aid "
        "(https://datascience.uchicago.edu/education/tuition-fees-aid/)"
    )


def test_real_corpus_loads():
    from rag.store import CHUNKS_PATH

    chunks = load_chunks(CHUNKS_PATH)
    assert len(chunks) > 400
    assert all(all(field in c for field in REQUIRED) for c in chunks)
