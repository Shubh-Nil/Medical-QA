
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional
import os, json, hashlib
from pypdf import PdfReader

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += (chunk_size - overlap)
        if (chunk_size - overlap) <= 0:
            break
    return chunks

@dataclass
class DocChunk:
    id: str
    text: str
    meta: Dict[str, Any]

def hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)

def ingest_guidelines(folder: str) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                text = load_pdf(os.path.join(root, fn))
            elif fn.lower().endswith((".txt",".md")):
                text = open(os.path.join(root, fn), "r", encoding="utf-8", errors="ignore").read()
            else:
                continue
            for idx, ch in enumerate(chunk_text(text)):
                cid = hash_id(f"{fn}:{idx}:{ch[:50]}")
                chunks.append(DocChunk(id=cid, text=ch, meta={"source": fn, "type": "guideline"}))
    return chunks

def write_corpus_jsonl(chunks: List[DocChunk], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps({"id": ch.id, "text": ch.text, "meta": ch.meta}, ensure_ascii=False) + "\n")
