
from __future__ import annotations
import argparse, json, os
from .embed_store import EmbedStore
from .ingest import ingest_guidelines, write_corpus_jsonl, DocChunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", type=str, default="data/sample_corpus.jsonl", help="Optional prebuilt corpus jsonl")
    ap.add_argument("--guidelines", type=str, default="examples/guidelines", help="Folder with PDFs/txt")
    ap.add_argument("--out", type=str, required=True, help="Output index dir")
    ap.add_argument("--emb-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    store = EmbedStore(args.emb_model)
    texts, metas = [], []

    if os.path.exists(args.in_jsonl):
        for line in open(args.in_jsonl, "r", encoding="utf-8"):
            js = json.loads(line)
            texts.append(js["text"]); metas.append(js.get("meta", {}))

    # also ingest local guideline files if present
    if os.path.isdir(args.guidelines):
        chunks = ingest_guidelines(args.guidelines)
        for ch in chunks:
            texts.append(ch.text); metas.append(ch.meta)

    store.add(texts, metas)
    store.save(args.out)
    print(f"Index saved to {args.out} with {len(texts)} chunks.")

if __name__ == "__main__":
    main()
