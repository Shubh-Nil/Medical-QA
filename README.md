# Medical Question Answering with RAG + QLoRA

**Season:** Spring 2025  
**Summary:** Retrieval-augmented generation (RAG) over PubMed + medical guidelines, fine-tuned with **QLoRA adapters** on **Llama‑2‑7B** to improve formatting and **citation correctness**. Produces grounded answers with numbered references.

> ⚠️ **Medical Safety Notice**: This repository is for research/education. It **does not** provide medical advice and may be inaccurate. Always consult qualified clinicians and official guidelines.

---

## Architecture

```
User Question
     │
     ▼
Retriever ──► PubMed (E-utilities) + Local Guidelines  ──► Chunk & Embed (FAISS)
     │                                                      ▲
     └──── top‑k contexts & metadata (PMID/DOI/URLs) ───────┘
                          │
                          ▼
           Generator (Llama‑2‑7B + QLoRA adapters)
                          │
                          ▼
        Answer + inline [1],[2]… citations + References
```

- **Retrieval**: PubMed E‑utilities + optional local guideline PDFs. Embeddings via Sentence-Transformers; FAISS vector index.
- **Generation**: HF Transformers with 4‑bit quantization, PEFT **QLoRA** adapters.
- **Evaluation**: Heuristics for **citation formatting** and **grounding** (do cited IDs appear among retrieved docs?).

---

## Quickstart

```bash
# 1) Environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) (Optional) PubMed API creds to increase limits
export PUBMED_EMAIL="you@example.com"
export PUBMED_API_KEY="<your-nih-key>"

# 3) Build index from sample data (or provide your own PDFs in examples/guidelines)
python src/rag/build_index.py --out data/index --emb-model sentence-transformers/all-MiniLM-L6-v2

# 4) Run the RAG demo (uses a small open chat model by default)
python examples/run_rag.py --question "What are first-line treatments for community-acquired pneumonia?"

# 5) (Optional) Fine-tune QLoRA adapters on your curated data
accelerate launch training/finetune_qlora.py --config training/configs/qlora.yaml
```

> **Models**: This repo references model names only. You must accept licenses and set `--base-model` accordingly (e.g., `meta-llama/Llama-2-7b-chat-hf`). The example script defaults to a permissive small model to run out-of-the-box.

---

## Repo Layout

```
MedQA-RAG-QLoRA/
├─ src/
│  └─ rag/
│     ├─ pubmed.py          # PubMed search/fetch via E-utilities
│     ├─ ingest.py          # Ingest PDFs / text; chunking utilities
│     ├─ embed_store.py     # SentenceTransformer + FAISS index
│     ├─ qa_pipeline.py     # End-to-end RAG + generation with citations
│     └─ build_index.py     # CLI: build vector index from local docs
├─ training/
│  ├─ finetune_qlora.py     # QLoRA SFT script
│  └─ configs/qlora.yaml    # Hyperparameters
├─ eval/
│  └─ evaluate.py           # Checks formatting and grounding of citations
├─ examples/
│  ├─ run_rag.py            # Minimal demo
│  └─ guidelines/           # Put local PDFs or .txt guidelines here
├─ tests/
│  ├─ test_retrieval.py     # Retrieval smoke tests
│  └─ test_eval.py          # Citation evaluation tests
├─ data/
│  ├─ sample_corpus.jsonl   # Tiny sample docs (abstracts/guideline snippets)
│  └─ index/                # (generated) FAISS index + metadata
├─ requirements.txt
├─ requirements-gpu.txt
├─ pyproject.toml
├─ LICENSE
└─ README.md
```

---

## Notes on QLoRA

- Loads base model in **4-bit** with `bitsandbytes`, then applies **LoRA** adapters with **PEFT**.
- Training targets formatting (structured answer + **References** block) and **citation alignment** (IDs in text must map to supplied references).

---

## Ethical Use

Do not deploy for clinical decision-making. If you work in healthcare, include human review, disclaimers, and strict guardrails.
