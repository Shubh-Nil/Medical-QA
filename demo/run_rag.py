
import argparse, os, json
from rag.qa_pipeline import QAPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--index", type=str, default="data/index")
    ap.add_argument("--base-model", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    ap.add_argument("--pubmed", type=str, default=None, help="Optional extra PubMed query (e.g., add disease OR guideline terms)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    qa = QAPipeline(index_path=args.index, base_model=args.base-model if False else args.base_model)  # avoid hyphen issue
    retrieved = qa.retrieve(args.question, extra_pubmed_query=args.pubmed, k=args.k)
    result = qa.generate(args.question, retrieved)

    print("\n=== Answer ===\n")
    print(result.answer)
    print("\n=== References (retrieved metadata) ===\n")
    for r in result.references:
        print(r)

if __name__ == "__main__":
    main()
