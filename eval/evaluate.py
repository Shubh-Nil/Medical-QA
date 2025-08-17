
from __future__ import annotations
import json, re, argparse

def extract_citations(text: str):
    # naive [n] matcher
    return sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", text)))

def has_references_block(text: str):
    return "References" in text or "REFERENCES" in text.upper()

def grounding_score(text: str, retrieved_count: int) -> float:
    # fraction of cited ids that are within retrieved range
    cited = extract_citations(text)
    if not cited: return 0.0
    valid = [c for c in cited if 1 <= c <= retrieved_count]
    return len(valid) / len(cited)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, required=True, help="JSONL with fields: question, answer, retrieved (k)")
    args = ap.parse_args()

    n = 0
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            ex = json.loads(line)
            ans = ex["answer"]
            k = ex.get("retrieved_k", 5)
            gs = grounding_score(ans, k)
            print(f"Example {n}: citations={extract_citations(ans)} | has_refs={has_references_block(ans)} | grounding={gs:.2f}")

if __name__ == "__main__":
    main()
