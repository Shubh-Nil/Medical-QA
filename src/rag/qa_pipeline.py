
from __future__ import annotations
import os, textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .embed_store import EmbedStore, Retrieved
from .pubmed import PubMedClient
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

PROMPT_TEMPLATE = """You are a careful medical assistant. Use the provided CONTEXT to answer the QUESTION.
- Be concise and clinically cautious.
- Include inline numeric citations like [1], [2] where you use evidence.
- End with a "References" section listing each cited item as [n] Title. Source (Year). URL/DOI/PMID.
- If you are unsure, say you are unsure and suggest consulting a clinician.

QUESTION:
{question}

CONTEXT:
{context}

Answer:
"""

def format_context(blocks: List[Retrieved]) -> str:
    lines = []
    for i, b in enumerate(blocks, start=1):
        src = b.meta.get("source") or b.meta.get("url") or b.meta.get("pmid") or "unknown"
        lines.append(f"[{i}] {b.text}\n(Source: {src})")
    return "\n\n".join(lines)

@dataclass
class QAResult:
    answer: str
    references: List[Dict[str, Any]]
    retrieved: List[Retrieved]

class QAPipeline:
    def __init__(self, index_path: str, base_model: str, device: str = "auto"):
        self.store = EmbedStore.load(index_path)
        self.model_name = base_model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")

    def retrieve(self, question: str, extra_pubmed_query: Optional[str] = None, k: int = 5) -> List[Retrieved]:
        retrieved = self.store.search(question, k=k)
        # optionally augment with PubMed live results
        if extra_pubmed_query:
            pmc = PubMedClient()
            pmids = pmc.search(extra_pubmed_query, retmax=5)
            recs = pmc.fetch(pmids)
            for r in recs:
                retrieved.append(Retrieved(text=(r.abstract or r.title), score=0.0, meta={"pmid": r.pmid, "title": r.title, "url": r.url, "type": "pubmed"}))
        return retrieved[:k]

    def generate(self, question: str, retrieved: List[Retrieved], max_new_tokens: int = 512, temperature: float = 0.1) -> QAResult:
        context = format_context(retrieved)
        prompt = PROMPT_TEMPLATE.format(question=question, context=context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=temperature>0.0, eos_token_id=self.tokenizer.eos_token_id)
        ans = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # naive reference block extraction (post-gen)
        refs = []
        for i, b in enumerate(retrieved, start=1):
            meta = b.meta.copy()
            meta["id"] = i
            refs.append(meta)
        return QAResult(answer=ans, references=refs, retrieved=retrieved)
