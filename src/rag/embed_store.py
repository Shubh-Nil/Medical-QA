
from __future__ import annotations
import os, json, faiss, numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

@dataclass
class Retrieved:
    text: str
    score: float
    meta: Dict[str, Any]

class EmbedStore:
    def __init__(self, emb_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.emb_model_name = emb_model
        self.model = SentenceTransformer(emb_model)
        self.index = None
        self.texts: list[str] = []
        self.metas: list[dict] = []

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=64, normalize_embeddings=True)
        self.texts.extend(texts)
        self.metas.extend(metas)
        if self.index is None:
            d = embs.shape[1]
            self.index = faiss.IndexFlatIP(d)
        self.index.add(embs.astype(np.float32))

    def search(self, query: str, k: int = 5) -> List[Retrieved]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q.astype(np.float32), k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1: continue
            out.append(Retrieved(text=self.texts[idx], score=float(score), meta=self.metas[idx]))
        return out

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "store.json"), "w", encoding="utf-8") as f:
            json.dump({
                "emb_model": self.emb_model_name,
                "texts": self.texts,
                "metas": self.metas,
            }, f)

    @classmethod
    def load(cls, path: str) -> "EmbedStore":
        with open(os.path.join(path, "store.json"), "r", encoding="utf-8") as f:
            js = json.load(f)
        store = cls(js["emb_model"])
        store.texts = js["texts"]
        store.metas = js["metas"]
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        return store
