import torch
from sentence_transformers import SentenceTransformer
import chromadb

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class search_result:
    def __init__(self, text:str, meta:dict, score:float):
        self.text = text
        self.meta = meta
        self.score = score

    def format(self, index) -> str:
        reference = f"[{index}] {self.text} \nPubMed ID: {self.meta.get('pubid', 'unknown')} \n\n"
        return reference


class vector_database:
    def __init__(self, model_name: str, database_path: str):
        self.model = SentenceTransformer(model_name).to(device)
        self.database = chromadb.PersistentClient(path=database_path).get_or_create_collection(name="PubMedQA")

    def add(self, chunks: list[dict]):
        texts = [chunk['text'] for chunk in chunks]
        metas = [chunk['meta'] for chunk in chunks]

        text_ids = [str(i + self.database.count()) for i in range(len(texts))]
        text_embs = self.model.encode(texts,
                                      batch_size=64,
                                      convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=True).tolist()
        self.database.add(ids=text_ids,
                          documents=texts,
                          embeddings=text_embs,
                          metadatas=metas)

    def search(self, query: str, k: int = 5, filters = None) -> list[search_result]:
        query_emb = self.model.encode([query], 
                                      convert_to_numpy=True, 
                                      normalize_embeddings=True).tolist()
        results = self.database.query(query_embeddings=query_emb,
                                      n_results=k,
                                      where=filters)
        
        retrieve = []
        for text, meta, score in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            retrieve.append(search_result(text, meta, score))
        return retrieve
