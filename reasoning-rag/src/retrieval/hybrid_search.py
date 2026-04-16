import os
import json
import faiss
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # must match dense_index.py


class HybridRetriever:
    def __init__(
        self,
        dense_index_path = "index/dense.faiss",
        bm25_index_path  = "index/bm25.pkl",
        meta_path        = "index/metadata.json",
    ):
        print("Loading Embedding Model...")
        self.device    = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model     = AutoModel.from_pretrained(EMBED_MODEL).to(self.device)
        self.model.eval()

        print(f"Loading FAISS from {dense_index_path}...")
        self.index = faiss.read_index(dense_index_path)

        print(f"Loading BM25 from {bm25_index_path}...")
        with open(bm25_index_path, "rb") as f:
            self.bm25 = pickle.load(f)

        print(f"Loading metadata from {meta_path}...")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns float32 array shape (1, dim)."""
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded)
        emb = output.last_hidden_state[:, 0, :]       # CLS token
        emb = F.normalize(emb, p=2, dim=1)            # L2 normalise
        return emb.cpu().numpy().astype("float32")

    def hybrid_retrieve(self, query: str, top_k: int = 20):
        # 1. Dense retrieval
        q_embedding = self._embed(query)               # (1, 768)
        dense_scores, dense_ids = self.index.search(q_embedding, top_k)

        # 2. Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        sparse_scores   = self.bm25.get_scores(tokenized_query)
        sparse_ids      = sparse_scores.argsort()[-top_k:][::-1]

        # 3. Reciprocal Rank Fusion
        rrf_scores = {}
        for rank, idx in enumerate(dense_ids[0]):
            idx = int(idx)
            if idx == -1:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)
        for rank, idx in enumerate(sparse_ids):
            idx = int(idx)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

        # 4. Sort by RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, score in fused[:top_k]:
            meta = self.metadata[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "score":    score,
                "metadata": meta,
            })
        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    res = retriever.hybrid_retrieve("How to reverse a list in Python?")
    for r in res[:3]:
        print(r)
