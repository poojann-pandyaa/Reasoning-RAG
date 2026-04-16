import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, dense_index_path="index/dense.faiss", bm25_index_path="index/bm25.pkl", meta_path="index/metadata.json"):
        print("Loading Embedding Model...")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        
        print(f"Loading FAISS from {dense_index_path}...")
        self.index = faiss.read_index(dense_index_path)
        
        print(f"Loading BM25 from {bm25_index_path}...")
        with open(bm25_index_path, "rb") as f:
            self.bm25 = pickle.load(f)
            
        print(f"Loading metadata from {meta_path}...")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
    def hybrid_retrieve(self, query, top_k=20):
        # 1. Dense Retrieval
        q_embedding = self.model.encode([query])
        faiss.normalize_L2(q_embedding)
        # Search FAISS
        dense_scores, dense_ids = self.index.search(q_embedding, top_k)
        
        # 2. Sparse Retrieval (BM25)
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_ids = sparse_scores.argsort()[-top_k:][::-1]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, idx in enumerate(dense_ids[0]):
            idx = int(idx)
            if idx == -1: # FAISS returns -1 if there are fewer than top_k items
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)
            
        for rank, idx in enumerate(sparse_ids):
            idx = int(idx)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)
            
        # 4. Sort by RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return list of dicts with chunk_id, metadata and fused score
        results = []
        for chunk_id, score in fused[:top_k]:
            meta = self.metadata[chunk_id]
            results.append({
                'chunk_id': chunk_id,
                'score': score,
                'metadata': meta
            })
            
        return results

if __name__ == "__main__":
    retriever = HybridRetriever()
    res = retriever.hybrid_retrieve("How to reverse a list in Python?")
    for r in res[:3]:
        print(r)
