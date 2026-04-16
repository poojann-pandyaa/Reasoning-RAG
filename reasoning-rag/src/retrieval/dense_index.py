import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def create_dense_index(data_path="data/processed_dataset.jsonl", index_path="index/dense.faiss", meta_path="index/metadata.json"):
    print("Loading embedding model...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    chunks = []
    metadata = []
    
    print("Reading dataset & chunking...")
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f)):
            record = json.loads(line)
            title = record.get('title', '')
            domain = record.get('domain', '')
            q_id = record.get('question_id', '')
            
            for ans in record.get('answers', []):
                ans_body = ans.get('body_clean', '')
                score = ans.get('score', ans.get('pm_score', 0))
                is_accepted = ans.get('is_accepted', False)
                
                # Semantic Chunking + Prepending title as per instructions
                chunk_text = f"Q: {title}\nA: {ans_body}"
                chunks.append(chunk_text)
                metadata.append({
                    'chunk_id': len(chunks) - 1,
                    'question_id': q_id,
                    'score': score,
                    'is_accepted': is_accepted,
                    'domain': domain,
                    'chunk_text': chunk_text
                })

    print(f"Total chunks created: {len(chunks)}")
    
    if len(chunks) == 0:
        print("No chunks to embed! Exiting.")
        return

    # Embed in batches
    print("Embedding chunks...")
    batch_size = 64
    
    # We will compute embeddings one batch at a time and add them to FAISS
    # to avoid blowing up RAM for huge datasets, though model.encode already handles batches.
    embeddings = model.encode(chunks, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # Inner product = cosine after normalisation
    index.add(embeddings)
    
    print(f"Writing index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Writing metadata to {meta_path}...")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)
        
    print("Dense Indexing Complete.")

if __name__ == "__main__":
    create_dense_index()
