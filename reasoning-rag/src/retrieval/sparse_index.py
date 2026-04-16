import json
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def create_sparse_index(meta_path="index/metadata.json", index_path="index/bm25.pkl"):
    print(f"Loading metadata from {meta_path} to get chunks for BM25...")
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Metadata file not found! Run dense_index.py first so metadata is created.")
        return
        
    print("Tokenizing corpus...")
    # chunk_text contains "Q: {title}\nA: {ans_body}"
    tokenized_corpus = [meta['chunk_text'].lower().split() for meta in tqdm(metadata, desc="Tokenizing")]
    
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"Saving BM25 index to {index_path}...")
    with open(index_path, "wb") as f:
        pickle.dump(bm25, f)
        
    print("Sparse Indexing Complete.")

if __name__ == "__main__":
    create_sparse_index()
