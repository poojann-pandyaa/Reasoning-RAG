import os
import json
import faiss
import numpy as np
from tqdm import tqdm

EMBED_MODEL   = "BAAI/bge-base-en-v1.5"
BATCH_SIZE    = 32     # safe for 16GB MPS -- no OOM, no segfault
MAX_CHUNK_LEN = 1024  # chars
MIN_SCORE     = 3
MAX_ANSWERS   = 3


def embed_batch_torch(model, tokenizer, texts, device):
    """
    Encode a list of strings directly with torch -- no sentence_transformers
    multiprocessing that segfaults on macOS Python 3.9.
    Returns float32 numpy array of shape (len(texts), hidden_dim).
    """
    import torch
    import torch.nn.functional as F

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    # CLS-token embedding (index 0) -- standard for BGE
    embeddings = output.last_hidden_state[:, 0, :]
    # L2 normalise for cosine similarity via inner product
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype("float32")


def create_dense_index(
    data_path  = "data/processed_dataset.jsonl",
    index_path = "index/dense.faiss",
    meta_path  = "index/metadata.json",
):
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[dense_index] model={EMBED_MODEL}  device={device}  batch={BATCH_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model     = AutoModel.from_pretrained(EMBED_MODEL).to(device)
    model.eval()

    # -- Chunking --
    chunks   = []
    metadata = []

    print("Reading & chunking dataset...")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Chunking"):
            record  = json.loads(line)
            title   = record.get("title", "")[:200]
            domain  = record.get("domain", "")
            q_id    = record.get("question_id", "")
            answers = record.get("answers", [])

            good = [
                a for a in answers
                if a.get("is_accepted") or a.get("score", 0) >= MIN_SCORE
            ]
            if not good:
                good = sorted(answers, key=lambda a: a.get("score", 0), reverse=True)[:1]

            good = sorted(
                good,
                key=lambda a: (a.get("is_accepted", False), a.get("score", 0)),
                reverse=True,
            )[:MAX_ANSWERS]

            for ans in good:
                body       = ans.get("body_clean", "")[:MAX_CHUNK_LEN]
                chunk_text = f"Q: {title}\nA: {body}"[:MAX_CHUNK_LEN]
                chunks.append(chunk_text)
                metadata.append({
                    "chunk_id":    len(chunks) - 1,
                    "question_id": q_id,
                    "score":       ans.get("score", ans.get("pm_score", 0)),
                    "is_accepted": ans.get("is_accepted", False),
                    "domain":      domain,
                    "chunk_text":  chunk_text,
                })

    total = len(chunks)
    print(f"Total chunks: {total}")
    if total == 0:
        print("No chunks. Exiting."); return

    # -- Embed in batches, stream into FAISS --
    dim   = model.config.hidden_size
    index = faiss.IndexFlatIP(dim)

    print(f"Embedding {total} chunks on {device} (batch={BATCH_SIZE})...")
    for start in tqdm(range(0, total, BATCH_SIZE), desc="Embedding"):
        batch = chunks[start : start + BATCH_SIZE]
        vecs  = embed_batch_torch(model, tokenizer, batch, device)
        index.add(vecs)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    print(f"Saving FAISS index -> {index_path}  ({index.ntotal} vectors)")
    faiss.write_index(index, index_path)

    print(f"Saving metadata   -> {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print("Dense indexing complete.")


if __name__ == "__main__":
    create_dense_index()
