import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -- Model options --
# EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # 1024-dim, very slow on M4
EMBED_MODEL   = "BAAI/bge-base-en-v1.5"   # 768-dim  <- DEFAULT
# EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, fastest

BATCH_SIZE    = 512   # push MPS harder
MAX_CHUNK_LEN = 256   # chars (~64 tokens) -- enough for retrieval
MIN_SCORE     = 3     # skip low-quality answers
MAX_ANSWERS   = 3     # top-3 answers per question only


def create_dense_index(
    data_path  = "data/processed_dataset.jsonl",
    index_path = "index/dense.faiss",
    meta_path  = "index/metadata.json",
):
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[dense_index] model={EMBED_MODEL}  device={device}  batch={BATCH_SIZE}")
    model = SentenceTransformer(EMBED_MODEL, device=device)

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

            # Keep accepted answers + those with score >= MIN_SCORE, top MAX_ANSWERS
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
                body        = ans.get("body_clean", "")[:MAX_CHUNK_LEN * 4]
                score       = ans.get("score", ans.get("pm_score", 0))
                is_accepted = ans.get("is_accepted", False)
                chunk_text  = f"Q: {title}\nA: {body}"[:MAX_CHUNK_LEN * 4]

                chunks.append(chunk_text)
                metadata.append({
                    "chunk_id":    len(chunks) - 1,
                    "question_id": q_id,
                    "score":       score,
                    "is_accepted": is_accepted,
                    "domain":      domain,
                    "chunk_text":  chunk_text,
                })

    total = len(chunks)
    print(f"Total chunks to embed: {total}")
    if total == 0:
        print("No chunks found. Exiting."); return

    # Stream-embed into FAISS -- keeps RAM flat
    dim   = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    print(f"Embedding {total} chunks...")
    for start in tqdm(range(0, total, BATCH_SIZE), desc="Embedding"):
        batch = chunks[start : start + BATCH_SIZE]
        vecs  = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        index.add(vecs)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    print(f"Saving FAISS index  -> {index_path}  ({index.ntotal} vectors)")
    faiss.write_index(index, index_path)

    print(f"Saving metadata     -> {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print("Dense indexing complete.")


if __name__ == "__main__":
    create_dense_index()
