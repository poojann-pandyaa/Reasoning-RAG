import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Model options (comment/uncomment) ─────────────────────────────────────
# EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # 1024-dim, ~12h for 300k chunks on M4
EMBED_MODEL   = "BAAI/bge-base-en-v1.5"    # 768-dim,  ~45 min  ← DEFAULT
# EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim,  ~20 min  (slightly lower quality)
# ──────────────────────────────────────────────────────────────────────────

BATCH_SIZE    = 256          # larger batches = faster on MPS
MAX_CHUNK_LEN = 512          # truncate to keep embeddings fast & RAM stable
MAX_ANSWERS   = 5            # max answers per question (skip low-quality tail)


def create_dense_index(
    data_path  = "data/processed_dataset.jsonl",
    index_path = "index/dense.faiss",
    meta_path  = "index/metadata.json",
):
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading embedding model: {EMBED_MODEL}  (device={device})...")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    chunks   = []
    metadata = []

    print("Reading dataset & chunking...")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Chunking"):
            record  = json.loads(line)
            title   = record.get("title", "")
            domain  = record.get("domain", "")
            q_id    = record.get("question_id", "")
            answers = record.get("answers", [])

            # Sort by score desc, keep top MAX_ANSWERS
            answers = sorted(
                answers,
                key=lambda a: (a.get("is_accepted", False), a.get("score", 0)),
                reverse=True,
            )[:MAX_ANSWERS]

            for ans in answers:
                ans_body    = ans.get("body_clean", "")
                score       = ans.get("score", ans.get("pm_score", 0))
                is_accepted = ans.get("is_accepted", False)

                chunk_text  = f"Q: {title}\nA: {ans_body}"
                # Truncate at character level before embedding
                chunk_text  = chunk_text[:MAX_CHUNK_LEN * 4]  # ~4 chars/token

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
    print(f"Total chunks: {total}")

    if total == 0:
        print("No chunks found. Exiting.")
        return

    # ── Embed in batches, stream into FAISS to keep RAM flat ──────────────
    print(f"Embedding {total} chunks (batch={BATCH_SIZE})...")

    dim   = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)   # cosine after L2 normalisation

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

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    print(f"Saving FAISS index -> {index_path}")
    faiss.write_index(index, index_path)

    print(f"Saving metadata   -> {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print(f"Dense indexing complete. {index.ntotal} vectors stored.")


if __name__ == "__main__":
    create_dense_index()
