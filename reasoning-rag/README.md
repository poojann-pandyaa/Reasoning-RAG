# Reasoning-RAG

A Retrieval-Augmented Generation (RAG) system built on Stack Exchange data. Given a natural language question, it retrieves relevant Stack Overflow answers via hybrid search and generates a grounded, structured response using **Gemma-2-2B-IT** — all running **100% locally** on Apple Silicon via MLX.

---

## Architecture

```
Stack Exchange Data (50k questions)
        ↓  preprocess.py
processed_dataset.jsonl          ✅  86,712 high-quality chunks
        ↓  dense_index.py  (bge-base-en-v1.5, 768-dim, MPS)
dense.faiss                      ✅  FAISS flat inner-product index
        ↓  sparse_index.py  (BM25)
bm25.pkl                         ✅  Rank-BM25 sparse index
        ↓  hybrid_search.py  (RRF fusion + cross-encoder reranker)
Top-3 reranked sources           ✅  ms-marco-MiniLM-L-6-v2
        ↓  QueryClassifier  (flan-t5-base, MPS)
Reasoning type: commonsense / adaptive / strategic
        ↓  FinalGenerator  (Gemma-2-2B-IT, MLX)
Grounded answer                  ✅
        ↓  LoRA fine-tuning  (train_mlx.py)
Domain-adapted adapter           ✅  iter 1600 best checkpoint
```

---

## Features

- **Hybrid retrieval** — dense (FAISS cosine) + sparse (BM25) fused via Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` reranks top-10 candidates to top-3
- **Reasoning-aware generation** — query is classified into `commonsense`, `adaptive`, or `strategic` reasoning paths before generation
- **Self-consistency decoding** — strategic queries run 3 generations and select the highest-diversity answer
- **LoRA fine-tuning** — Gemma-2-2B-IT fine-tuned on 49,781 Stack Overflow Q&A pairs via MLX
- **100% local** — no API keys, no cloud, runs on M1/M2/M3/M4 Mac

---

## Project Structure

```
reasoning-rag/
├── src/
│   ├── demo.py                    # Interactive CLI demo
│   ├── train_mlx.py               # LoRA fine-tuning (MLX)
│   ├── generation/
│   │   └── generator.py           # FinalGenerator (MLX + PyTorch fallback)
│   ├── retrieval/
│   │   ├── dense_index.py         # FAISS index builder
│   │   ├── sparse_index.py        # BM25 index builder
│   │   └── hybrid_search.py       # RRF fusion + reranker
│   ├── reasoning/                 # Query classification & reasoning paths
│   ├── ingestion/
│   │   └── preprocess.py          # Stack Exchange → JSONL chunks
│   └── evaluation/
│       └── compare_demo.py        # Base vs fine-tuned comparison
├── index/
│   ├── dense.faiss                # 86,712-vector FAISS index
│   ├── bm25.pkl                   # BM25 index
│   └── metadata.json             # Chunk metadata
├── data/
│   └── processed_dataset.jsonl   # Preprocessed chunks
├── outputs/
│   └── gemma-2-2b-it-mlx-lora-v2/
│       └── 0001600_adapters.safetensors  # Best LoRA checkpoint
├── assets/
│   └── val_loss_lora.png          # Training curve
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
cd reasoning-rag
pip install -r requirements.txt
```

### 2. Build the indices (or use pre-built)

```bash
# Preprocess Stack Exchange data
python3 src/ingestion/preprocess.py

# Build dense index (bge-base-en-v1.5, MPS accelerated)
python3 src/retrieval/dense_index.py

# Build sparse index (BM25)
python3 src/retrieval/sparse_index.py
```

### 3. Run the demo

```bash
# Base model
python3 src/demo.py

# With LoRA adapter (best checkpoint)
python3 src/demo.py --adapter outputs/gemma-2-2b-it-mlx-lora-v2/0001600_adapters.safetensors
```

### 4. Run base vs fine-tuned comparison

```bash
python3 src/evaluation/compare_demo.py \
    --adapter outputs/gemma-2-2b-it-mlx-lora-v2/0001600_adapters.safetensors
```

---

## LoRA Fine-tuning

Gemma-2-2B-IT was fine-tuned on **49,781 Stack Overflow Q&A pairs** using MLX LoRA.

```bash
# Full training run (overnight, ~2000 iterations)
python3 src/train_mlx.py --skip-data-prep

# Resume from checkpoint
python3 src/train_mlx.py --skip-data-prep --resume-adapter-file outputs/.../0001600_adapters.safetensors
```

### Training Curve

![LoRA Validation Loss](assets/val_loss_lora.png)

| Checkpoint | Val Loss | Note |
|---|---|---|
| iter 1 (baseline) | 3.357 | Untrained |
| iter 400 | 1.945 | Second best |
| **iter 1600** | **1.693** | ✅ Best checkpoint |
| iter 2000 | 2.199 | Final saved |

**Total loss drop: −1.664 (49.6% reduction from baseline)**

The oscillation between checkpoints is expected for batch_size=1 with variable sequence lengths. The model did not collapse — loss never exceeded the baseline.

---

## Evaluation Results

Base model vs fine-tuned (iter 1600 adapter), 5 Stack Overflow queries:

| Query | Base | Fine-tuned | Winner |
|---|---|---|---|
| Reverse a list in Python | Terse lambda, no code block | Full function + comments + markdown | ✅ FT |
| `==` vs `is` in Python | No answer (retrieval miss) | No answer (retrieval miss) | Tie |
| Segfault in C | Verbose, wordy | Clean, direct diagnosis | ✅ FT |
| `git rebase` | Leaks raw SO URL into answer | Paraphrases cleanly, no URL | ✅ FT |
| `async/await` in Python | No answer (retrieval miss) | No answer (retrieval miss) | Tie |

**Fine-tuned wins 3/5 queries.** The 2 ties are retrieval misses (topics not covered in the index), not generation failures.

| Metric | Base | Fine-tuned |
|---|---|---|
| Avg answer length (words) | 33 | 36 |
| Avg latency (s) | 79.0 | 91.7 |

---

## Bugs Fixed During Development

| # | Problem | Fix |
|---|---|---|
| 1 | `dense_index.py` taking 2+ hours | Reduced chunks 199k → 86k via `MIN_SCORE=3`, `MAX_ANSWERS=3` |
| 2 | MPS OOM at batch=512 | Dropped embedding batch size to 32 |
| 3 | Segfault on macOS Python 3.9 | Replaced `SentenceTransformer.encode()` with raw `AutoModel` + `torch.no_grad()` — bypasses multiprocessing bug |
| 4 | `AssertionError` in FAISS search | `hybrid_search.py` used bge-large (1024-dim) but index had bge-base (768-dim) — aligned both to bge-base |
| 5 | `Q:` fields blank in all chunks | `preprocess.py` had `title: ''` hardcoded — fixed to use first 120 chars of the question body |
| 6 | `train_mlx.py` CLI error | `--lora-layers` renamed to `--num-layers` in newer `mlx_lm`; also fixed subcommand syntax |
| 7 | One-sentence generation quality regression | `commonsense` prompt said "concisely" + Rule 3 blocked elaboration — rewrote prompt to encourage thorough answers with code blocks |

---

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- 16GB RAM minimum (24GB+ recommended for fine-tuning)

Key packages: `mlx`, `mlx-lm`, `faiss-cpu`, `rank-bm25`, `sentence-transformers`, `transformers`, `torch`

See `requirements.txt` for full pinned versions.

---

## Models Used

| Role | Model |
|---|---|---|
| Generator | `google/gemma-2-2b-it` (MLX) |
| Dense embeddings | `BAAI/bge-base-en-v1.5` (768-dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Query classifier | `google/flan-t5-base` |

---

## Data

Stack Exchange data dump (Computer Science / Stack Overflow subset) — 50,000 questions, filtered to 86,712 high-quality answer chunks (score ≥ 3, max 3 answers per question).
