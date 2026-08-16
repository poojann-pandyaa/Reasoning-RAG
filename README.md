# Reasoning-Augmented RAG (Reasoning-RAG)

> **Contextual, Adaptive & Strategic Reasoning over Retrieval-Augmented Generation**  
> A production-grade, 100% local RAG system built on Stack Exchange data, featuring hybrid retrieval, query classification, cross-encoder reranking, and LoRA fine-tuning on Apple Silicon via MLX.

---

## 📂 Project Navigation

* **Complete Project Code & Models:** [`reasoning-rag/`](file:///Users/poojan/Downloads/Reasoning-RAG-main/reasoning-rag)
* **Comprehensive Architecture Documentation & Benchmarks:** [`reasoning-rag/README.md`](file:///Users/poojan/Downloads/Reasoning-RAG-main/reasoning-rag/README.md)
* **System Protocol Specification:** [`README_EndToEnd_Protocol.md`](file:///Users/poojan/Downloads/Reasoning-RAG-main/README_EndToEnd_Protocol.md)
* **Ordered Implementation Steps:** [`README_OrderedSteps.md`](file:///Users/poojan/Downloads/Reasoning-RAG-main/README_OrderedSteps.md)

---

## 🏛 System Architecture Overview

```mermaid
flowchart TD
    %% Global Styling
    classDef index fill:#ede7f6,stroke:#5e35b1,stroke-width:2px,color:#311b92;
    classDef router fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef retrieval fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;
    classDef generator fill:#fffde7,stroke:#fbc02d,stroke-width:2px,color:#f57f17;

    %% STEP 1: PREPARATION
    subgraph S1 ["1. OFFLINE DATA & INDEX PREPARATION"]
        RAW["Stack Exchange Stream"] --> DEDUP["Deduplication & Quality Filter<br/>(Score ≥ 5 OR Accepted)"]
        DEDUP --> CHUNK["Semantic Chunking<br/>(Score ≥ 3, Max 3/Q, 256 tokens)"]
        CHUNK --> FAISS_DB[("dense.faiss<br/>768-dim BGE Vectors")]
        CHUNK --> BM25_DB[("bm25.pkl<br/>BM25 Inverted Index")]
    end
    class RAW,DEDUP,CHUNK,FAISS_DB,BM25_DB index;

    %% STEP 2: QUERY UNDERSTANDING
    subgraph S2 ["2. ONLINE QUERY ROUTING"]
        USER_Q(["User Query"]) --> ROUTER["Flan-T5 + Asymmetric Safety Net"]
        ROUTER --> COMMONSENSE["Commonsense (1 Linear Pass)"]
        ROUTER --> ADAPTIVE["Adaptive (Parallel Sub-Qs)"]
        ROUTER --> STRATEGIC["Strategic (Two-Level Tree)"]
    end
    class USER_Q,ROUTER,COMMONSENSE,ADAPTIVE,STRATEGIC router;

    %% STEP 3: SEARCH & RANK
    subgraph S3 ["3. HYBRID SEARCH & CONTEXT RERANKING"]
        COMMONSENSE & ADAPTIVE & STRATEGIC --> RETRIEVE["Hybrid Fetcher (FAISS + BM25)"]
        RETRIEVE --> RRF["Reciprocal Rank Fusion (k = 60)"]
        RRF --> RERANK["Cross-Encoder (ms-marco-MiniLM)"]
        RERANK --> PRIOR["Human Preference Prior<br/>(+0.10*Score/100 + 0.15*Accepted)"]
    end
    class RETRIEVE,RRF,RERANK,PRIOR retrieval;

    %% STEP 4: GENERATION
    subgraph S4 ["4. GENERATION & LoRA ADAPTATION"]
        PRIOR --> PROMPT["Dynamic CoT Prompt + 4 Guardrails"]
        PROMPT --> LLM["Gemma-2-2B-IT + MLX LoRA Adapter<br/>(Iteration 1600: Loss = 1.693)"]
        LLM --> OUT(["Structured Grounded Answer"])
    end
    class PROMPT,LLM,OUT generator;
```

---

## 🚀 Quickstart

```bash
cd reasoning-rag
pip install -r requirements.txt

# Run the interactive Reasoning CLI demo
python3 src/demo.py --adapter outputs/gemma-2-2b-it-mlx-lora-v2/0001600_adapters.safetensors

# Run the head-to-head comparison harness (Base vs. Fine-Tuned)
python3 src/evaluation/compare_demo.py --adapter outputs/gemma-2-2b-it-mlx-lora-v2/0001600_adapters.safetensors
```

For the complete technical breakdown, mathematical formulas, evaluation metrics, and design trade-offs, see the full [reasoning-rag/README.md](file:///Users/poojan/Downloads/Reasoning-RAG-main/reasoning-rag/README.md).
