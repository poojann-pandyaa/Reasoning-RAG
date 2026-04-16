# 🧠 Reasoning-Augmented Retrieval-Augmented Generation (Reasoning RAG)

> A fully offline, end-to-end Reasoning-Augmented RAG pipeline for software engineering Q\&A — powered by hybrid retrieval (FAISS + BM25), cross-encoder re-ranking, multi-path reasoning, and a local instruction-tuned LLM.

---

## 📋 Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Proposed Solution](#2-proposed-solution)
- [3. System Architecture](#3-system-architecture)
- [4. Dataset](#4-dataset)
- [5. Project Structure](#5-project-structure)
- [6. Module Deep Dive](#6-module-deep-dive)
  - [6.1 Data Ingestion & Preprocessing](#61-data-ingestion--preprocessing)
  - [6.2 Dense Indexing (FAISS)](#62-dense-indexing-faiss)
  - [6.3 Sparse Indexing (BM25)](#63-sparse-indexing-bm25)
  - [6.4 Hybrid Retrieval & Reciprocal Rank Fusion](#64-hybrid-retrieval--reciprocal-rank-fusion)
  - [6.5 Cross-Encoder Re-Ranking](#65-cross-encoder-re-ranking)
  - [6.6 Query Classification (Reasoning Layer)](#66-query-classification-reasoning-layer)
  - [6.7 Multi-Path Reasoning Engine](#67-multi-path-reasoning-engine)
  - [6.8 Grounded Generation](#68-grounded-generation)
  - [6.9 Evaluation](#69-evaluation)
- [7. Setup & Installation](#7-setup--installation)
- [8. Usage](#8-usage)
- [9. Example Queries](#9-example-queries)
- [10. Technologies Used](#10-technologies-used)
- [11. Limitations & Future Work](#11-limitations--future-work)

---

## 1. Problem Statement

Standard Retrieval-Augmented Generation (RAG) systems are fundamentally **"dumb retrievers"**. When a user asks a question:

1. They blindly embed the query and fetch the top-K most similar chunks from a vector database.
2. They dump all the retrieved text into a prompt with no strategic reasoning about what information is actually needed.
3. They hope the language model can figure out the answer from potentially irrelevant, duplicated, or poorly ranked sources.

This approach completely breaks down when:

- Queries are **complex or multi-part** (e.g., *"Compare Redis vs Memcached for session storage and explain tradeoffs"*).
- Queries are **ambiguous** and require clarification or decomposition.
- The knowledge base contains answers of **varying quality** (e.g., upvoted vs. downvoted StackOverflow answers).
- **Exact keyword matching** is important (e.g., searching for error code `0xDEADBEEF` — purely semantic embeddings will miss this).

**The result:** Standard RAG pipelines produce hallucinated, irrelevant, or shallow answers for any non-trivial technical query.

---

## 2. Proposed Solution

This project implements a **Reasoning-Augmented RAG** system that makes _intelligent reasoning_ a first-class citizen at every stage of the pipeline. Inspired by the dual-process theory of cognition (System 1 / System 2 thinking), the system:

1. **Classifies** the incoming query to determine intent, complexity, and the appropriate reasoning strategy _before_ retrieving anything.
2. **Decomposes** complex queries into sub-questions that can each be independently retrieved and answered.
3. **Retrieves** using a hybrid approach combining dense semantic search (FAISS) with sparse keyword matching (BM25), fused via Reciprocal Rank Fusion (RRF).
4. **Re-ranks** candidates using a Cross-Encoder model that also incorporates community preference signals (upvotes, accepted-answer status) from StackOverflow.
5. **Generates** answers grounded in the retrieved evidence using chain-of-thought prompting, with self-consistency decoding for high-stakes queries.
6. **Operates 100% offline** — no API keys, no internet, no cloud dependencies. Everything runs locally using `google/flan-t5-base`.

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                     │
│                   "How to convert decimal to double in C#?"            │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   QUERY CLASSIFIER     │  ← flan-t5-base (local)
              │  (Intent, Reasoning    │
              │   Type, Sub-questions) │
              └──────────┬─────────────┘
                         │
            ┌────────────┼────────────────┐
            │            │                │
            ▼            ▼                ▼
    ┌──────────┐  ┌────────────┐  ┌──────────────┐
    │COMMONSENSE│  │  ADAPTIVE  │  │  STRATEGIC   │
    │  (Simple) │  │(Sub-query  │  │(Hierarchical │
    │           │  │ Decompose) │  │ + Self-Cons.) │
    └─────┬────┘  └─────┬──────┘  └──────┬───────┘
          │             │                │
          └─────────────┼────────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │     HYBRID RETRIEVAL        │
         │  ┌────────┐  ┌──────────┐   │
         │  │ FAISS  │  │  BM25    │   │
         │  │(Dense) │  │ (Sparse) │   │
         │  └───┬────┘  └────┬─────┘   │
         │      └──────┬─────┘         │
         │      Reciprocal Rank        │
         │         Fusion (RRF)        │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │    CROSS-ENCODER RE-RANKER  │
         │  ms-marco-MiniLM-L-6-v2    │
         │  + StackOverflow Preference │
         │    Signals (score, accepted)│
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │   GROUNDED GENERATION       │
         │   flan-t5-base (local)      │
         │   Chain-of-Thought Prompt   │
         │   + Self-Consistency (n=3)  │
         └──────────────┬──────────────┘
                        │
                        ▼
              ┌────────────────────┐
              │   FINAL ANSWER     │
              │  (Grounded in      │
              │   retrieved sources)│
              └────────────────────┘
```

---

## 4. Dataset

### Source
- **HuggingFace Dataset:** [`HuggingFaceH4/stack-exchange-preferences`](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences)
- This is a large-scale dataset containing Q\&A pairs from 170+ Stack Exchange sites, enriched with human preference signals (upvotes, accepted answers).

### Filtering Criteria
The raw dataset is filtered during preprocessing to retain only **high-quality, software-relevant** Q\&A:

| Filter | Condition |
|--------|-----------|
| **Domain** | Only `stackoverflow`, `askubuntu`, `softwareengineering` |
| **Quality** | At least one answer with `pm_score >= 5` OR `selected == true` (accepted answer) |
| **Deduplication** | Unique `qid` only — no duplicate questions |

### Processed Data Statistics
| Metric | Value |
|--------|-------|
| Total questions | **1,001** |
| Total answer chunks | **9,523** |
| Average answers per question | ~9.5 |
| Domains | `Stackoverflow` |
| Output format | JSONL (`data/processed_dataset.jsonl`) |

### Data Schema (per JSONL line)
```json
{
  "question_id": 4,
  "title": "",
  "question": "I want to assign the decimal variable...",
  "domain": "Stackoverflow",
  "reasoning_category": "Procedural",
  "answers": [
    {
      "answer_id": 7,
      "author": "Kevin Dente",
      "pm_score": 10,
      "selected": true,
      "text": "<p>An explicit cast to <code>double</code>...</p>",
      "body_clean": "An explicit cast to double like this isn't necessary...",
      "score": 10,
      "is_accepted": true
    }
  ]
}
```

---

## 5. Project Structure

```
reasoning-rag/
├── configs/
│   └── taxonomy.json              # Domain → Reasoning category mapping
├── data/
│   └── processed_dataset.jsonl    # Preprocessed StackOverflow Q&A data
├── index/
│   ├── dense.faiss                # FAISS vector index (BGE-large embeddings)
│   ├── bm25.pkl                   # Serialized BM25 sparse index
│   └── metadata.json              # Chunk metadata (score, domain, text, etc.)
├── src/
│   ├── demo.py                    # Interactive CLI demo
│   ├── ingestion/
│   │   └── preprocess.py          # Data download, filtering, cleaning
│   ├── retrieval/
│   │   ├── dense_index.py         # FAISS index builder
│   │   ├── sparse_index.py        # BM25 index builder
│   │   ├── hybrid_search.py       # Hybrid retrieval + RRF fusion
│   │   └── reranker.py            # Cross-encoder re-ranking
│   ├── reasoning/
│   │   ├── classifier.py          # Query classification (intent, reasoning type)
│   │   └── engine.py              # Multi-path reasoning orchestrator
│   ├── generation/
│   │   ├── generator.py           # LLM-based grounded answer generation
│   │   └── trace.py               # Reasoning trace data structure
│   └── evaluation/
│       └── evaluator.py           # ROUGE & BERTScore evaluation
├── notebooks/                     # Jupyter notebooks for experimentation
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
└── venv/                          # Virtual environment
```

---

## 6. Module Deep Dive

### 6.1 Data Ingestion & Preprocessing
**File:** `src/ingestion/preprocess.py`

Downloads the `HuggingFaceH4/stack-exchange-preferences` dataset in streaming mode (avoids loading 20GB+ into RAM) and applies:

1. **Domain Extraction:** Parses the `metadata` URL field to extract the sub-site (e.g., `stackoverflow.com/...` → `stackoverflow`).
2. **Domain Filtering:** Retains only software-relevant domains: `stackoverflow`, `askubuntu`, `softwareengineering`.
3. **Quality Filtering:** Keeps a question only if at least one answer has a community score ≥ 5 or is the accepted answer.
4. **HTML Cleaning:** Strips all HTML tags from answer bodies using BeautifulSoup, producing clean plaintext.
5. **Taxonomy Labeling:** Maps each domain to a reasoning category (e.g., `stackoverflow` → `Procedural`) using `configs/taxonomy.json`.

**Run:**
```bash
./venv/bin/python src/ingestion/preprocess.py --max-samples 1000
```

---

### 6.2 Dense Indexing (FAISS)
**File:** `src/retrieval/dense_index.py`

Builds a dense vector index for semantic similarity search:

- **Embedding Model:** [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) — a state-of-the-art 1024-dim embedding model, top-ranked on the MTEB leaderboard.
- **Chunk Format:** Each answer is stored as `"Q: {title}\nA: {answer_body}"` — prepending the question gives the embedding model context about what the answer is addressing.
- **Index Type:** `faiss.IndexFlatIP` (Inner Product) — since embeddings are L2-normalized during encoding, inner product is equivalent to cosine similarity.
- **Batch Processing:** Embeddings are computed in batches of 64 to manage memory.

**Output:**
- `index/dense.faiss` — The FAISS binary index file.
- `index/metadata.json` — Parallel metadata array (chunk_id, question_id, score, is_accepted, domain, chunk_text).

---

### 6.3 Sparse Indexing (BM25)
**File:** `src/retrieval/sparse_index.py`

Builds a keyword-based sparse index using the Okapi BM25 algorithm:

- **Tokenization:** Simple whitespace tokenization after lowercasing.
- **Why BM25?** Dense embeddings capture semantic meaning ("similar vibes") but can miss exact keyword matches. BM25 excels at finding documents containing specific terms like error codes, API names, or exact function names.
- **Source:** Reads chunk text from the metadata generated by the dense indexer.

**Output:** `index/bm25.pkl` — Serialized BM25 index.

---

### 6.4 Hybrid Retrieval & Reciprocal Rank Fusion
**File:** `src/retrieval/hybrid_search.py`

Combines dense and sparse retrieval for the best of both worlds:

1. **Dense Search:** Encodes the query with BGE-large, searches FAISS for top-K semantically similar chunks.
2. **Sparse Search:** Tokenizes the query, scores all chunks with BM25, takes top-K by keyword relevance.
3. **Reciprocal Rank Fusion (RRF):** Merges both ranked lists using the formula:

   ```
   RRF_score(doc) = Σ 1 / (k + rank_i(doc))    where k = 60
   ```

   This gives each document a combined score based on its position in both ranked lists. Documents appearing in both lists get boosted. The constant `k = 60` is the standard value from the original RRF paper.

---

### 6.5 Cross-Encoder Re-Ranking
**File:** `src/retrieval/reranker.py`

After hybrid retrieval returns ~20 candidates, a **Cross-Encoder** re-reads every (query, chunk) pair jointly and produces a refined relevance score:

- **Model:** [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) — a distilled BERT model fine-tuned on MS MARCO for passage re-ranking.
- **Preference Signal Incorporation:** The cross-encoder score is augmented with StackOverflow community signals:
  ```
  final_score = CE_score + 0.1 * min(upvote_score / 100, 1.0) + 0.15 * is_accepted
  ```
  This means accepted answers get a +0.15 bonus and highly upvoted answers get up to +0.1 bonus, ensuring community-validated answers are preferred.

---

### 6.6 Query Classification (Reasoning Layer)
**File:** `src/reasoning/classifier.py`

The brain of the reasoning system. Before any retrieval happens, the classifier analyzes the incoming query and determines:

| Field | Values | Purpose |
|-------|--------|---------|
| **Intent** | `factual`, `procedural`, `comparative`, `conceptual`, `opinion`, `debugging` | What kind of answer is the user expecting? |
| **Reasoning Type** | `commonsense`, `adaptive`, `strategic` | Which reasoning path to route to (see §6.7) |
| **Scope** | `single_topic`, `multi_topic`, `hierarchical` | How broad is the query? |
| **Ambiguity** | `low`, `medium`, `high` | How uncertain is the intent? |
| **Sub-questions** | List of 1–3 decomposed queries | Breaking complex queries into answerable parts |

**Implementation Details:**
- Uses `google/flan-t5-base` running locally via HuggingFace Transformers.
- The prompt requests structured `Key: Value` text output (not JSON — small models cannot reliably produce valid JSON).
- A custom string parser extracts each field using line-by-line splitting.
- Includes robust fallback defaults if parsing fails.

---

### 6.7 Multi-Path Reasoning Engine
**File:** `src/reasoning/engine.py`

The engine orchestrates three distinct reasoning paths inspired by cognitive science:

#### Commonsense Path (System 1 — Fast)
- **When:** Simple, factual, single-topic queries (e.g., *"What is a pointer in C?"*).
- **Behavior:** Single-pass hybrid retrieval → re-rank top-5 → generate answer directly.
- **Analogy:** Like a developer Googling a quick syntax question.

#### Adaptive Path (System 2 — Moderate)
- **When:** Multi-faceted queries that benefit from decomposition (e.g., *"How do I set up SSH keys and configure port forwarding?"*).
- **Behavior:** Decomposes into sub-questions → independent retrieval per sub-question → merge and deduplicate → re-rank → synthesize unified answer.
- **Analogy:** Like a developer breaking a complex task into steps and researching each.

#### Strategic Path (System 2 — Deep)
- **When:** Complex comparative, architectural, or hierarchical queries (e.g., *"Compare SQL vs NoSQL databases for a high-throughput real-time analytics pipeline"*).
- **Behavior:** Level-1 broad retrieval + per-sub-question targeted retrieval → hierarchical merge and deduplication → re-rank → **self-consistency decoding** (generate N=3 candidate answers, select the most comprehensive).
- **Analogy:** Like a senior engineer researching a design decision from multiple angles.

---

### 6.8 Grounded Generation
**File:** `src/generation/generator.py`

Generates the final answer using `google/flan-t5-base`:

- **Grounding:** The prompt explicitly includes retrieved source text, forcing the model to base its answer on evidence rather than hallucinate.
- **Chain-of-Thought (CoT):** Different reasoning-type-specific instructions guide the model:
  - *Commonsense:* "Answer the question directly based on the sources above."
  - *Adaptive:* "First address each sub-question separately, then synthesize into a unified answer."
  - *Strategic:* "First identify the main categories... then address each... finally provide a cross-category answer."
- **Context Truncation:** To prevent token overflow in the small model, context is hard-limited to top-1 source with 500 character cap.
- **Self-Consistency Decoding:** For strategic queries, generates 3 candidate answers and selects the longest (most comprehensive) one.

---

### 6.9 Evaluation
**File:** `src/evaluation/evaluator.py`

Provides automated evaluation metrics:

| Metric | Library | Purpose |
|--------|---------|---------|
| **ROUGE** (1, 2, L) | `evaluate` | Measures n-gram overlap between generated and reference answers |
| **BERTScore** | `bert_score` | Measures semantic similarity using contextual BERT embeddings |
| **Recall@K** | Custom | Checks if the expected answer appears in the top-K retrieved chunks |

---

## 7. Setup & Installation

### Prerequisites
- Python 3.9+
- ~4 GB free disk space (for models and indices)
- No GPU required (runs on CPU/Apple MPS)

### Step-by-Step

```bash
# Clone or navigate to the project
cd reasoning-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Step 1: Download & preprocess the dataset (takes ~2-3 min)
python src/ingestion/preprocess.py --max-samples 1000

# Step 2: Build FAISS dense index (takes ~15-40 min on CPU)
python src/retrieval/dense_index.py

# Step 3: Build BM25 sparse index (takes ~5 seconds)
python src/retrieval/sparse_index.py

# Step 4: Launch the interactive demo
python src/demo.py
```

---

## 8. Usage

After running `python src/demo.py`, you will see:

```
Loading Models and Indices... This might take a moment.
Loading local LLM for classification: google/flan-t5-base...
Initializing Reasoning Engine...
Loading Embedding Model...
Loading FAISS from index/dense.faiss...
Loading BM25 from index/bm25.pkl...

Type your question below (or 'quit' to exit):

Query: _
```

Type any software engineering question and press Enter. The system will:
1. Classify your query (intent, reasoning type, scope).
2. Optionally decompose into sub-questions.
3. Retrieve and re-rank relevant StackOverflow answers.
4. Generate a grounded answer citing the sources.

Type `quit` or `exit` to stop.

---

## 9. Example Queries

Here are queries that work well with our indexed StackOverflow data:

| Query | Expected Reasoning Path |
|-------|------------------------|
| "How to convert a decimal to double in C#?" | Commonsense |
| "What is the difference between decimal and double?" | Commonsense / Adaptive |
| "How do I calculate someone's age from their birthday in C#?" | Procedural / Commonsense |
| "Why does percentage width not work in absolutely positioned divs in IE7?" | Debugging / Commonsense |
| "Compare implicit vs explicit type casting in C# and VB.NET" | Comparative / Strategic |
| "How to set up SSH keys on Ubuntu?" | Procedural / Commonsense |

---

## 10. Technologies Used

| Component | Technology | Role |
|-----------|-----------|------|
| **Language Model** | `google/flan-t5-base` | Query classification + answer generation (local, offline) |
| **Dense Embeddings** | `BAAI/bge-large-en-v1.5` | Semantic vector representations (1024-dim) |
| **Vector Database** | `faiss-cpu` (IndexFlatIP) | Approximate nearest neighbor search |
| **Sparse Retrieval** | `rank_bm25` (BM25Okapi) | Keyword-based retrieval |
| **Re-Ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Pairwise relevance scoring |
| **Orchestration** | `langchain` | Prompt templates, LLM chaining |
| **Data Source** | `HuggingFaceH4/stack-exchange-preferences` | 170+ Stack Exchange sites with preference signals |
| **HTML Cleaning** | `beautifulsoup4` | Strip HTML from raw answer bodies |
| **Evaluation** | `evaluate`, `rouge_score`, `bert_score` | Generation and retrieval quality metrics |
| **Runtime** | Python 3.9 + venv | Fully local, no cloud dependencies |

---

## 11. Limitations & Future Work

### Current Limitations

1. **Small LLM:** `flan-t5-base` (250M params) has limited reasoning capacity. It sometimes produces repetitive or truncated answers for complex queries.
2. **No GPU Acceleration:** FAISS index build takes ~35 minutes on CPU. With a CUDA GPU, this would take ~2 minutes.
3. **Context Window:** The 512-token limit of flan-t5-base forces aggressive context truncation (top-1 source, 500 chars). Important context from lower-ranked sources may be lost.
4. **Dataset Scale:** Currently indexed with 1,000 questions (9,523 answer chunks). The full dataset contains millions of questions.
5. **Missing Titles:** The HuggingFace dataset schema doesn't expose question titles at the root level, so chunk embeddings are formatted as `Q: \nA: {answer}` without explicit titles.

### Future Improvements

1. **Upgrade LLM:** Switch to `google/gemma-2b-it`, `Llama-3-8B-Instruct`, or `Mistral-7B-Instruct` for dramatically better reasoning and generation quality.
2. **Scale Dataset:** Remove the `--max-samples` limit and index the full dataset across all software-related Stack Exchange sites.
3. **GPU Indexing:** Use `faiss-gpu` for 10–20× faster index building.
4. **Chunking Strategy:** Implement semantic chunking (split long answers at paragraph/code-block boundaries) instead of treating entire answers as single chunks.
5. **Evaluation Pipeline:** Run automated evaluation against a held-out test set using ROUGE/BERTScore to quantify answer quality improvements.
6. **Web UI:** Build a Gradio or Streamlit frontend for a visual, browser-based demo instead of CLI.

---

## 📄 License

This project is for educational and research purposes.

---

*Built with ❤️ using HuggingFace Transformers, FAISS, LangChain, and the Stack Exchange community's collective knowledge.*
