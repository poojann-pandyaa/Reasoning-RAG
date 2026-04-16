# Reasoning-Augmented RAG — End-to-End Protocol

> **Project:** Contextual, Adaptive & Strategic Reasoning over RAG  
> **Dataset:** `HuggingFaceH4/stack-exchange-preferences` (HuggingFace)  
> **Role:** Senior RAG Architect (20 years experience)

---

## Overview

This document defines the **end-to-end system protocol** for building a Reasoning-Augmented Retrieval-Augmented Generation (RAG) system. The system goes beyond vanilla RAG by making **reasoning a first-class citizen** — query understanding, decomposition, and reasoning strategy selection happen *before* any retrieval occurs.

The core problem this solves: standard RAG retrieves a flat list of loosely related chunks and concatenates them. This system instead reasons about *what kind of answer is needed*, *what sub-questions must be answered*, and *how to synthesize across retrieved evidence* before generating a response.

---

## System Philosophy

```
Standard RAG:    Query → Retrieve → Generate
This System:     Query → Understand → Reason → Decompose → Retrieve → Re-rank → Reason → Generate
```

Reasoning is applied at **two points** in the pipeline:

1. **Pre-retrieval** — to understand, classify, and decompose the query into a structured retrieval plan
2. **Post-retrieval** — to synthesize across retrieved chunks using chain-of-thought before generating the final answer

---

## Architecture Layers

### Layer 0 — User Query Interface
The entry point. Accepts raw natural language input with no constraints on phrasing. The system handles ambiguity internally rather than requiring structured input from the user.

### Layer 1 — Contextual Query Understanding Module
Parses the raw query to extract:
- **Intent** — what the user actually wants (fact, comparison, explanation, procedure, opinion)
- **Entities** — named concepts, technologies, topics
- **Scope** — single-topic or multi-category
- **Reasoning type required** — commonsense, adaptive, or strategic
- **Ambiguity level** — whether clarification or assumption-logging is needed

Output: structured JSON classification used by all downstream modules.

### Layer 2 — Reasoning Engine
The central decision layer. Selects and executes one of three reasoning modes based on Layer 1's output:

| Reasoning Mode | Trigger | Behaviour |
|---|---|---|
| **Commonsense** | Simple, unambiguous, single-topic queries | Direct retrieval, minimal decomposition |
| **Adaptive** | Moderate complexity, 2–3 related sub-topics | Decomposes into parallel sub-queries, merges contexts |
| **Strategic** | Complex, hierarchical, multi-domain queries | Builds a retrieval tree, executes level by level, synthesizes across branches |

The IIITB placement example from the project brief is a *strategic reasoning* case: broad query → sub-categories (BTech, MTech) → specific courses (CSE, ECE, DSAi) → specific facts per category.

### Layer 3 — Query Decomposer and Sub-query Planner
Translates the reasoning plan into concrete retrieval sub-tasks. For each sub-task, outputs:
- A refined query string optimised for retrieval
- A retrieval priority rank
- The expected answer type (factual snippet, list, comparison table, etc.)

### Layer 4 — Hybrid Retrieval
Two parallel retrieval systems run for each sub-query:

**Dense retrieval** uses embedding-based similarity search against a vector index. Captures semantic meaning, handles paraphrasing and synonyms well. Uses `sentence-transformers/BAAI/bge-large-en-v1.5` or equivalent.

**Sparse retrieval** uses BM25 keyword matching. Captures exact terms, error codes, API names, and technical identifiers that dense retrieval misses.

Results from both systems are merged using **Reciprocal Rank Fusion (RRF)** — a position-based fusion method that is robust to score scale differences between the two systems.

### Layer 5 — Context-Aware Re-ranker
A cross-encoder re-ranks the fused candidate chunks against the original query. Unlike bi-encoder retrieval (which computes query and document embeddings independently), a cross-encoder sees the query and document together and produces a more accurate relevance score.

The re-ranker incorporates **dataset-specific signals** from the Stack Exchange Preferences dataset:
- Upvote score of the answer
- Whether the answer was accepted
- Recency of the answer (for time-sensitive topics)

These signals act as a prior — a high-voted, accepted answer is ranked higher than a semantically similar low-voted answer, reflecting real human preference.

### Layer 6 — Reasoning-Augmented Generator
The generation stage is explicitly structured to surface reasoning. The prompt construction follows this template:

```
[System role + reasoning mode declaration]
[Retrieved context chunks, labelled by source]
[Chain-of-thought instruction: reason through the evidence before answering]
[Original user query]
```

For high-stakes or ambiguous queries, **self-consistency decoding** is applied: three responses are generated with non-zero temperature, and the consensus or majority answer is selected.

Every response includes a **reasoning trace** — a structured log of:
- Sub-queries issued
- Chunks retrieved and their sources
- How the reasoning was assembled into the final answer

### Layer 7 — Structured Grounded Response
The final output to the user contains:
- The generated answer
- Source attribution (which Stack Exchange answers were used)
- The reasoning trace (collapsible in UI)
- Confidence signal (derived from self-consistency agreement rate)

---

## Dataset: Stack Exchange Preferences

**Source:** `HuggingFaceH4/stack-exchange-preferences` on HuggingFace  
**Scale:** ~10 million Q&A pairs across Stack Exchange domains  
**Key fields:** `question_id`, `title`, `body`, `answers`, `score`, `is_accepted`

The dataset is uniquely suited to this project because:

1. It contains **human preference signals** (upvotes, accepted answers) that can supervise the re-ranker without additional annotation
2. It spans **diverse reasoning types** — procedural (how-to), factual (what-is), comparative (which-is-better), debugging (why-is-this-wrong)
3. The Q&A structure naturally supports **ground-truth evaluation** — the accepted answer provides a gold standard for measuring retrieval and generation quality

---

## Reasoning Types Demonstrated

### Commonsense Reasoning
Applied when the query is direct and the answer is contained within a single retrieved passage. Example: "What is the default value of max_connections in PostgreSQL?"

The system retrieves, identifies the relevant passage, and generates a concise answer. Reasoning trace is minimal.

### Adaptive Reasoning
Applied when the query spans 2–3 related aspects. Example: "What are the tradeoffs between using Redis and Memcached for session storage?"

The system decomposes into sub-queries for Redis capabilities, Memcached capabilities, and comparison criteria. Retrieves independently, merges contexts, and generates a structured comparison.

### Strategic Reasoning
Applied when the query is broad and hierarchical. Example: "What are the best practices for database indexing across different use cases?"

The system first identifies the top-level categories (OLTP, OLAP, time-series, full-text search), then retrieves for each, then synthesizes a structured answer that addresses each category before giving cross-cutting recommendations. This is the same pattern as the IIITB placement example: broad → category → specific.

---

## Evaluation Protocol

| Dimension | Metric | Source |
|---|---|---|
| Retrieval quality | NDCG@10 | Held-out Q&A pairs |
| Generation quality | ROUGE-L, BERTScore | Accepted answers as gold standard |
| Reasoning quality | Human eval on trace | Sample of 200 queries |
| Re-ranker calibration | Accepted-answer recall@5 | Dataset labels |

---

## Technology Stack

| Component | Technology |
|---|---|
| Dataset loading | `datasets` (HuggingFace) |
| Embedding model | `BAAI/bge-large-en-v1.5` |
| Vector store | FAISS (local) / Qdrant (production) |
| Sparse retrieval | BM25 via `rank_bm25` |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM (generation) | Open-source LLM via HuggingFace or API |
| Orchestration | LangChain / custom pipeline |
| Evaluation | `evaluate`, `nltk`, `bert_score` |

---

## Key Design Decisions

**Reasoning before retrieval, not after.** The query classification and decomposition happen before the first vector lookup. This allows the retrieval strategy to be tailored to the query type, rather than applying a one-size-fits-all top-k retrieval.

**Hybrid retrieval is non-negotiable for technical domains.** Stack Exchange content is full of exact identifiers — function names, error codes, version numbers — that dense retrieval handles poorly. BM25 is not a fallback; it is a required complement.

**Human preference signals replace annotation cost.** The upvote and accepted-answer labels in the dataset are a free source of quality supervision for the re-ranker. This is a major advantage of the chosen dataset.

**Reasoning traces are a product feature, not just a debug tool.** Users of technical Q&A systems benefit from seeing *why* an answer was given, not just *what* the answer is. The trace builds trust and allows users to verify or question the reasoning.

---

*This protocol document is the companion to `README_OrderedSteps.md`, which provides the concrete implementation sequence.*
