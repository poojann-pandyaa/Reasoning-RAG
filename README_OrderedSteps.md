# Reasoning-Augmented RAG — Ordered Implementation Steps

> **Project:** Contextual, Adaptive & Strategic Reasoning over RAG  
> **Dataset:** `HuggingFaceH4/stack-exchange-preferences`  
> **Companion doc:** `README_EndToEnd_Protocol.md`

---

## How to use this document

Follow these steps in strict order. Each phase builds on the previous. Do not proceed to Phase 3 (embedding) before Phase 2 (preprocessing) is complete — dirty data in the vector index is very hard to fix without re-indexing from scratch.

Steps marked `[VALIDATE]` must pass before moving on.

---

## Phase 1 — Environment Setup

**Step 1 — Create project structure**

```
reasoning-rag/
├── data/               # Raw and processed dataset files
├── index/              # Vector store and BM25 index artifacts
├── src/
│   ├── ingestion/      # Dataset loading and preprocessing
│   ├── retrieval/      # Dense, sparse, hybrid, and re-ranker
│   ├── reasoning/      # Query classifier, decomposer, reasoning engine
│   ├── generation/     # Prompt builder and generator
│   └── evaluation/     # Metrics and evaluation harness
├── notebooks/          # Exploration and analysis
├── configs/            # Model names, thresholds, hyperparameters
└── tests/
```

**Step 2 — Install dependencies**

```bash
pip install datasets transformers sentence-transformers faiss-cpu rank_bm25 \
            langchain openai evaluate rouge_score bert_score tqdm pandas
```

For GPU environments replace `faiss-cpu` with `faiss-gpu`.

**Step 3 — Configure HuggingFace credentials**

```bash
huggingface-cli login
```

Set cache directory to a disk with sufficient space — the full dataset is large.

```python
import os
os.environ["HF_DATASETS_CACHE"] = "/path/to/large/disk/hf_cache"
```

---

## Phase 2 — Dataset Acquisition and Profiling

**Step 4 — Load the dataset**

```python
from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceH4/stack-exchange-preferences",
    split="train",
    streaming=True  # Use streaming to inspect before full download
)
```

Inspect the schema. Key fields you will use: `question`, `answers` (list), `score` per answer, `is_accepted` per answer.

**Step 5 — Profile the data**

Before any processing, run a profiling notebook to understand:
- Domain distribution across Stack Exchange sub-sites
- Distribution of answer scores (mean, median, percentiles)
- Question length distribution
- Ratio of accepted vs non-accepted answers
- Proportion of questions with zero highly-scored answers

This profile determines your filtering thresholds in the next step.

**Step 6 — Filter and clean**

Apply these filters:
- Keep only questions that have at least one answer with score >= 5 OR an accepted answer
- Strip HTML tags from question body and answer body (use `BeautifulSoup` or `html.parser`)
- Remove code blocks longer than 500 tokens from the main text body (keep a separate code field)
- Deduplicate by question ID

```python
from bs4 import BeautifulSoup

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
```

`[VALIDATE]` After filtering, confirm: at least 1M records remain, no null body fields, HTML tags absent in a random sample of 100 records.

**Step 7 — Build category taxonomy**

Extract the Stack Exchange sub-site tag from each question (available in the metadata). Map to coarse reasoning categories:

| Stack Exchange Domain | Reasoning Category |
|---|---|
| stackoverflow, superuser | Procedural / Debugging |
| math, stats, datascience | Analytical / Comparative |
| english, writing | Linguistic / Factual |
| physics, chemistry | Conceptual / Explanatory |
| workplace, academia | Opinion / Advisory |

Save the taxonomy as a config file. It is used by the Query Classifier in Phase 5.

---

## Phase 3 — Embedding and Vector Index

**Step 8 — Choose and load embedding model**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
```

This model outperforms `all-mpnet-base-v2` on technical retrieval benchmarks. If GPU memory is limited, use `BAAI/bge-base-en-v1.5` (768-dim, smaller).

**Step 9 — Design chunking strategy**

Do NOT chunk by fixed token count. For Q&A data, use **semantic chunking**:
- Each accepted or high-score answer is one chunk
- For very long answers (> 512 tokens), split at paragraph boundaries, not mid-sentence
- Prepend the question title to every answer chunk: `"Q: {title}\nA: {answer_body}"`

This prepending is critical — it gives the embedding model the question context when encoding the answer, producing better retrieval.

**Step 10 — Embed and store with metadata**

```python
import faiss
import numpy as np

embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Normalise for cosine similarity
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine after normalisation
index.add(embeddings)

faiss.write_index(index, "index/dense.faiss")
```

Store metadata (question_id, answer_id, score, is_accepted, domain) in a parallel list or SQLite table keyed by chunk index position. You will need this for the re-ranker and for source attribution in outputs.

`[VALIDATE]` Run 5 test queries. Inspect top-10 results. Confirm they are semantically relevant, not gibberish.

---

## Phase 4 — Sparse Retrieval Index

**Step 11 — Build BM25 index**

```python
from rank_bm25 import BM25Okapi
import pickle

tokenised_corpus = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenised_corpus)

with open("index/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)
```

For production scale (> 5M chunks), replace `rank_bm25` with Elasticsearch or OpenSearch, which supports distributed BM25 with filtering.

`[VALIDATE]` Query with an exact technical term (e.g., a specific function name). Confirm BM25 returns it in top-5 even if the dense index does not.

---

## Phase 5 — Contextual Query Understanding Module

**Step 12 — Build the query classifier**

Create a prompted LLM call that takes a raw user query and outputs structured JSON:

```python
CLASSIFIER_PROMPT = """
You are a query analysis expert. Given a user query, output ONLY valid JSON with these fields:
- intent: one of [factual, procedural, comparative, conceptual, opinion, debugging]
- reasoning_type: one of [commonsense, adaptive, strategic]
- entities: list of key named concepts or technologies
- scope: one of [single_topic, multi_topic, hierarchical]
- ambiguity: one of [low, medium, high]
- sub_questions: list of 1-5 sub-questions that must be answered to fully address the query

Query: {query}
"""
```

Commonsense = single_topic, low ambiguity, factual or procedural.
Adaptive = multi_topic, moderate complexity.
Strategic = hierarchical scope, or queries where the answer requires assembling information across categories.

**Step 13 — Add ambiguity handling**

If `ambiguity == high`, log an assumption before proceeding. Do not ask the user for clarification in the first version — assume the broadest reasonable interpretation and note it in the reasoning trace.

---

## Phase 6 — Reasoning Engine

**Step 14 — Implement the commonsense reasoning path**

Direct top-k retrieval. No decomposition. Pass top-5 chunks to the generator with a simple context template.

**Step 15 — Implement the adaptive reasoning path**

For each sub-question from the classifier output, run independent retrieval (both dense and sparse). Merge all retrieved chunks. Deduplicate by chunk ID. Pass merged context to generator, instructing it to address each sub-question in order.

**Step 16 — Implement the strategic reasoning path**

This is the core differentiator. Implement as a tree-structured retrieval loop:

```
Level 1: Retrieve for the broad query → identify top categories in results
Level 2: For each category, retrieve independently using category-specific sub-query
Level 3: For each category result, retrieve for specific facts if needed
Synthesise: pass all level results to generator with explicit structure instruction
```

The number of levels is determined by the `scope` classification. Hierarchical scope triggers 2-3 level retrieval. Multi-topic triggers 2-level. Single-topic triggers 1-level (commonsense path).

---

## Phase 7 — Hybrid Retrieval and Fusion

**Step 17 — Implement hybrid retrieval function**

```python
def hybrid_retrieve(query, top_k=20):
    # Dense
    q_embedding = model.encode([query])
    faiss.normalize_L2(q_embedding)
    dense_scores, dense_ids = index.search(q_embedding, top_k)

    # Sparse
    tokenised_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenised_query)
    sparse_ids = sparse_scores.argsort()[-top_k:][::-1]

    # Reciprocal Rank Fusion
    rrf_scores = {}
    for rank, idx in enumerate(dense_ids[0]):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)
    for rank, idx in enumerate(sparse_ids):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in fused[:top_k]]
```

The constant 60 in the RRF formula is standard. Do not tune it unless you have a labelled validation set.

**Step 18 — Implement cross-encoder re-ranker**

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidate_ids, top_k=5):
    pairs = [(query, chunks[idx]) for idx in candidate_ids]
    scores = reranker.predict(pairs)

    # Incorporate Stack Exchange preference signals
    for i, idx in enumerate(candidate_ids):
        meta = metadata[idx]
        scores[i] += 0.1 * min(meta["score"] / 100, 1.0)   # normalised upvote signal
        scores[i] += 0.15 if meta["is_accepted"] else 0     # accepted answer bonus

    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:top_k]]
```

The weighting of the metadata signals (0.1, 0.15) are starting heuristics. Tune on a held-out evaluation set after Phase 9.

`[VALIDATE]` On 20 test queries, confirm that accepted answers appear in top-3 more often than random baseline.

---

## Phase 8 — Generation

**Step 19 — Build the generation prompt template**

```python
def build_prompt(query, retrieved_chunks, reasoning_type, sub_questions):
    context = "\n\n".join([
        f"[Source {i+1} | Score: {metadata[idx]['score']} | Accepted: {metadata[idx]['is_accepted']}]\n{chunks[idx]}"
        for i, idx in enumerate(retrieved_chunks)
    ])

    cot_instruction = {
        "commonsense": "Answer the question directly based on the sources above.",
        "adaptive":    "First address each sub-question separately, then synthesise into a unified answer.",
        "strategic":   "First identify the main categories relevant to this query. Then address each category using the sources. Finally, provide a synthesised cross-category answer."
    }[reasoning_type]

    return f"""You are a technical expert answering based on retrieved Stack Exchange content.

Retrieved context:
{context}

Sub-questions to address: {sub_questions}

Instruction: {cot_instruction}

Question: {query}

Reason step by step through the evidence before writing your final answer."""
```

**Step 20 — Implement self-consistency for strategic queries**

```python
def generate_with_consistency(prompt, n=3, temperature=0.7):
    responses = [llm(prompt, temperature=temperature) for _ in range(n)]
    # Simple version: return the longest response (proxy for most complete)
    # Advanced version: cluster responses and pick the centroid
    return max(responses, key=len)
```

Apply self-consistency only for `reasoning_type == "strategic"` — it is expensive and not justified for simpler queries.

**Step 21 — Build the reasoning trace logger**

```python
class ReasoningTrace:
    def __init__(self, query):
        self.query = query
        self.classification = {}
        self.sub_queries = []
        self.retrieved_per_subquery = {}
        self.reranked_final = []
        self.generation_prompt = ""
        self.final_answer = ""

    def to_dict(self):
        return self.__dict__
```

Attach a `ReasoningTrace` object to every request. Populate it at each step. Include it in the final response payload.

---

## Phase 9 — Evaluation

**Step 22 — Build the evaluation harness**

Hold out 5,000 question-answer pairs from the dataset (not used in indexing). For each held-out question:
- Run the full pipeline
- Compare generated answer to the accepted answer using ROUGE-L and BERTScore
- Check whether the accepted answer's chunk was retrieved in top-5 (Recall@5)

```python
from evaluate import load
rouge = load("rouge")
bertscore = load("bertscore")
```

**Step 23 — Run baseline comparison**

Build a minimal vanilla RAG baseline: retrieve top-5 by dense similarity only, generate with a flat context, no chain-of-thought. Compare metrics against the reasoning-augmented system.

Document the delta. This is the primary evidence that reasoning adds value.

**Step 24 — Human evaluation on reasoning traces**

For 200 queries (stratified across reasoning types), have a human annotator assess:
- Did the decomposition correctly identify the sub-questions? (0-2 score)
- Did the retrieval surface relevant chunks for each sub-question? (0-2 score)
- Does the reasoning trace accurately explain the answer? (0-2 score)

Compute inter-annotator agreement if resources allow.

---

## Phase 10 — Demonstration Build

**Step 25 — Build the demo harness**

Create a CLI or notebook interface that for any input query outputs:

```
=== Query ===
[user query]

=== Classification ===
Intent: procedural | Reasoning type: adaptive | Scope: multi_topic

=== Sub-queries issued ===
1. [sub-query 1]
2. [sub-query 2]

=== Retrieved sources ===
[Source 1] Score: 47 | Accepted: True | Domain: stackoverflow
[chunk text...]

=== Reasoning trace ===
Step 1: Addressed sub-question 1 using Source 1 and Source 3...
Step 2: Addressed sub-question 2 using Source 2...
Step 3: Synthesised both into unified answer...

=== Final Answer ===
[generated answer]

--- Baseline (vanilla RAG, no reasoning) ---
[vanilla answer for comparison]
```

**Step 26 — Prepare three showcase queries**

One per reasoning type:
- Commonsense: "How do I reverse a list in Python?"
- Adaptive: "What are the tradeoffs between using Redis and Memcached?"
- Strategic: "What are best practices for database indexing across OLTP, OLAP, and time-series use cases?"

Run all three, capture outputs, and include them in the project documentation.

---

## Completion Checklist

- [ ] Dataset loaded, profiled, cleaned, and indexed (Phases 1-4)
- [ ] Query classifier tested on 50 diverse queries with correct classification rate > 80%
- [ ] Hybrid retrieval validated: accepted-answer Recall@5 > 60%
- [ ] Re-ranker validated: accepted-answer moves up in ranking vs pre-rerank baseline
- [ ] Generation produces chain-of-thought traces for all three reasoning modes
- [ ] ROUGE-L of reasoning system > vanilla RAG baseline on held-out set
- [ ] Three showcase queries documented with full traces
- [ ] Reasoning trace logger attached to all responses

---

*This implementation guide is the companion to `README_EndToEnd_Protocol.md`, which explains the system architecture and design decisions.*
