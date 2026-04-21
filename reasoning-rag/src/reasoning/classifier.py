import os
import torch
from typing import Optional
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

CLASSIFIER_PROMPT = """You are a query classification system. Classify the query into the correct reasoning type.

Reasoning types:
- commonsense: simple factual question with a single, direct answer
- adaptive: multi-part question that requires covering several sub-topics or concepts together
- strategic: complex comparison, tradeoff, or design decision with no single correct answer

Examples:
Query: How do I reverse a list in Python?
Intent: procedural
Reasoning Type: commonsense
Scope: single_topic
Sub-questions: How do I reverse a list in Python?

Query: What does git stash do?
Intent: factual
Reasoning Type: commonsense
Scope: single_topic
Sub-questions: What does git stash do?

Query: What is async/await and when should I use it?
Intent: conceptual
Reasoning Type: adaptive
Scope: multi_topic
Sub-questions: What is async/await in Python?, How does the event loop work with async/await?, When should you use async/await vs threading?

Query: What is LoRA and how do I implement it?
Intent: conceptual
Reasoning Type: adaptive
Scope: multi_topic
Sub-questions: What is LoRA fine-tuning?, How does LoRA reduce trainable parameters?, How do I implement LoRA with a transformer model?

Query: TCP vs UDP which should I use?
Intent: comparative
Reasoning Type: strategic
Scope: multi_topic
Sub-questions: What are the differences between TCP and UDP?, What are the tradeoffs of TCP vs UDP?, When should you choose TCP over UDP and vice versa?

Query: SQL vs NoSQL for a high traffic web app
Intent: comparative
Reasoning Type: strategic
Scope: multi_topic
Sub-questions: What are the differences between SQL and NoSQL databases?, How does each perform under high traffic?, Which should you choose based on use case?

Query: multiprocessing vs multithreading in Python
Intent: comparative
Reasoning Type: strategic
Scope: multi_topic
Sub-questions: What is the difference between multiprocessing and multithreading in Python?, What are the tradeoffs of each approach?, When should you use multiprocessing vs multithreading?

Now classify the following query. Return ONLY the structured format below, nothing else.

Query: {query}
Intent: <one of factual, procedural, comparative, conceptual, opinion, debugging>
Reasoning Type: <one of commonsense, adaptive, strategic>
Scope: <one of single_topic, multi_topic>
Sub-questions: <1-3 focused sub-questions separated by commas>"""


VALID_REASONING_TYPES = {"commonsense", "adaptive", "strategic"}
VALID_INTENTS = {"factual", "procedural", "comparative", "conceptual", "opinion", "debugging"}

# Keyword-based fallback rules applied BEFORE model output is trusted
STRATEGIC_KEYWORDS = [
    " vs ", " versus ", "compare", "comparison", "difference between",
    "which is better", "should i use", "pros and cons", "tradeoff",
    "when to use", "sql or nosql", "tcp or udp",
]
ADAPTIVE_KEYWORDS = [
    "and how", "and when", "and why", "explain and", "what is.*and",
    "how does.*work", "when should i", "what are the.*and",
]


def _keyword_fallback(query: str) -> Optional[str]:
    """Return a reasoning_type override based on simple keyword rules, or None."""
    q = query.lower()
    for kw in STRATEGIC_KEYWORDS:
        if kw in q:
            return "strategic"
    # Check adaptive patterns (simple substring, not full regex)
    adaptive_triggers = ["and how", "and when", "and why", "how does", "when should"]
    for kw in adaptive_triggers:
        if kw in q:
            return "adaptive"
    return None


class QueryClassifier:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"Loading local LLM for classification: {model_name}...")
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Device set to use {device}")
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=256,
            device=device,
        )
        self.llm    = HuggingFacePipeline(pipeline=hf_pipeline)
        self.prompt = PromptTemplate(
            template=CLASSIFIER_PROMPT,
            input_variables=["query"],
        )
        self.chain  = self.prompt | self.llm

    def classify(self, query: str) -> dict:
        try:
            response = self.chain.invoke({"query": query})

            # Normalise output -- may be str or dict depending on LangChain version
            if isinstance(response, dict):
                response = response.get("text", str(response))
            response = response.strip()

            # Default parsed result
            parsed = {
                "intent":         "factual",
                "reasoning_type": "commonsense",
                "entities":       [],
                "scope":          "single_topic",
                "ambiguity":      "low",
                "sub_questions":  [query],
            }

            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue

                key, _, value = line.partition(":")
                key   = key.strip().lower()
                value = value.strip().lower()

                if key == "intent" and value in VALID_INTENTS:
                    parsed["intent"] = value

                elif key == "reasoning type":
                    # Accept only known values; strip noise like extra words
                    for rt in VALID_REASONING_TYPES:
                        if rt in value:
                            parsed["reasoning_type"] = rt
                            break

                elif key == "scope":
                    if "multi" in value:
                        parsed["scope"] = "multi_topic"
                    else:
                        parsed["scope"] = "single_topic"

                elif key == "sub-questions":
                    raw_sqs = line.split(":", 1)[1].strip()
                    if raw_sqs:
                        sqs = [sq.strip() for sq in raw_sqs.split(",") if sq.strip()]
                        if sqs:
                            parsed["sub_questions"] = sqs

            # --- Keyword-based safety override ---
            # If the model returned commonsense but the query contains strong
            # comparative or multi-part signals, override with the correct type.
            keyword_type = _keyword_fallback(query)
            if keyword_type and parsed["reasoning_type"] == "commonsense":
                parsed["reasoning_type"] = keyword_type
                parsed["scope"] = "multi_topic"
                # If sub_questions was not decomposed, generate basic ones
                if len(parsed["sub_questions"]) == 1:
                    parsed["sub_questions"] = _generate_fallback_subquestions(
                        query, keyword_type
                    )

            return parsed

        except Exception as e:
            print(f"Classification failed: {e}")
            # Even on failure, try keyword fallback before returning pure default
            keyword_type = _keyword_fallback(query) or "commonsense"
            return {
                "intent":         "factual",
                "reasoning_type": keyword_type,
                "entities":       [],
                "scope":          "multi_topic" if keyword_type != "commonsense" else "single_topic",
                "ambiguity":      "low",
                "sub_questions":  [query],
            }


def _generate_fallback_subquestions(query: str, reasoning_type: str) -> list:
    """Generate basic sub-questions when the model failed to decompose the query."""
    q = query.strip().rstrip("?")
    if reasoning_type == "strategic":
        return [
            f"What are the key differences relevant to: {q}?",
            f"What are the tradeoffs for each option in: {q}?",
            f"What is the recommended choice and why for: {q}?",
        ]
    elif reasoning_type == "adaptive":
        return [
            f"What is the core concept in: {q}?",
            f"How does it work in practice: {q}?",
            f"When and why should you use it: {q}?",
        ]
    return [query]
