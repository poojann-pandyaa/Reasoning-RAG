import os
import torch
from typing import Optional
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

CLASSIFIER_PROMPT = """You are a query analysis expert. Given a user query, analyze it and return exactly the following structured format. Do not write anything else.

Intent: <one of factual, procedural, comparative, conceptual, opinion, debugging>
Reasoning Type: <one of commonsense, adaptive, strategic>
Scope: <one of single_topic, multi_topic, hierarchical>
Ambiguity: <one of low, medium, high>
Sub-questions: <write 1-3 smaller questions needed to address the query, separated by commas>

Query: {query}
Analysis:"""


class QueryClassifier:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"Loading local LLM for classification: {model_name}...")
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
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
        # Modern LangChain: use pipe operator instead of deprecated LLMChain
        self.chain  = self.prompt | self.llm

    def classify(self, query: str) -> dict:
        try:
            response = self.chain.invoke({"query": query})

            # Normalise output -- may be str or dict depending on LangChain version
            if isinstance(response, dict):
                response = response.get("text", str(response))
            response = response.strip()

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
                if line.lower().startswith("intent:"):
                    parsed["intent"] = line.split(":", 1)[1].strip().lower()
                elif line.lower().startswith("reasoning type:"):
                    parsed["reasoning_type"] = line.split(":", 1)[1].strip().lower()
                elif line.lower().startswith("scope:"):
                    parsed["scope"] = line.split(":", 1)[1].strip().lower()
                elif line.lower().startswith("ambiguity:"):
                    parsed["ambiguity"] = line.split(":", 1)[1].strip().lower()
                elif line.lower().startswith("sub-questions:"):
                    sqs = line.split(":", 1)[1].strip()
                    if sqs:
                        parsed["sub_questions"] = [
                            sq.strip() for sq in sqs.split(",") if sq.strip()
                        ]

            return parsed

        except Exception as e:
            print(f"Classification failed: {e}")
            return {
                "intent":         "factual",
                "reasoning_type": "commonsense",
                "entities":       [],
                "scope":          "single_topic",
                "ambiguity":      "low",
                "sub_questions":  [query],
            }
