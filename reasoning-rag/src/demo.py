import sys
import os
import argparse

from reasoning.classifier import QueryClassifier
from reasoning.engine import ReasoningEngine
from generation.trace import ReasoningTrace


def parse_args():
    parser = argparse.ArgumentParser(description="Reasoning-RAG Interactive Demo")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (e.g. outputs/gemma-2b-it-mlx-lora). "
             "If not provided, uses the base Gemma-2B-IT model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2b-it",
        help="HuggingFace model name or local path (default: google/gemma-2b-it)"
    )
    return parser.parse_args()


def run_demo():
    args = parse_args()

    print("Loading Models and Indices... This might take a moment.")
    if args.adapter:
        print(f"Using LoRA adapter: {args.adapter}")
    else:
        print("No adapter specified -- using base model.")

    try:
        classifier = QueryClassifier()
        engine     = ReasoningEngine(
            model_name=args.model,
            lora_adapter_path=args.adapter,
        )
    except Exception as e:
        print(f"Failed to initialize system. Error: {e}")
        return

    print("\nType your question below (or 'quit' to exit):")
    while True:
        try:
            query = input("\nQuery: ").strip()
        except EOFError:
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            break

        print("\n" + "=" * 60)
        print(f"=== Query ===\n{query}\n")

        # Phase 1: Classification
        classification  = classifier.classify(query)
        intent          = classification.get('intent', 'unknown')
        r_type          = classification.get('reasoning_type', 'commonsense')
        scope           = classification.get('scope', 'unknown')
        sub_questions   = classification.get('sub_questions', [])

        print(f"=== Classification ===")
        print(f"Intent: {intent} | Reasoning type: {r_type} | Scope: {scope}\n")
        print("=== Sub-queries issued ===")
        for i, sq in enumerate(sub_questions):
            print(f"{i+1}. {sq}")
        print()

        # Phase 2: Engine Execution
        trace = ReasoningTrace(query)
        trace.classification = classification
        trace = engine.execute(trace)

        # Output
        print("=== Retrieved Sources (Top 3) ===")
        for i, cand in enumerate(trace.reranked_final[:3]):
            meta = cand['metadata']
            print(f"[Source {i+1}] Score: {meta.get('score', 0)} | "
                  f"Accepted: {meta.get('is_accepted', False)} | "
                  f"Domain: {meta.get('domain', 'unknown')}")
            print(f"{meta.get('chunk_text', '')[:150]}...")
            print("-")

        print("\n=== Final Answer ===")
        print(trace.final_answer)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
