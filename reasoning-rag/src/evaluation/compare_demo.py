"""
compare_demo.py — Visual before/after comparison: Base vs Fine-tuned Gemma-2-2B-IT
====================================================================================
Runs 5 Stack Overflow-style queries through both models and prints answers
side by side so you can clearly see what fine-tuning improved.

Usage (run AFTER training completes):
    cd reasoning-rag
    python3 src/evaluation/compare_demo.py --adapter outputs/gemma-2-2b-it-mlx-lora

What to look for in fine-tuned answers vs base:
  - More direct, Stack Overflow-style tone
  - Proper code blocks with correct syntax
  - Structured: explanation → example → note
  - Fewer filler phrases ("Great question!", "Of course!", etc.)
"""

import argparse
import sys
import os
import time

# Make sure src/ is on the path when running from reasoning-rag/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reasoning.engine import ReasoningEngine
from reasoning.classifier import QueryClassifier
from generation.trace import ReasoningTrace

# ── 5 representative Stack Overflow-style queries ─────────────────────────────
QUERIES = [
    "How do I reverse a list in Python?",
    "What is the difference between == and is in Python?",
    "How do I fix a segmentation fault in C?",
    "What does git rebase do and when should I use it?",
    "How do I use async/await in Python with an example?",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def run_query(engine: ReasoningEngine, classifier: QueryClassifier, question: str) -> tuple:
    """Run one query through the full RAG pipeline. Returns (answer, latency_s)."""
    t0 = time.time()
    trace = ReasoningTrace(question)
    trace.classification = classifier.classify(question)
    trace = engine.execute(trace)
    return trace.final_answer.strip(), round(time.time() - t0, 1)


def print_banner(text: str, width: int = 65):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_side_by_side(q_num: int, question: str,
                        base_ans: str, base_t: float,
                        ft_ans: str, ft_t: float):
    sep = "-" * 65
    print(f"\n{'='*65}")
    print(f"  Q{q_num}: {question}")
    print(f"{'='*65}")

    print(f"\n── BASE MODEL  ({base_t}s) ──────────────────────────────────")
    print(base_ans if base_ans else "(no answer returned)")

    print(f"\n── FINE-TUNED  ({ft_t}s) ──────────────────────────────────")
    print(ft_ans if ft_ans else "(no answer returned)")
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison: Base vs Fine-tuned Gemma-2-2B-IT"
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to LoRA adapter directory, e.g. outputs/gemma-2-2b-it-mlx-lora"
    )
    parser.add_argument(
        "--model",
        default="google/gemma-2-2b-it",
        help="Base model name (default: google/gemma-2-2b-it)"
    )
    args = parser.parse_args()

    # ── load shared classifier (same for both models) ─────────────────────────
    print_banner("Reasoning-RAG  |  Base vs Fine-tuned Comparison")
    print("\nLoading QueryClassifier (shared)...")
    classifier = QueryClassifier()

    # ── load base model ───────────────────────────────────────────────────────
    print("Loading BASE model...")
    base_engine = ReasoningEngine(model_name=args.model)

    # ── load fine-tuned model ─────────────────────────────────────────────────
    print(f"Loading FINE-TUNED model (adapter: {args.adapter})...")
    ft_engine = ReasoningEngine(model_name=args.model,
                                lora_adapter_path=args.adapter)

    # ── run all queries ───────────────────────────────────────────────────────
    print_banner(f"Running {len(QUERIES)} queries — please wait...")

    base_answers, ft_answers = [], []

    for i, q in enumerate(QUERIES):
        print(f"\n[{i+1}/{len(QUERIES)}] {q}")

        print("  → base model...", end=" ", flush=True)
        b_ans, b_t = run_query(base_engine, classifier, q)
        print(f"done ({b_t}s)")

        print("  → fine-tuned...", end=" ", flush=True)
        f_ans, f_t = run_query(ft_engine, classifier, q)
        print(f"done ({f_t}s)")

        base_answers.append((b_ans, b_t))
        ft_answers.append((f_ans, f_t))

    # ── print results ─────────────────────────────────────────────────────────
    print_banner("RESULTS")

    for i, q in enumerate(QUERIES):
        b_ans, b_t = base_answers[i]
        f_ans, f_t = ft_answers[i]
        print_side_by_side(i + 1, q, b_ans, b_t, f_ans, f_t)

    # ── quick summary ─────────────────────────────────────────────────────────
    avg_base_len = sum(len(b.split()) for b, _ in base_answers) / len(base_answers)
    avg_ft_len   = sum(len(f.split()) for f, _ in ft_answers)   / len(ft_answers)
    avg_base_t   = sum(t for _, t in base_answers) / len(base_answers)
    avg_ft_t     = sum(t for _, t in ft_answers)   / len(ft_answers)

    print_banner("SUMMARY")
    print(f"{'Metric':<30} {'Base':>10} {'Fine-tuned':>12}")
    print("-" * 55)
    print(f"{'Avg answer length (words)':<30} {avg_base_len:>10.1f} {avg_ft_len:>12.1f}")
    print(f"{'Avg latency (s)':<30} {avg_base_t:>10.1f} {avg_ft_t:>12.1f}")
    print()
    print("Tip: look for more code blocks, direct tone, and")
    print("     Stack Overflow-style structure in fine-tuned answers.")


if __name__ == "__main__":
    main()
