"""
prepare_finetune.py
-------------------
Converts processed_dataset.jsonl -> finetune_dataset.jsonl

Each output record is a Gemma-IT instruction-following example:
{
  "instruction": "Answer this Stack Overflow question:",
  "input":  "<question text>",
  "output": "<best answer text>",
  "text":   "<full Gemma chat-formatted string>"   # used directly by SFTTrainer
}

Usage (default -- accepted + fallback to highest-scored):
  python3 src/ingestion/prepare_finetune.py \
      --input  data/processed_dataset.jsonl \
      --output data/finetune_dataset.jsonl

Usage (accepted answers only -- recommended for fine-tuning v2):
  python3 src/ingestion/prepare_finetune.py \
      --input  data/processed_dataset.jsonl \
      --output data/finetune_dataset.jsonl \
      --accepted-only
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional


def pick_best_answer(answers: list, accepted_only: bool = False) -> Optional[dict]:
    """Return the accepted answer if present.
    If accepted_only=True, returns None when no accepted answer exists.
    Otherwise falls back to the highest-scored answer.
    """
    if not answers:
        return None
    accepted = [a for a in answers if a.get("is_accepted") or a.get("selected")]
    if accepted:
        return max(accepted, key=lambda a: a.get("score", a.get("pm_score", 0)))
    if accepted_only:
        # No accepted answer -- skip this question entirely
        return None
    # Fallback: highest-scored unaccepted answer
    return max(answers, key=lambda a: a.get("score", a.get("pm_score", 0)))


def format_gemma_chat(question: str, answer: str) -> str:
    """Wrap in Gemma-IT <start_of_turn> chat template."""
    return (
        "<start_of_turn>user\n"
        "You are a helpful software engineering assistant.\n\n"
        f"Question: {question}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{answer}\n"
        "<end_of_turn>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         default="data/processed_dataset.jsonl")
    parser.add_argument("--output",        default="data/finetune_dataset.jsonl")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Only use questions that have an accepted answer. "
             "Skips questions where only unaccepted answers exist. "
             "Recommended for fine-tuning v2 -- produces cleaner training data."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0
    skipped_no_accepted = 0

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            question = (
                record.get("title")
                or record.get("question", "")
            ).strip()
            if not question:
                skipped += 1
                continue

            best = pick_best_answer(
                record.get("answers", []),
                accepted_only=args.accepted_only,
            )
            if not best:
                if args.accepted_only:
                    skipped_no_accepted += 1
                else:
                    skipped += 1
                continue

            answer_text = best.get("body_clean") or best.get("text", "")
            answer_text = answer_text.strip()
            if len(answer_text) < 30:
                skipped += 1
                continue

            out = {
                "instruction": "Answer this Stack Overflow question:",
                "input":  question,
                "output": answer_text,
                "text":   format_gemma_chat(question, answer_text),
            }
            fout.write(json.dumps(out) + "\n")
            written += 1

    print(f"Done. Written: {written} | Skipped: {skipped}", end="")
    if args.accepted_only:
        print(f" | Skipped (no accepted answer): {skipped_no_accepted}")
    else:
        print()
    print(f"Output -> {output_path}")


if __name__ == "__main__":
    main()
