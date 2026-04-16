"""
prepare_finetune.py
-------------------
Converts processed_dataset.jsonl  →  finetune_dataset.jsonl

Each output record is a Gemma-IT instruction-following example:
{
  "instruction": "Answer this Stack Overflow question:",
  "input":  "<question text>",
  "output": "<best answer text>",
  "text":   "<full Gemma chat-formatted string>"   # used directly by SFTTrainer
}

Usage:
  python src/ingestion/prepare_finetune.py \
      --input  data/processed_dataset.jsonl \
      --output data/finetune_dataset.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def pick_best_answer(answers: list) -> dict | None:
    """Return the accepted answer if present, otherwise the highest-scored one."""
    if not answers:
        return None
    accepted = [a for a in answers if a.get("is_accepted") or a.get("selected")]
    if accepted:
        return max(accepted, key=lambda a: a.get("score", a.get("pm_score", 0)))
    return max(answers, key=lambda a: a.get("score", a.get("pm_score", 0)))


def format_gemma_chat(question: str, answer: str) -> str:
    """Wrap in Gemma-IT <start_of_turn> chat template."""
    return (
        "<start_of_turn>user\n"
        f"You are a helpful software engineering assistant.\n\n"
        f"Question: {question}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{answer}\n"
        "<end_of_turn>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/processed_dataset.jsonl")
    parser.add_argument("--output", default="data/finetune_dataset.jsonl")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0

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

            best = pick_best_answer(record.get("answers", []))
            if not best:
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

    print(f"Done. Written: {written} | Skipped: {skipped}")
    print(f"Output → {output_path}")


if __name__ == "__main__":
    main()
