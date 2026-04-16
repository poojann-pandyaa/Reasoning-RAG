"""
train_mlx.py -- LoRA fine-tuning for Gemma-2B-IT using Apple MLX
=================================================================
Optimised for M1/M2/M3/M4 Apple Silicon Macs.
Uses the Neural Engine via MLX -- 3-5x faster than PyTorch MPS.

Expected time on M4 16GB:
  ~45-90 minutes for 1000 iterations (roughly 3 epochs on 900 examples)

Usage:
  python3 src/train_mlx.py

Output:
  outputs/gemma-2b-it-mlx-lora/  -- LoRA adapter weights (~50-100 MB)

Requirements:
  pip3 install mlx mlx-lm
"""

import subprocess
import sys
import os
import json
from pathlib import Path


MODEL_NAME   = "google/gemma-2b-it"
DATA_DIR     = "data/mlx_data"
OUTPUT_DIR   = "outputs/gemma-2b-it-mlx-lora"
ITERS        = 1000    # ~3 epochs on 900 train examples with batch=4
BATCH_SIZE   = 4       # safe for M4 16GB
LEARN_RATE   = 2e-4
LORA_LAYERS  = 8       # number of transformer layers to apply LoRA
VAL_BATCHES  = 25


def check_mlx():
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("MLX not found. Installing mlx and mlx-lm ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mlx", "mlx-lm"])
        print("Installed. Please re-run this script.")
        sys.exit(0)


def convert_to_mlx_format(src: str = "data/finetune_dataset.jsonl"):
    """
    MLX-LM expects a data/ directory with train.jsonl, valid.jsonl, test.jsonl.
    Each line must have a single "text" key with the full formatted string.
    We split 80/10/10 from finetune_dataset.jsonl.
    """
    src_path = Path(src)
    if not src_path.exists():
        print(f"ERROR: {src} not found.")
        print("Run first: python3 src/ingestion/prepare_finetune.py")
        sys.exit(1)

    records = []
    with src_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                # MLX only needs the 'text' field
                records.append({"text": record["text"]})

    total = len(records)
    train_end = int(total * 0.80)
    val_end   = int(total * 0.90)

    splits = {
        "train": records[:train_end],
        "valid": records[train_end:val_end],
        "test":  records[val_end:],
    }

    out_dir = Path(DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_records in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with out_path.open("w") as f:
            for r in split_records:
                f.write(json.dumps(r) + "\n")
        print(f"  {split_name}: {len(split_records)} examples -> {out_path}")

    print(f"MLX data ready in {DATA_DIR}/")


def run_training():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model",        MODEL_NAME,
        "--data",         DATA_DIR,
        "--train",
        "--batch-size",   str(BATCH_SIZE),
        "--iters",        str(ITERS),
        "--learning-rate",str(LEARN_RATE),
        "--lora-layers",  str(LORA_LAYERS),
        "--val-batches",  str(VAL_BATCHES),
        "--adapter-path", OUTPUT_DIR,
        "--save-every",   "200",       # checkpoint every 200 iters
    ]

    print("\nStarting MLX LoRA fine-tuning ...")
    print("Command:", " ".join(cmd))
    print("-" * 60)
    subprocess.run(cmd, check=True)


def fuse_adapter():
    """
    Optional: fuse the LoRA adapter back into the base model weights.
    Creates a standalone model in outputs/gemma-2b-it-fused/
    Useful if you want a single model file without adapter loading overhead.
    """
    fused_dir = "outputs/gemma-2b-it-fused"
    Path(fused_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model",        MODEL_NAME,
        "--adapter-path", OUTPUT_DIR,
        "--save-path",    fused_dir,
    ]
    print("\nFusing LoRA adapter into base model ...")
    subprocess.run(cmd, check=True)
    print(f"Fused model saved to: {fused_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip data conversion (if already done)")
    parser.add_argument("--fuse", action="store_true",
                        help="Fuse adapter into base model after training")
    args = parser.parse_args()

    check_mlx()

    print("=" * 60)
    print(" Reasoning-RAG -- MLX Fine-tuning (M4 Mac)")
    print("=" * 60)

    if not args.skip_data_prep:
        print("\nStep 1: Converting dataset to MLX format ...")
        convert_to_mlx_format()
    else:
        print("\nStep 1: Skipping data prep (--skip-data-prep set)")

    print("\nStep 2: Fine-tuning Gemma-2B-IT with LoRA ...")
    run_training()

    if args.fuse:
        print("\nStep 3: Fusing adapter ...")
        fuse_adapter()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA adapter saved to: {OUTPUT_DIR}")
    print("\nTo run inference with fine-tuned model:")
    print("  python3 src/demo.py --adapter outputs/gemma-2b-it-mlx-lora")
    print("=" * 60)
