"""
train_mlx.py -- LoRA fine-tuning for Gemma-2-2B-IT using Apple MLX
====================================================================
Optimised for M1/M2/M3/M4 Apple Silicon Macs.
Uses the Neural Engine via MLX -- 3-5x faster than PyTorch MPS.

Expected time on M4 16GB:
  ~3-5 hours for 1000 iterations (batch=1)

Usage:
  python3 src/train_mlx.py

Output:
  outputs/gemma-2-2b-it-mlx-lora/  -- LoRA adapter weights (~50-100 MB)

Requirements:
  pip3 install mlx mlx-lm
"""

import subprocess
import sys
import os
import json
from pathlib import Path


MODEL_NAME   = "google/gemma-2-2b-it"
DATA_DIR     = "data/mlx_data"
OUTPUT_DIR   = "outputs/gemma-2-2b-it-mlx-lora"
ITERS        = 1000
BATCH_SIZE   = 1      # reduced from 4 -- prevents Metal OOM on M4 16GB
LEARN_RATE   = 2e-4
NUM_LAYERS   = 4      # reduced from 8 -- fewer LoRA layers = less GPU memory
VAL_BATCHES  = 10     # reduced from 25 -- shorter eval pass


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
    MLX-LM expects train.jsonl / valid.jsonl / test.jsonl with a 'text' key.
    Split: 80 / 10 / 10.
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
                records.append({"text": record["text"]})

    total     = len(records)
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

    # --grad-checkpoint: trades compute for memory (essential for 16GB M4)
    # --num-layers 4: only 4 LoRA layers instead of 8 -- halves gradient memory
    # --batch-size 1: single sample per step -- prevents Metal OOM
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model",         MODEL_NAME,
        "--data",          DATA_DIR,
        "--train",
        "--grad-checkpoint",
        "--batch-size",    str(BATCH_SIZE),
        "--iters",         str(ITERS),
        "--learning-rate", str(LEARN_RATE),
        "--num-layers",    str(NUM_LAYERS),
        "--val-batches",   str(VAL_BATCHES),
        "--adapter-path",  OUTPUT_DIR,
        "--save-every",    "200",
    ]

    print("\nStarting MLX LoRA fine-tuning ...")
    print("Command:", " ".join(cmd))
    print("-" * 60)
    subprocess.run(cmd, check=True)


def fuse_adapter():
    fused_dir = "outputs/gemma-2-2b-it-fused"
    Path(fused_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
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
    print(" Reasoning-RAG -- MLX Fine-tuning (M4 Mac, Gemma-2-2B-IT)")
    print("=" * 60)

    if not args.skip_data_prep:
        print("\nStep 1: Converting dataset to MLX format ...")
        convert_to_mlx_format()
    else:
        print("\nStep 1: Skipping data prep (--skip-data-prep set)")

    print("\nStep 2: Fine-tuning Gemma-2-2B-IT with LoRA ...")
    run_training()

    if args.fuse:
        print("\nStep 3: Fusing adapter ...")
        fuse_adapter()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA adapter saved to: {OUTPUT_DIR}")
    print("\nTo run inference with fine-tuned model:")
    print("  python3 src/demo.py --adapter outputs/gemma-2-2b-it-mlx-lora")
    print("=" * 60)
