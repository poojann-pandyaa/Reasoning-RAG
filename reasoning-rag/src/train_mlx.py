"""
train_mlx.py -- LoRA fine-tuning for Gemma-2-2B-IT using Apple MLX
====================================================================
Optimised for M1/M2/M3/M4 Apple Silicon Macs.
Uses the Neural Engine via MLX -- 3-5x faster than PyTorch MPS.

Expected time on M4 16GB:
  v1 (1000 iters): ~3-5 hours
  v2 (2000 iters): ~6-8 hours  <-- recommended, run overnight

Usage (v2 -- recommended, accepted answers only):
  python3 src/train_mlx.py --v2

Usage (v1 -- original run, kept for reproducibility):
  python3 src/train_mlx.py

Output:
  v1: outputs/gemma-2-2b-it-mlx-lora/
  v2: outputs/gemma-2-2b-it-mlx-lora-v2/

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

# v1 config (original)
V1_OUTPUT_DIR  = "outputs/gemma-2-2b-it-mlx-lora"
V1_ITERS       = 1000
V1_BATCH_SIZE  = 1
V1_LEARN_RATE  = 2e-4
V1_NUM_LAYERS  = 4
V1_VAL_BATCHES = 10

# v2 config -- accepted-only data, more iters, lower LR, more LoRA layers
V2_OUTPUT_DIR  = "outputs/gemma-2-2b-it-mlx-lora-v2"
V2_ITERS       = 2000
V2_BATCH_SIZE  = 1
V2_LEARN_RATE  = 1e-5
V2_NUM_LAYERS  = 8
V2_VAL_BATCHES = 25


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


def run_training(output_dir, iters, batch_size, learn_rate, num_layers, val_batches):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model",         MODEL_NAME,
        "--data",          DATA_DIR,
        "--train",
        "--grad-checkpoint",
        "--batch-size",    str(batch_size),
        "--iters",         str(iters),
        "--learning-rate", str(learn_rate),
        "--num-layers",    str(num_layers),
        "--val-batches",   str(val_batches),
        "--adapter-path",  output_dir,
        "--save-every",    "200",
    ]

    print("\nStarting MLX LoRA fine-tuning ...")
    print("Command:", " ".join(cmd))
    print("-" * 60)
    subprocess.run(cmd, check=True)


def fuse_adapter(output_dir):
    fused_dir = output_dir + "-fused"
    Path(fused_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model",        MODEL_NAME,
        "--adapter-path", output_dir,
        "--save-path",    fused_dir,
    ]
    print("\nFusing LoRA adapter into base model ...")
    subprocess.run(cmd, check=True)
    print(f"Fused model saved to: {fused_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Train v2: accepted-only data, 2000 iters, lr=1e-5, 8 LoRA layers. Run overnight."
    )
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip data conversion (if already done)")
    parser.add_argument("--fuse", action="store_true",
                        help="Fuse adapter into base model after training")
    args = parser.parse_args()

    check_mlx()

    print("=" * 60)
    print(" Reasoning-RAG -- MLX Fine-tuning (M4 Mac, Gemma-2-2B-IT)")
    print("=" * 60)

    if args.v2:
        output_dir  = V2_OUTPUT_DIR
        iters       = V2_ITERS
        batch_size  = V2_BATCH_SIZE
        learn_rate  = V2_LEARN_RATE
        num_layers  = V2_NUM_LAYERS
        val_batches = V2_VAL_BATCHES
        data_src    = "data/finetune_dataset.jsonl"
        print("\n[v2] Config: 2000 iters | lr=1e-5 | 8 LoRA layers | accepted-only data")
        print(f"[v2] Output: {output_dir}")

        if not args.skip_data_prep:
            print("\nStep 1: Rebuilding dataset (accepted answers only) ...")
            subprocess.run([
                sys.executable,
                "src/ingestion/prepare_finetune.py",
                "--input",  "data/processed_dataset.jsonl",
                "--output", data_src,
                "--accepted-only",
            ], check=True)
            print("\nStep 2: Converting to MLX format ...")
            convert_to_mlx_format(src=data_src)
        else:
            print("\nStep 1+2: Skipping data prep (--skip-data-prep set)")
    else:
        output_dir  = V1_OUTPUT_DIR
        iters       = V1_ITERS
        batch_size  = V1_BATCH_SIZE
        learn_rate  = V1_LEARN_RATE
        num_layers  = V1_NUM_LAYERS
        val_batches = V1_VAL_BATCHES
        print("\n[v1] Config: 1000 iters | lr=2e-4 | 4 LoRA layers | all answers")
        print(f"[v1] Output: {output_dir}")

        if not args.skip_data_prep:
            print("\nStep 1: Converting dataset to MLX format ...")
            convert_to_mlx_format()
        else:
            print("\nStep 1: Skipping data prep (--skip-data-prep set)")

    print("\nStep 3: Fine-tuning Gemma-2-2B-IT with LoRA ...")
    run_training(output_dir, iters, batch_size, learn_rate, num_layers, val_batches)

    if args.fuse:
        print("\nStep 4: Fusing adapter ...")
        fuse_adapter(output_dir)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA adapter saved to: {output_dir}")
    print("\nTo compare base vs fine-tuned:")
    print(f"  python3 src/evaluation/compare_demo.py --adapter {output_dir}")
    print("=" * 60)
