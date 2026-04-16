"""
train.py  —  QLoRA fine-tuning for google/gemma-2b-it
=======================================================
Trains a LoRA adapter on top of the frozen Gemma-2B-IT base model
using your StackOverflow instruction dataset.

Output:
  A small LoRA adapter saved to  outputs/gemma-2b-it-lora/
  (only ~50-100 MB — not the full 5 GB model).

Usage:
  python src/train.py \
      --dataset  data/finetune_dataset.jsonl \
      --output   outputs/gemma-2b-it-lora \
      --epochs   3

Requirements (already in requirements.txt):
  transformers>=4.40  peft>=0.10  trl>=0.8  bitsandbytes>=0.43  accelerate>=0.27
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
MODEL_NAME     = "google/gemma-2b-it"
MAX_SEQ_LENGTH = 1024   # safe for Gemma-2B on 8-16 GB RAM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/finetune_dataset.jsonl")
    p.add_argument("--output",  default="outputs/gemma-2b-it-lora")
    p.add_argument("--epochs",  type=int,   default=3)
    p.add_argument("--batch",   type=int,   default=2,    help="Per-device train batch size")
    p.add_argument("--grad_accum", type=int, default=4,   help="Gradient accumulation steps")
    p.add_argument("--lr",      type=float, default=2e-4, help="Learning rate")
    p.add_argument("--lora_r",  type=int,   default=16,   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32,  help="LoRA alpha")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model(device: str):
    """Load Gemma-2B-IT with 4-bit quantization (CUDA) or fp32 (MPS/CPU)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for SFT packing

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("[WARNING] CUDA not available. Fine-tuning on MPS/CPU will be very slow.")
        print("          Consider using Google Colab (free T4 GPU) for training.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"" : device},
            trust_remote_code=True,
        )

    return model, tokenizer


def build_lora_config(r: int, alpha: int) -> LoraConfig:
    """
    Target the attention projection layers of Gemma.
    q_proj / k_proj / v_proj / o_proj are the standard LoRA targets.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def main():
    args   = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # ---- 1. Dataset ----
    print(f"Loading dataset from {args.dataset} ...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    # 90 / 10 train-eval split
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ---- 2. Model ----
    model, tokenizer = load_base_model(device)
    lora_cfg         = build_lora_config(args.lora_r, args.lora_alpha)
    model            = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- 3. Training arguments ----
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir                  = args.output,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch,
        per_device_eval_batch_size  = args.batch,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        bf16                        = use_bf16,
        fp16                        = (not use_bf16 and device == "cuda"),
        logging_steps               = 20,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        report_to                   = "wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name                    = "gemma-2b-it-reasoning-rag",
        optim                       = "paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        dataloader_num_workers      = 0,
    )

    # ---- 4. Trainer ----
    # SFTTrainer automatically tokenises the 'text' column (the full Gemma chat string)
    trainer = SFTTrainer(
        model            = model,
        tokenizer        = tokenizer,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        dataset_text_field = "text",     # uses the pre-formatted Gemma chat string
        max_seq_length   = MAX_SEQ_LENGTH,
        args             = training_args,
        peft_config      = lora_cfg,
    )

    # ---- 5. Train ----
    print("Starting fine-tuning ...")
    trainer.train()

    # ---- 6. Save adapter only ----
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nLoRA adapter saved to: {args.output}")
    print("\nTo use the fine-tuned model, pass lora_adapter_path to FinalGenerator:")
    print(f"  FinalGenerator(lora_adapter_path='{args.output}')")


if __name__ == "__main__":
    main()
