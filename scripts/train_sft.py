#!/usr/bin/env python3
"""
LoRA SFT training script for ModelReduce shards (Session 7).

⚠️  HARDWARE NOTE — READ FIRST
This performs a real LoRA fine-tune of a base model (default Qwen2.5-7B) on a
ModelReduce shard. It requires torch + transformers + peft + trl and a
capable accelerator (a CUDA GPU, or an Apple-silicon machine with adequate
RAM and patience). It is intentionally NOT part of the package's test suite
or CI, and NOT runnable on a fanless laptop in any reasonable time.

It is also intentionally separate from the installable package: the heavy
training deps are not in any extra. Install them yourself when you have the
hardware:

    pip install torch transformers peft trl datasets accelerate

PRECONDITION (enforced by ModelReduce's own quality gates):
Do NOT train on a shard that fails `knowledgereduce model-eval --ci`. As of
the Session 5 calibration, the demo biochem shard FAILS the gates
(hallucination 0.286, agreement not yet a reliable precision signal). Probe
at much larger volume and re-check the gates before spending GPU hours.

USAGE
    python scripts/train_sft.py \
        --dataset train.jsonl \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --output-dir ./lora-out \
        --epochs 1 --batch-size 1 --grad-accum 8 --lr 2e-4

This file is dependency-light at import time (argparse only); heavy imports
happen inside main() so the repo stays importable without the training deps.
"""

import argparse
import json
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA SFT on a ModelReduce shard.")
    p.add_argument("--dataset", required=True, help="Prepared chat-JSONL dataset.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-dir", default="./lora-out")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate dataset + print the plan, then exit (no training).")
    return p


def _load_chat_dataset(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    rows = _load_chat_dataset(args.dataset)
    print(f"Dataset: {len(rows)} chat records from {args.dataset}")
    print(f"Base model: {args.base_model}")
    print(f"LoRA: r={args.lora_r} alpha={args.lora_alpha} | "
          f"epochs={args.epochs} bs={args.batch_size} ga={args.grad_accum} lr={args.lr}")

    if args.dry_run:
        print("[dry-run] dataset valid; exiting before loading training libs.")
        return 0

    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   TrainingArguments)
        from peft import LoraConfig
        from trl import SFTTrainer
    except ImportError as exc:
        print(f"error: training deps not installed ({exc}).\n"
              f"install: pip install torch transformers peft trl datasets accelerate",
              file=sys.stderr)
        return 3

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _format(rec):
        return {"text": tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list(rows).map(_format)

    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype="auto")
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, logging_steps=10, save_strategy="epoch",
        bf16=True, report_to="none")

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=ds,
        peft_config=peft_config, dataset_text_field="text",
        max_seq_length=args.max_seq_len, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
