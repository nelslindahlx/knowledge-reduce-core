#!/usr/bin/env python3
"""
LoRA SFT training script for ModelReduce shards on Apple Silicon (MLX-Based).

This script orchestrates fine-tuning using Apple's MLX Framework (mlx-lm).
It validates prepared chat-JSONL datasets, formats them, handles validation
splits if present, and invokes MLX-LM LoRA training.

USAGE
    python scripts/train_mlx.py \
        --dataset train.jsonl \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --iters 600 --batch-size 4 --lr 1e-5
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Dict, Any


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apple Silicon LoRA SFT on a ModelReduce shard.")
    p.add_argument("--dataset", required=True, help="Prepared chat-JSONL dataset.")
    p.add_argument("--base-model", default="mlx-community/Qwen2.5-3B-Instruct-4bit",
                   help="Hugging Face ID or local path to base model / mlx-quantized model.")
    p.add_argument("--adapter-path", default="./adapters",
                   help="Directory to save the resulting LoRA adapters (default: ./adapters).")
    p.add_argument("--iters", type=int, default=600,
                   help="Number of training iterations (default: 600).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Training batch size (default: 4).")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate (default: 1e-5).")
    p.add_argument("--lora-layers", type=int, default=16,
                   help="Number of layers to apply LoRA to (default: 16).")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate dataset + print the command, then exit (no training).")
    return p


def _load_and_validate_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
        
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {idx} in {path} is not valid JSON: {exc}")
                
            if "messages" not in data:
                raise ValueError(f"Line {idx} in {path} is missing 'messages' key.")
                
            messages = data["messages"]
            if not isinstance(messages, list):
                raise ValueError(f"Line {idx} in {path} 'messages' must be a list.")
                
            for msg_idx, msg in enumerate(messages):
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValueError(
                        f"Line {idx} in {path}, message {msg_idx} must contain "
                        f"'role' and 'content' keys."
                    )
            rows.append(data)
    return rows


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    print("=" * 60)
    print("🚀 KNOWLEDGEREDUCE LOCAL MLX TRAINING SYSTEM")
    print("=" * 60)

    # 1. Load and validate train dataset
    try:
        train_rows = _load_and_validate_dataset(args.dataset)
        print(f"[*] Validated training dataset: {len(train_rows)} records from {args.dataset}")
    except Exception as exc:
        print(f"Error validating dataset: {exc}", file=sys.stderr)
        return 1

    # 2. Check for optional validation split
    val_path = args.dataset + ".val"
    has_validation = False
    val_rows = []
    if os.path.exists(val_path):
        try:
            val_rows = _load_and_validate_dataset(val_path)
            print(f"[*] Validated validation split: {len(val_rows)} records from {val_path}")
            has_validation = True
        except Exception as exc:
            print(f"Warning: validation split '{val_path}' exists but failed validation: {exc}", file=sys.stderr)

    # 3. Handle dry-run mode
    if args.dry_run:
        print("\n[dry-run] Verification complete! The dataset is valid.")
        print("[dry-run] If running, the system will execute:")
        print(f"  python3 -m mlx_lm.lora \\")
        print(f"    --model {args.base_model} \\")
        print(f"    --data <temp_data_dir> \\")
        print(f"    --train \\")
        print(f"    --iters {args.iters} \\")
        print(f"    --batch-size {args.batch_size} \\")
        print(f"    --learning-rate {args.lr} \\")
        print(f"    --lora-layers {args.lora_layers} \\")
        print(f"    --adapter-path {args.adapter_path}")
        print("=" * 60)
        return 0

    # 4. Verify dependencies
    try:
        import mlx.core as mx  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        print(f"Error: MLX training libraries not installed: {exc}\n"
              f"Please install via: pip install mlx-lm", file=sys.stderr)
        return 3

    # 5. Prepare temporary directory structure for mlx_lm.lora
    temp_dir = tempfile.mkdtemp(prefix="mlx_sft_")
    try:
        # Write train.jsonl
        train_temp = os.path.join(temp_dir, "train.jsonl")
        with open(train_temp, "w", encoding="utf-8") as fh:
            for r in train_rows:
                fh.write(json.dumps(r) + "\n")
                
        # Write valid.jsonl if present
        if has_validation:
            val_temp = os.path.join(temp_dir, "valid.jsonl")
            with open(val_temp, "w", encoding="utf-8") as fh:
                for r in val_rows:
                    fh.write(json.dumps(r) + "\n")

        # 6. Execute MLX-LM fine-tuning
        print(f"\n[*] Starting MLX fine-tuning...")
        print(f"    Base Model:   {args.base_model}")
        print(f"    Iterations:   {args.iters}")
        print(f"    Batch Size:   {args.batch_size}")
        print(f"    Adapter Path: {args.adapter_path}")
        print("-" * 60)

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", args.base_model,
            "--data", temp_dir,
            "--train",
            "--iters", str(args.iters),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.lr),
            "--lora-layers", str(args.lora_layers),
            "--adapter-path", args.adapter_path
        ]
        
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("-" * 60)
            print(f"🎉 MLX fine-tuning completed successfully!")
            print(f"[*] LoRA adapters saved to: {args.adapter_path}")
        else:
            print("-" * 60)
            print(f"Error: MLX training command exited with code {res.returncode}", file=sys.stderr)
        return res.returncode

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
