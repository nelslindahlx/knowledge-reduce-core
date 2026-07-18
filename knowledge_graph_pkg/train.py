import os
import sys
import argparse
from typing import Any, Dict, Optional

class MLXTrainer:
    """Wrapper to run local LoRA fine-tuning using Apple Silicon GPU acceleration (via mlx-lm)."""

    def __init__(self, model_path: str, data_dir: str, adapter_path: str = "adapters"):
        try:
            import mlx_lm.lora as lora
        except ImportError as exc:
            raise ImportError(
                "MLXTrainer requires mlx-lm to be installed: pip install mlx-lm"
            ) from exc

        self.model_path = model_path
        self.data_dir = os.path.abspath(data_dir)
        self.adapter_path = os.path.abspath(adapter_path)

    def train_lora(self, iters: int = 100, batch_size: int = 4, lr: float = 1e-5,
                   num_layers: int = 16) -> None:
        """Execute the mlx-lm LoRA fine-tuning loop."""
        import mlx_lm.lora as lora

        print(f"[MLX] Beginning LoRA training on model {self.model_path}...")
        print(f"[MLX] Training data directory: {self.data_dir}")

        # Ensure directory contains required files
        for split in ("train.jsonl", "valid.jsonl"):
            split_path = os.path.join(self.data_dir, split)
            if not os.path.isfile(split_path):
                raise FileNotFoundError(
                    f"MLX fine-tuning requires {split} in data directory: {self.data_dir}. "
                    f"Please run knowledgereduce model-prep first."
                )

        # Create args namespace matching mlx_lm.lora.py configuration parameters
        args = argparse.Namespace(
            model=self.model_path,
            train=True,
            data=self.data_dir,
            seed=0,
            num_layers=num_layers,
            batch_size=batch_size,
            iters=iters,
            val_batches=10,
            learning_rate=lr,
            steps_per_report=10,
            steps_per_eval=50,
            resume_adapter_file=None,
            adapter_path=self.adapter_path,
            save_every=100,
            test=False,
            test_batches=50,
            max_seq_length=2048,
            config=None,
            grad_checkpoint=True,
            grad_accumulation_steps=1,
            lr_schedule=None,
            lora_parameters={"rank": 8, "dropout": 0.0, "scale": 20.0},
            mask_prompt=False,
            report_to=None,
            project_name=None,
            fine_tune_type="lora",
            optimizer="adamw"
        )

        try:
            lora.run(args)
            print(f"[MLX] LoRA fine-tuning complete. Adapters saved to {self.adapter_path}")
        except Exception as exc:
            print(f"[MLX] Training loop failed: {exc}", file=sys.stderr)
            raise exc
