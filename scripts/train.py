"""Reproducible Transformer LM training loop for tokenized TinyStories data."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from scripts.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, clip_gradient_norm, cross_entropy, get_lr_cosine_schedule


CheckpointMode = Literal["none", "model", "full"]


@dataclass
class TrainConfig:
    train_data: Path
    valid_data: Path
    output_dir: Path
    vocab_size: int = 10_000
    token_dtype: str = "auto"
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1_344
    num_layers: int = 4
    num_heads: int = 16
    rope_theta: float = 10_000.0
    batch_size: int = 32
    total_tokens: int = 40_960_000
    max_steps: int | None = None
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_batches: int = 20
    log_interval: int = 10
    seed: int = 42
    device: str = "auto"
    amp_dtype: str = "none"
    compile_model: bool = False
    checkpoint: CheckpointMode = "none"
    divergence_threshold: float = 100.0
    divergence_factor: float = 3.0
    overwrite: bool = False

    def resolved_max_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        tokens_per_step = self.batch_size * self.context_length
        return max(1, math.ceil(self.total_tokens / tokens_per_step))

    def validate(self) -> None:
        positive_integer_fields = (
            "vocab_size",
            "context_length",
            "d_model",
            "d_ff",
            "num_layers",
            "num_heads",
            "batch_size",
            "total_tokens",
            "eval_interval",
            "eval_batches",
            "log_interval",
        )
        for field_name in positive_integer_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("d_model // num_heads must be even for RoPE")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.warmup_steps > self.resolved_max_steps():
            raise ValueError("warmup_steps cannot exceed the number of training steps")
        if self.amp_dtype not in {"none", "bfloat16"}:
            raise ValueError("amp_dtype must be 'none' or 'bfloat16'")
        if self.checkpoint not in {"none", "model", "full"}:
            raise ValueError("checkpoint must be 'none', 'model', or 'full'")
        if self.divergence_threshold <= 0 or self.divergence_factor <= 1:
            raise ValueError("divergence_threshold must be positive and divergence_factor must exceed 1")


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1)

    def log(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, allow_nan=False) + "\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _jsonable_config(config: TrainConfig) -> dict[str, Any]:
    payload = dataclasses.asdict(config)
    payload["train_data"] = str(config.train_data.resolve())
    payload["valid_data"] = str(config.valid_data.resolve())
    payload["output_dir"] = str(config.output_dir.resolve())
    return payload


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_token_dtype(path: Path, requested_dtype: str) -> np.dtype:
    if requested_dtype != "auto":
        return np.dtype(requested_dtype)
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Cannot infer token dtype because {metadata_path} is missing; pass --token-dtype explicitly"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return np.dtype(metadata["dtype"])


def load_token_data(path: Path, dtype: np.dtype) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size % dtype.itemsize != 0:
        raise ValueError(f"{path} size is not divisible by dtype size {dtype.itemsize}")
    return np.memmap(path, mode="r", dtype=dtype)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "none":
        return contextlib.nullcontext()
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("bfloat16 autocast is supported by this script only on CUDA and CPU")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
    amp_dtype: str,
    seed: int,
) -> float:
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _ in range(eval_batches):
        inputs, targets = get_batch(data, batch_size, context_length, device, rng=rng)
        with _autocast_context(device, amp_dtype):
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
        losses.append(loss.float().item())
    model.train(was_training)
    return float(np.mean(losses))


def _save_final_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
    summary: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state_dict": model.state_dict(),
        "iteration": step,
        "config": _jsonable_config(config),
        "summary": summary,
    }
    if config.checkpoint == "full":
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def run_training(config: TrainConfig) -> dict[str, Any]:
    config.validate()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "config.json"
    checkpoint_path = output_dir / "final_checkpoint.pt"
    generated_paths = (metrics_path, summary_path, config_path, checkpoint_path)
    existing_paths = [path for path in generated_paths if path.exists()]
    if existing_paths and not config.overwrite:
        raise FileExistsError(
            "Run outputs already exist; choose another output directory or pass --overwrite: "
            + ", ".join(str(path) for path in existing_paths)
        )
    if config.overwrite:
        for path in existing_paths:
            path.unlink()

    _atomic_json_dump(_jsonable_config(config), config_path)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    token_dtype = _resolve_token_dtype(config.train_data, config.token_dtype)
    train_data = load_token_data(config.train_data, token_dtype)
    valid_data = load_token_data(config.valid_data, token_dtype)
    if len(train_data) <= config.context_length or len(valid_data) <= config.context_length:
        raise ValueError("Both train and validation data must contain more than context_length tokens")

    raw_model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    optimizer = AdamW(
        raw_model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    model: torch.nn.Module = raw_model
    if config.compile_model:
        compile_kwargs: dict[str, str] = {}
        if device.type == "mps":
            compile_kwargs["backend"] = "aot_eager"
        model = torch.compile(raw_model, **compile_kwargs)

    max_steps = config.resolved_max_steps()
    tokens_per_step = config.batch_size * config.context_length
    train_rng = np.random.default_rng(config.seed)
    initial_validation_loss: float | None = None
    best_validation_loss = math.inf
    final_validation_loss: float | None = None
    final_train_loss: float | None = None
    divergence_reason: str | None = None
    completed_steps = 0
    started_at = time.perf_counter()

    with JsonlLogger(metrics_path) as logger:
        initial_validation_loss = evaluate(
            model,
            valid_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            eval_batches=config.eval_batches,
            device=device,
            amp_dtype=config.amp_dtype,
            seed=config.seed + 1,
        )
        best_validation_loss = initial_validation_loss
        effective_divergence_threshold = min(
            config.divergence_threshold,
            config.divergence_factor * initial_validation_loss,
        )
        logger.log(
            {
                "event": "validation",
                "step": 0,
                "tokens": 0,
                "loss": initial_validation_loss,
                "elapsed_seconds": time.perf_counter() - started_at,
            }
        )

        raw_model.train()
        for step in range(1, max_steps + 1):
            learning_rate = get_lr_cosine_schedule(
                step,
                config.learning_rate,
                config.learning_rate * config.min_lr_ratio,
                config.warmup_steps,
                max_steps,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            inputs, targets = get_batch(
                train_data,
                config.batch_size,
                config.context_length,
                device,
                rng=train_rng,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, config.amp_dtype):
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
            loss_value = loss.detach().float().item()
            final_train_loss = loss_value

            if not math.isfinite(loss_value):
                divergence_reason = f"non-finite training loss at step {step}"
                break
            if loss_value > effective_divergence_threshold:
                divergence_reason = (
                    f"training loss {loss_value:.6g} exceeded threshold "
                    f"{effective_divergence_threshold:.6g} at step {step}"
                )
                break

            loss.backward()
            grad_norm = clip_gradient_norm(raw_model.parameters(), config.grad_clip)
            if not math.isfinite(grad_norm):
                divergence_reason = f"non-finite gradient norm at step {step}"
                break
            optimizer.step()
            completed_steps = step

            if step == 1 or step % config.log_interval == 0 or step == max_steps:
                _synchronize(device)
                elapsed = time.perf_counter() - started_at
                logger.log(
                    {
                        "event": "train",
                        "step": step,
                        "tokens": step * tokens_per_step,
                        "loss": loss_value,
                        "learning_rate": learning_rate,
                        "gradient_norm": grad_norm,
                        "elapsed_seconds": elapsed,
                        "tokens_per_second": step * tokens_per_step / max(elapsed, 1e-9),
                    }
                )

            if step % config.eval_interval == 0 or step == max_steps:
                final_validation_loss = evaluate(
                    model,
                    valid_data,
                    batch_size=config.batch_size,
                    context_length=config.context_length,
                    eval_batches=config.eval_batches,
                    device=device,
                    amp_dtype=config.amp_dtype,
                    seed=config.seed + 1,
                )
                best_validation_loss = min(best_validation_loss, final_validation_loss)
                logger.log(
                    {
                        "event": "validation",
                        "step": step,
                        "tokens": step * tokens_per_step,
                        "loss": final_validation_loss,
                        "elapsed_seconds": time.perf_counter() - started_at,
                    }
                )
                if not math.isfinite(final_validation_loss):
                    divergence_reason = f"non-finite validation loss at step {step}"
                    break

        if divergence_reason is not None:
            logger.log(
                {
                    "event": "divergence",
                    "step": completed_steps,
                    "tokens": completed_steps * tokens_per_step,
                    "reason": divergence_reason,
                    "elapsed_seconds": time.perf_counter() - started_at,
                }
            )

    _synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    if final_validation_loss is None and divergence_reason is None:
        final_validation_loss = evaluate(
            model,
            valid_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            eval_batches=config.eval_batches,
            device=device,
            amp_dtype=config.amp_dtype,
            seed=config.seed + 1,
        )
        best_validation_loss = min(best_validation_loss, final_validation_loss)

    summary: dict[str, Any] = {
        "status": "diverged" if divergence_reason else "completed",
        "divergence_reason": divergence_reason,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "parameter_count": parameter_count,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "tokens_per_step": tokens_per_step,
        "tokens_processed": completed_steps * tokens_per_step,
        "requested_total_tokens": config.total_tokens,
        "initial_validation_loss": initial_validation_loss,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_validation_loss,
        "best_validation_loss": best_validation_loss,
        "elapsed_seconds": elapsed_seconds,
        "mean_tokens_per_second": completed_steps * tokens_per_step / max(elapsed_seconds, 1e-9),
        "learning_rate": config.learning_rate,
        "effective_divergence_threshold": effective_divergence_threshold,
        "seed": config.seed,
    }
    _atomic_json_dump(summary, summary_path)
    if config.checkpoint != "none" and divergence_reason is None:
        _save_final_checkpoint(checkpoint_path, raw_model, optimizer, completed_steps, config, summary)
    return summary


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--token-dtype", default="auto")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1_344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--total-tokens", type=int, default=40_960_000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="none")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--checkpoint", choices=("none", "model", "full"), default="none")
    parser.add_argument("--divergence-threshold", type=float, default=100.0)
    parser.add_argument("--divergence-factor", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser.parse_args()


def main() -> None:
    config = TrainConfig(**vars(parse_args()))
    summary = run_training(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
