from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model.funtional import (
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    sample_train_data,
    save_checkpoint,
)
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.model.optimizer import AdamW
from cs336_basics.model.transformer_language_model import TransformerLanguageModel


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 32
    num_layers: int = 2
    d_model: int = 64
    num_heads: int = 4
    max_seq_len: int = 32
    theta: float = 10_000.0
    d_ff: int | None = None


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 10
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float | None = 1.0


@dataclass
class DataConfig:
    train_path: str | None = None
    dtype: str = "int64"
    use_memmap: bool = True
    synthetic_num_tokens: int = 4096


@dataclass
class CheckpointConfig:
    path: str = "checkpoints/latest.pt"
    save_interval: int = 100
    resume_from: str | None = None


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_interval: int = 10
    log_file: str | None = None


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    batch_size: int = 8
    total_steps: int = 100
    device: str = "auto"
    seed: int = 1337


def _merge_dataclass(default: Any, overrides: dict[str, Any]) -> Any:
    if not is_dataclass(default):
        return overrides

    values = {item.name: getattr(default, item.name) for item in fields(default)}
    field_types = {item.name: item.type for item in fields(default)}

    for key, value in overrides.items():
        if key not in values:
            raise ValueError(f"Unknown config field: {key}")
        current_value = getattr(default, key)
        if is_dataclass(current_value):
            if not isinstance(value, dict):
                raise ValueError(f"Config field '{key}' must be an object")
            values[key] = _merge_dataclass(current_value, value)
        else:
            values[key] = value

    return type(default)(**{key: _coerce_optional_none(value, field_types[key]) for key, value in values.items()})


def _coerce_optional_none(value: Any, field_type: Any) -> Any:
    if value != "none":
        return value
    if "NoneType" in str(field_type) or "None" in str(field_type):
        return None
    return value


def load_config(config_path: str | None) -> TrainingConfig:
    config = TrainingConfig()
    if config_path is None:
        return config

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        overrides = json.load(f)
    if not isinstance(overrides, dict):
        raise ValueError("Training config JSON must contain an object at the top level")

    return _merge_dataclass(config, overrides)


def configure_logging(config: LoggingConfig) -> logging.Logger:
    logger = logging.getLogger("train_model")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if config.log_file is not None:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(config: DataConfig, vocab_size: int, logger: logging.Logger) -> np.ndarray:
    if config.train_path is None:
        logger.warning("No data.train_path provided; using a tiny synthetic token dataset for smoke testing.")
        return (np.arange(config.synthetic_num_tokens, dtype=np.int64) % vocab_size).astype(np.int64)

    path = Path(config.train_path)
    if path.suffix == ".npy":
        mmap_mode = "r" if config.use_memmap else None
        dataset = np.load(path, mmap_mode=mmap_mode)
    else:
        dataset = np.fromfile(path, dtype=np.dtype(config.dtype))

    if dataset.ndim != 1:
        raise ValueError(f"Expected a 1D token dataset, got shape {dataset.shape}")
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two tokens")

    logger.info("Loaded dataset from %s with %d tokens", path, len(dataset))
    return dataset


def build_model(config: ModelConfig, device: str) -> TransformerLanguageModel:
    return TransformerLanguageModel(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        theta=config.theta,
        d_ff=config.d_ff,
        device=torch.device(device),
    )


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def train(config: TrainingConfig) -> None:
    logger = configure_logging(config.logging)
    device = resolve_device(config.device)
    set_seed(config.seed)

    if config.model.max_seq_len < config.model.context_length:
        raise ValueError("model.max_seq_len must be greater than or equal to model.context_length")

    dataset = load_dataset(config.data, config.model.vocab_size, logger)
    if len(dataset) <= config.model.context_length:
        raise ValueError("Dataset length must be greater than model.context_length")

    model = build_model(config.model, device)
    optimizer = build_optimizer(model, config.optimizer)
    start_step = 0

    if config.checkpoint.resume_from is not None:
        start_step = load_checkpoint(config.checkpoint.resume_from, model, optimizer)
        logger.info("Resumed checkpoint from %s at step %d", config.checkpoint.resume_from, start_step)

    checkpoint_path = Path(config.checkpoint.path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    start_time = time.time()
    last_log_time = start_time

    logger.info(
        "Starting training: steps=%d, batch_size=%d, context_length=%d, device=%s",
        config.total_steps,
        config.batch_size,
        config.model.context_length,
        device,
    )

    for step in range(start_step + 1, config.total_steps + 1):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=config.optimizer.learning_rate,
            min_learning_rate=config.optimizer.min_learning_rate,
            warmup_iters=config.optimizer.warmup_iters,
            cosine_cycle_iters=config.total_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        inputs, targets = sample_train_data(
            dataset=dataset,
            batch_size=config.batch_size,
            context_length=config.model.context_length,
            device=device,
        )

        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if config.optimizer.max_grad_norm is not None:
            gradient_clipping(model.parameters(), config.optimizer.max_grad_norm)
        optimizer.step()

        if step % config.logging.log_interval == 0 or step == 1:
            now = time.time()
            elapsed = now - start_time
            steps_per_sec = config.logging.log_interval / max(now - last_log_time, 1e-8)
            last_log_time = now
            logger.info(
                "step=%d/%d loss=%.6f lr=%.6g elapsed=%.1fs steps_per_sec=%.2f",
                step,
                config.total_steps,
                loss.item(),
                lr,
                elapsed,
                steps_per_sec,
            )

        if config.checkpoint.save_interval > 0 and step % config.checkpoint.save_interval == 0:
            save_checkpoint(model, optimizer, step, checkpoint_path)
            logger.info("Saved checkpoint to %s at step %d", checkpoint_path, step)

    save_checkpoint(model, optimizer, config.total_steps, checkpoint_path)
    logger.info("Training complete. Final checkpoint saved to %s", checkpoint_path)


def print_example_config() -> None:
    print(json.dumps(asdict(TrainingConfig()), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model from a JSON config.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON training config.")
    parser.add_argument("--print-example-config", action="store_true", help="Print a complete example config and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_example_config:
        print_example_config()
        return

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
