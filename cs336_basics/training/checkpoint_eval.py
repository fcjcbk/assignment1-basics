from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cs336_basics.evaluation import ValidationLossResult, evaluate_validation_loss, with_checkpoint_iteration
from cs336_basics.model.funtional import load_checkpoint
from cs336_basics.training.config import TrainingConfig
from cs336_basics.training.data import load_validation_dataset
from cs336_basics.training.factory import build_model, build_optimizer
from cs336_basics.training.runtime import configure_logging, resolve_device, set_seed


def evaluate_model_on_validation(
    model: torch.nn.Module,
    dataset: np.ndarray,
    config: TrainingConfig,
    device: str,
    mode: str | None = None,
    num_batches: int | None = None,
    batch_size: int | None = None,
) -> ValidationLossResult:
    eval_mode = mode or config.eval.mode
    eval_num_batches = num_batches if num_batches is not None else config.eval.num_batches
    eval_batch_size = batch_size or config.eval.batch_size or config.batch_size
    return evaluate_validation_loss(
        model=model,
        dataset=dataset,
        mode=eval_mode,
        batch_size=eval_batch_size,
        context_length=config.model.context_length,
        device=device,
        num_batches=eval_num_batches,
    )


def evaluate_checkpoint(
    config: TrainingConfig,
    checkpoint_path: str | Path,
    valid_path: str | None = None,
    mode: str | None = None,
    num_batches: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> ValidationLossResult:
    logger = configure_logging(config.logging)
    resolved_device = resolve_device(device or config.device)
    set_seed(config.seed)

    validation_path = valid_path or config.eval.valid_path
    if validation_path is None:
        raise ValueError("A validation path is required. Set eval.valid_path or pass --valid-path.")

    validation_dataset = load_validation_dataset(config, validation_path, logger)
    model = build_model(config.model, resolved_device)
    optimizer = build_optimizer(model, config.optimizer)
    checkpoint_iteration = load_checkpoint(checkpoint_path, model, optimizer)
    result = evaluate_model_on_validation(
        model=model,
        dataset=validation_dataset,
        config=config,
        device=resolved_device,
        mode=mode,
        num_batches=num_batches,
        batch_size=batch_size,
    )
    return with_checkpoint_iteration(result, checkpoint_iteration)


def format_validation_result(result: ValidationLossResult, step: int | None = None) -> str:
    pieces: list[str] = []
    if step is not None:
        pieces.append(f"step={step}")
    if result.checkpoint_iteration is not None:
        pieces.append(f"checkpoint_iteration={result.checkpoint_iteration}")
    pieces.extend(
        [
            f"mode={result.mode}",
            f"val_loss={result.loss:.6f}",
            f"val_ppl={result.perplexity:.6f}",
            f"eval_tokens={result.token_count}",
            f"eval_elapsed={result.elapsed_seconds:.2f}s",
        ]
    )
    return " ".join(pieces)
