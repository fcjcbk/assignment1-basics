from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace

import einops
import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.model.funtional import cross_entropy, sample_train_data


@dataclass(frozen=True)
class ValidationLossResult:
    loss: float
    perplexity: float
    token_count: int
    mode: str
    elapsed_seconds: float
    checkpoint_iteration: int | None = None


def evaluate_loss_sampled(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    num_batches: int,
    device: str,
) -> ValidationLossResult:
    _validate_common_args(dataset, batch_size, context_length)
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(num_batches):
            yield sample_train_data(dataset, batch_size, context_length, device)

    return _evaluate_batches(model, batches(), mode="sampled")


def evaluate_loss_full(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> ValidationLossResult:
    _validate_common_args(dataset, batch_size, context_length)
    num_examples = (len(dataset) - 1) // context_length
    if num_examples <= 0:
        raise ValueError("Validation dataset does not contain a complete context window")

    def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for example_start in range(0, num_examples, batch_size):
            example_end = min(example_start + batch_size, num_examples)
            starts = range(example_start * context_length, example_end * context_length, context_length)
            inputs = np.stack([dataset[start : start + context_length] for start in starts])
            targets = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
            yield (
                torch.tensor(inputs, dtype=torch.long, device=device),
                torch.tensor(targets, dtype=torch.long, device=device),
            )

    return _evaluate_batches(model, batches(), mode="full")


def evaluate_validation_loss(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    mode: str,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 50,
) -> ValidationLossResult:
    if mode == "sampled":
        return evaluate_loss_sampled(model, dataset, batch_size, context_length, num_batches, device)
    if mode == "full":
        return evaluate_loss_full(model, dataset, batch_size, context_length, device)
    raise ValueError(f"Unknown validation mode: {mode!r}")


def with_checkpoint_iteration(result: ValidationLossResult, checkpoint_iteration: int) -> ValidationLossResult:
    return replace(result, checkpoint_iteration=checkpoint_iteration)


def _validate_common_args(dataset: npt.NDArray, batch_size: int, context_length: int) -> None:
    if dataset.ndim != 1:
        raise ValueError(f"Expected a 1D validation token dataset, got shape {dataset.shape}")
    if len(dataset) <= context_length:
        raise ValueError("Validation dataset length must be greater than context_length")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")


def _evaluate_batches(
    model: torch.nn.Module,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    mode: str,
) -> ValidationLossResult:
    was_training = model.training
    vocab_size = _infer_vocab_size(model)
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    model.eval()
    try:
        with torch.no_grad():
            for inputs, targets in batches:
                _validate_token_ids(inputs, targets, vocab_size)
                logits = model(inputs)
                flat_logits = einops.rearrange(logits, "batch sequence vocab -> (batch sequence) vocab")
                flat_targets = einops.rearrange(targets, "batch sequence -> (batch sequence)")
                loss = cross_entropy(flat_logits, flat_targets)
                token_count = flat_targets.numel()
                total_loss += loss.item() * token_count
                total_tokens += token_count
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    if total_tokens == 0:
        raise ValueError("No validation tokens were evaluated")

    average_loss = total_loss / total_tokens
    return ValidationLossResult(
        loss=average_loss,
        perplexity=math.exp(average_loss),
        token_count=total_tokens,
        mode=mode,
        elapsed_seconds=time.time() - start_time,
    )


def _infer_vocab_size(model: torch.nn.Module) -> int | None:
    embedding = getattr(model, "embedding", None)
    weight = getattr(embedding, "weight", None)
    if weight is None:
        return None
    return int(weight.shape[0])


def _validate_token_ids(inputs: torch.Tensor, targets: torch.Tensor, vocab_size: int | None) -> None:
    if inputs.numel() == 0 or targets.numel() == 0:
        raise ValueError("Validation batches must not be empty")
    min_token_id = min(int(inputs.min().item()), int(targets.min().item()))
    if min_token_id < 0:
        raise ValueError(f"Validation token ids must be non-negative, got {min_token_id}")
    if vocab_size is None:
        return
    max_token_id = max(int(inputs.max().item()), int(targets.max().item()))
    if max_token_id >= vocab_size:
        raise ValueError(
            f"Validation token id {max_token_id} is outside model vocab size {vocab_size}"
        )
