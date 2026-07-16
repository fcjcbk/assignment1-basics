from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from secrets import token_hex
from typing import Any

import numpy as np
import torch
from einops import rearrange

from cs336_basics.model.funtional import (
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    sample_train_data,
    save_checkpoint,
)
from cs336_basics.training.checkpoint_eval import evaluate_model_on_validation, format_validation_result
from cs336_basics.training.config import TrainingConfig, validate_run_name
from cs336_basics.training.data import load_dataset, load_validation_dataset
from cs336_basics.training.factory import build_model, build_optimizer
from cs336_basics.training.plotting import LossCurvePlotter
from cs336_basics.training.runtime import configure_logging, resolve_device, set_seed


BatchSampler = Callable[..., tuple[torch.Tensor, torch.Tensor]]


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        sample_batch_fn: BatchSampler = sample_train_data,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.sample_batch_fn = sample_batch_fn
        self.logger = logger

    def train(self) -> None:
        config = _config_for_training_run(self.config)
        self.config = config
        logger = self.logger or configure_logging(config.logging)
        device = resolve_device(config.device)
        set_seed(config.seed)

        self._validate_model_sequence_lengths()

        dataset = load_dataset(config.data, config.model.vocab_size, logger)
        if len(dataset) <= config.model.context_length:
            raise ValueError("Dataset length must be greater than model.context_length")

        validation_dataset: np.ndarray | None = None
        if config.eval.valid_path is not None and config.eval.interval > 0:
            validation_dataset = load_validation_dataset(config, config.eval.valid_path, logger)
            if len(validation_dataset) <= config.model.context_length:
                raise ValueError("Validation dataset length must be greater than model.context_length")

        model = build_model(config.model, device)
        optimizer = build_optimizer(model, config.optimizer)
        start_step = self._maybe_resume(model, optimizer, logger)

        checkpoint_path = Path(config.checkpoint.path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        loss_plotter = LossCurvePlotter(config.plot, config.total_steps) if config.plot.enabled else None
        if loss_plotter is not None:
            logger.info(
                "Writing matplotlib training monitor to %s every %d step(s)%s",
                loss_plotter.path,
                config.plot.interval,
                " and refreshing an interactive window" if config.plot.show else "",
            )

        model.train()
        start_time = time.time()
        last_log_time = start_time
        completed_step = start_step
        last_checkpoint_step: int | None = None
        shutdown_requested = False
        shutdown_signal: str | None = None

        def request_shutdown(signum: int, _frame: Any) -> None:
            nonlocal shutdown_requested, shutdown_signal
            shutdown_requested = True
            shutdown_signal = _signal_name(signum)

        original_signal_handlers = _install_shutdown_handlers(request_shutdown)

        logger.info(
            "Starting training: run=%s, steps=%d, batch_size=%d, context_length=%d, device=%s",
            config.run.name,
            config.total_steps,
            config.batch_size,
            config.model.context_length,
            device,
        )

        try:
            for step in range(start_step + 1, config.total_steps + 1):
                lr = self._set_learning_rate(optimizer, step)
                train_loss = self._train_step(model, optimizer, dataset, device)
                completed_step = step
                elapsed = time.time() - start_time
                completed_training_steps = step - start_step
                average_steps_per_sec = completed_training_steps / max(elapsed, 1e-8)

                if loss_plotter is not None:
                    loss_plotter.record_train_loss(
                        step,
                        train_loss,
                        learning_rate=lr,
                        steps_per_second=average_steps_per_sec,
                        elapsed_seconds=elapsed,
                    )
                    loss_plotter.maybe_render(step)

                if step % config.logging.log_interval == 0 or step == 1:
                    now = time.time()
                    steps_per_sec = config.logging.log_interval / max(now - last_log_time, 1e-8)
                    last_log_time = now
                    logger.info(
                        "step=%d/%d loss=%.6f lr=%.6g elapsed=%.1fs steps_per_sec=%.2f",
                        step,
                        config.total_steps,
                        train_loss,
                        lr,
                        elapsed,
                        steps_per_sec,
                    )

                if validation_dataset is not None and (step % config.eval.interval == 0 or step == config.total_steps):
                    eval_result = evaluate_model_on_validation(
                        model=model,
                        dataset=validation_dataset,
                        config=config,
                        device=device,
                    )
                    logger.info(format_validation_result(eval_result, step=step))
                    if loss_plotter is not None:
                        loss_plotter.record_validation_loss(step, eval_result.loss)
                        loss_plotter.maybe_render(step, force=True)

                saved_checkpoint_path: Path | None = None
                if config.checkpoint.save_interval > 0 and step % config.checkpoint.save_interval == 0:
                    saved_checkpoint_path = _save_step_checkpoint(model, optimizer, step, checkpoint_path, logger)
                    last_checkpoint_step = step

                if shutdown_requested:
                    if last_checkpoint_step != step:
                        saved_checkpoint_path = _save_step_checkpoint(model, optimizer, step, checkpoint_path, logger)
                        last_checkpoint_step = step
                    if saved_checkpoint_path is None:
                        saved_checkpoint_path = _checkpoint_path_for_step(checkpoint_path, step)
                    logger.info(
                        "Received %s. Exiting after step %d with checkpoint %s",
                        shutdown_signal or "shutdown request",
                        step,
                        saved_checkpoint_path,
                    )
                    if loss_plotter is not None:
                        loss_plotter.maybe_render(step, force=True)
                    return

            final_checkpoint_path: Path
            if last_checkpoint_step == config.total_steps:
                final_checkpoint_path = _checkpoint_path_for_step(checkpoint_path, config.total_steps)
            else:
                final_checkpoint_path = _save_step_checkpoint(model, optimizer, config.total_steps, checkpoint_path, logger)
            logger.info("Training complete. Final checkpoint saved to %s", final_checkpoint_path)
        except KeyboardInterrupt:
            if completed_step <= start_step:
                raise
            interrupted_checkpoint_path = _checkpoint_path_for_step(checkpoint_path, completed_step)
            if last_checkpoint_step != completed_step:
                interrupted_checkpoint_path = _save_step_checkpoint(
                    model,
                    optimizer,
                    completed_step,
                    checkpoint_path,
                    logger,
                )
            logger.info("Interrupted. Latest completed checkpoint saved to %s", interrupted_checkpoint_path)
        finally:
            if loss_plotter is not None:
                loss_plotter.maybe_render(completed_step, force=True)
            _restore_signal_handlers(original_signal_handlers)

    def _validate_model_sequence_lengths(self) -> None:
        if self.config.model.max_seq_len < self.config.model.context_length:
            raise ValueError("model.max_seq_len must be greater than or equal to model.context_length")

    def _maybe_resume(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, logger: logging.Logger) -> int:
        if self.config.checkpoint.resume_from is None:
            return 0
        start_step = load_checkpoint(self.config.checkpoint.resume_from, model, optimizer)
        logger.info("Resumed checkpoint from %s at step %d", self.config.checkpoint.resume_from, start_step)
        return start_step

    def _set_learning_rate(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=self.config.optimizer.learning_rate,
            min_learning_rate=self.config.optimizer.min_learning_rate,
            warmup_iters=self.config.optimizer.warmup_iters,
            cosine_cycle_iters=self.config.total_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _train_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: np.ndarray,
        device: str,
    ) -> float:
        inputs, targets = self.sample_batch_fn(
            dataset=dataset,
            batch_size=self.config.batch_size,
            context_length=self.config.model.context_length,
            device=device,
        )

        logits = model(inputs)
        flat_logits = rearrange(logits, "batch sequence vocab -> (batch sequence) vocab")
        flat_targets = rearrange(targets, "batch sequence -> (batch sequence)")
        loss = cross_entropy(flat_logits, flat_targets)

        optimizer.zero_grad()
        loss.backward()
        if self.config.optimizer.max_grad_norm is not None:
            gradient_clipping(model.parameters(), self.config.optimizer.max_grad_norm)
        optimizer.step()
        return float(loss.item())


def train(config: TrainingConfig, sample_batch_fn: BatchSampler = sample_train_data) -> None:
    Trainer(config, sample_batch_fn=sample_batch_fn).train()


def _config_for_training_run(config: TrainingConfig) -> TrainingConfig:
    run_name = _resolve_run_name(config)
    checkpoint_path = _artifact_path_for_run(Path(config.checkpoint.path), run_name)
    logging_config = config.logging
    if config.logging.log_file is not None:
        log_file = _artifact_path_for_run(Path(config.logging.log_file), run_name)
        logging_config = replace(config.logging, log_file=str(log_file))

    return replace(
        config,
        checkpoint=replace(config.checkpoint, path=str(checkpoint_path)),
        logging=logging_config,
        plot=replace(config.plot, path=str(_artifact_path_for_run(Path(config.plot.path), run_name))),
        run=replace(config.run, name=run_name),
    )


def _resolve_run_name(config: TrainingConfig) -> str:
    if config.run.name is not None:
        validate_run_name(config.run.name)
        return config.run.name
    return f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{token_hex(3)}"


def _artifact_path_for_run(path: Path, run_name: str) -> Path:
    return path.parent / run_name / path.name


def _checkpoint_path_for_step(checkpoint_path: Path, step: int) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_step_{step}{checkpoint_path.suffix}")


def _save_step_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_path: Path,
    logger: logging.Logger,
) -> Path:
    step_checkpoint_path = _checkpoint_path_for_step(checkpoint_path, step)
    save_checkpoint(model, optimizer, step, step_checkpoint_path)
    logger.info("Saved checkpoint to %s at step %d", step_checkpoint_path, step)
    return step_checkpoint_path


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


def _install_shutdown_handlers(handler: Callable[[int, Any], None]) -> dict[signal.Signals, Any]:
    original_signal_handlers: dict[signal.Signals, Any] = {}
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            original_signal_handlers[handled_signal] = signal.getsignal(handled_signal)
            signal.signal(handled_signal, handler)
        except (OSError, ValueError):
            pass
    return original_signal_handlers


def _restore_signal_handlers(original_signal_handlers: dict[signal.Signals, Any]) -> None:
    for handled_signal, original_handler in original_signal_handlers.items():
        signal.signal(handled_signal, original_handler)
