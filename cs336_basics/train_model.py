from __future__ import annotations

import argparse
import json
import logging
import math
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.evaluation import ValidationLossResult, evaluate_validation_loss, with_checkpoint_iteration
from cs336_basics.model.funtional import (
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    sample_train_data,
    save_checkpoint,
)
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
class EvalConfig:
    valid_path: str | None = None
    interval: int = 1000
    mode: str = "sampled"
    num_batches: int = 50
    batch_size: int | None = None


@dataclass
class LossPlotConfig:
    enabled: bool = False
    path: str = "log/loss_curve.png"
    interval: int = 10
    width: int = 960
    height: int = 540
    dpi: int = 120


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    plot: LossPlotConfig = field(default_factory=LossPlotConfig)
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


def load_validation_dataset(config: TrainingConfig, valid_path: str, logger: logging.Logger) -> np.ndarray:
    return load_dataset(
        DataConfig(train_path=valid_path, dtype=config.data.dtype, use_memmap=config.data.use_memmap),
        config.model.vocab_size,
        logger,
    )


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


class LossCurvePlotter:
    def __init__(self, config: LossPlotConfig, total_steps: int):
        if config.interval <= 0:
            raise ValueError("plot.interval must be positive")
        if config.width < 320:
            raise ValueError("plot.width must be at least 320")
        if config.height < 240:
            raise ValueError("plot.height must be at least 240")
        if config.dpi <= 0:
            raise ValueError("plot.dpi must be positive")

        self.config = config
        self.total_steps = total_steps
        self.path = Path(config.path)
        self.train_points: list[tuple[int, float]] = []
        self.validation_points: list[tuple[int, float]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._plt = _load_matplotlib_pyplot()

    def record_train_loss(self, step: int, loss: float) -> None:
        self._append_point(self.train_points, step, loss)

    def record_validation_loss(self, step: int, loss: float) -> None:
        self._append_point(self.validation_points, step, loss)

    def maybe_render(self, step: int, force: bool = False) -> None:
        if force or step == 1 or step % self.config.interval == 0:
            self.render()

    def render(self) -> None:
        render_loss_curve_png(
            plt=self._plt,
            train_points=self.train_points,
            validation_points=self.validation_points,
            total_steps=self.total_steps,
            path=self.path,
            width=self.config.width,
            height=self.config.height,
            dpi=self.config.dpi,
        )

    @staticmethod
    def _append_point(points: list[tuple[int, float]], step: int, loss: float) -> None:
        if math.isfinite(loss):
            points.append((step, loss))


def _load_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required when plot.enabled is true") from exc
    return plt


def render_loss_curve_png(
    plt: Any,
    train_points: list[tuple[int, float]],
    validation_points: list[tuple[int, float]],
    total_steps: int,
    path: Path,
    width: int = 960,
    height: int = 540,
    dpi: int = 120,
) -> None:
    figure_size = (width / dpi, height / dpi)
    figure, axis = plt.subplots(figsize=figure_size, dpi=dpi)
    tmp_path = path.with_suffix(".tmp.png")
    all_points = [*train_points, *validation_points]

    try:
        if train_points:
            train_steps, train_losses = zip(*train_points, strict=True)
            axis.plot(train_steps, train_losses, color="#2563eb", linewidth=2.0, label="train_loss")
            axis.scatter(train_steps[-1], train_losses[-1], color="#1d4ed8", s=26, zorder=3)

        if validation_points:
            validation_steps, validation_losses = zip(*validation_points, strict=True)
            axis.plot(
                validation_steps,
                validation_losses,
                color="#dc2626",
                linewidth=2.0,
                marker="o",
                markersize=4,
                label="val_loss",
            )
            axis.scatter(validation_steps[-1], validation_losses[-1], color="#b91c1c", s=32, zorder=3)

        axis.set_title("Training and validation loss")
        axis.set_xlabel("step")
        axis.set_ylabel("loss")
        max_observed_step = max((step for step, _ in all_points), default=1)
        axis.set_xlim(left=0, right=max(total_steps, max_observed_step, 1))
        axis.grid(True, alpha=0.25)

        if all_points:
            latest = " | ".join(
                piece
                for piece in (
                    _format_latest_loss("train_loss", train_points),
                    _format_latest_loss("val_loss", validation_points),
                )
                if piece
            )
            axis.text(
                0.5,
                1.02,
                latest,
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#4b5563",
            )
            axis.legend(loc="best")
        else:
            axis.text(0.5, 0.5, "Waiting for loss data", transform=axis.transAxes, ha="center", va="center")

        figure.tight_layout()
        figure.savefig(tmp_path, format="png")
        tmp_path.replace(path)
    finally:
        plt.close(figure)


def _format_latest_loss(label: str, points: list[tuple[int, float]]) -> str:
    if not points:
        return ""
    step, loss = points[-1]
    return f"{label}={loss:.6f} @ step {step}"


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


def train(config: TrainingConfig) -> None:
    logger = configure_logging(config.logging)
    device = resolve_device(config.device)
    set_seed(config.seed)

    if config.model.max_seq_len < config.model.context_length:
        raise ValueError("model.max_seq_len must be greater than or equal to model.context_length")

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
    start_step = 0

    if config.checkpoint.resume_from is not None:
        start_step = load_checkpoint(config.checkpoint.resume_from, model, optimizer)
        logger.info("Resumed checkpoint from %s at step %d", config.checkpoint.resume_from, start_step)

    checkpoint_path = Path(config.checkpoint.path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    loss_plotter = LossCurvePlotter(config.plot, config.total_steps) if config.plot.enabled else None
    if loss_plotter is not None:
        logger.info(
            "Writing live matplotlib loss curve to %s every %d step(s)",
            loss_plotter.path,
            config.plot.interval,
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

    original_signal_handlers: dict[signal.Signals, Any] = {}
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            original_signal_handlers[handled_signal] = signal.getsignal(handled_signal)
            signal.signal(handled_signal, request_shutdown)
        except (OSError, ValueError):
            pass

    logger.info(
        "Starting training: steps=%d, batch_size=%d, context_length=%d, device=%s",
        config.total_steps,
        config.batch_size,
        config.model.context_length,
        device,
    )

    try:
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
            completed_step = step
            train_loss = float(loss.item())

            if loss_plotter is not None:
                loss_plotter.record_train_loss(step, train_loss)
                loss_plotter.maybe_render(step)

            if step % config.logging.log_interval == 0 or step == 1:
                now = time.time()
                elapsed = now - start_time
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
        for handled_signal, original_handler in original_signal_handlers.items():
            signal.signal(handled_signal, original_handler)


def print_example_config() -> None:
    print(json.dumps(asdict(TrainingConfig()), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model from a JSON config.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON training config.")
    parser.add_argument("--print-example-config", action="store_true", help="Print a complete example config and exit.")
    parser.add_argument("--eval-checkpoint", type=str, default=None, help="Evaluate this checkpoint and exit without training.")
    parser.add_argument("--valid-path", type=str, default=None, help="Validation token dataset path for checkpoint eval.")
    parser.add_argument("--eval-mode", choices=("sampled", "full"), default=None, help="Validation mode override.")
    parser.add_argument("--eval-num-batches", type=int, default=None, help="Number of sampled validation batches.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Validation batch size override.")
    parser.add_argument("--device", type=str, default=None, help="Device override for checkpoint eval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_example_config:
        print_example_config()
        return

    config = load_config(args.config)
    if args.eval_checkpoint is not None:
        result = evaluate_checkpoint(
            config=config,
            checkpoint_path=args.eval_checkpoint,
            valid_path=args.valid_path,
            mode=args.eval_mode,
            num_batches=args.eval_num_batches,
            batch_size=args.eval_batch_size,
            device=args.device,
        )
        print(format_validation_result(result))
        return

    train(config)


if __name__ == "__main__":
    main()
