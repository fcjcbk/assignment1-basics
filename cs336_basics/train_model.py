from __future__ import annotations

import argparse

from cs336_basics.model.funtional import load_checkpoint, sample_train_data
from cs336_basics.training.checkpoint_eval import (
    evaluate_checkpoint,
    evaluate_model_on_validation,
    format_validation_result,
)
from cs336_basics.training.config import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    LoggingConfig,
    LossPlotConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    load_config,
    print_example_config,
)
from cs336_basics.training.data import load_dataset, load_validation_dataset
from cs336_basics.training.factory import build_model, build_optimizer
from cs336_basics.training.plotting import LossCurvePlotter, render_loss_curve_png
from cs336_basics.training.runtime import configure_logging, resolve_device, set_seed
from cs336_basics.training.trainer import (
    Trainer,
    _checkpoint_path_for_step,
    _save_step_checkpoint,
    _signal_name,
    train as _train,
)


def train(config: TrainingConfig) -> None:
    _train(config, sample_batch_fn=sample_train_data)


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


__all__ = [
    "CheckpointConfig",
    "DataConfig",
    "EvalConfig",
    "LoggingConfig",
    "LossCurvePlotter",
    "LossPlotConfig",
    "ModelConfig",
    "OptimizerConfig",
    "Trainer",
    "TrainingConfig",
    "_checkpoint_path_for_step",
    "_save_step_checkpoint",
    "_signal_name",
    "build_model",
    "build_optimizer",
    "configure_logging",
    "evaluate_checkpoint",
    "evaluate_model_on_validation",
    "format_validation_result",
    "load_checkpoint",
    "load_config",
    "load_dataset",
    "load_validation_dataset",
    "main",
    "parse_args",
    "print_example_config",
    "render_loss_curve_png",
    "resolve_device",
    "sample_train_data",
    "set_seed",
    "train",
]


if __name__ == "__main__":
    main()
