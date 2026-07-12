from __future__ import annotations

import argparse

from cs336_basics.training.checkpoint_eval import evaluate_checkpoint, format_validation_result
from cs336_basics.training.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation loss for a saved Transformer checkpoint.")
    parser.add_argument("--config", type=str, default="train_config.json", help="Path to a JSON training config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--valid-path", type=str, default=None, help="Validation token dataset path override.")
    parser.add_argument("--mode", choices=("sampled", "full"), default=None, help="Validation mode override.")
    parser.add_argument("--num-batches", type=int, default=None, help="Number of sampled validation batches.")
    parser.add_argument("--batch-size", type=int, default=None, help="Validation batch size override.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = evaluate_checkpoint(
        config=config,
        checkpoint_path=args.checkpoint,
        valid_path=args.valid_path,
        mode=args.mode,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(format_validation_result(result))


if __name__ == "__main__":
    main()
