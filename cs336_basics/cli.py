from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from einops import rearrange

from cs336_basics.data_pipeline import DEFAULT_SPECIAL_TOKENS, encode_validation_dataset, preprocess_tinystories
from cs336_basics.decode import generate as generate_tokens
from cs336_basics.training.checkpoint_eval import evaluate_checkpoint, format_validation_result
from cs336_basics.training.config import load_config, print_example_config
from cs336_basics.training.factory import build_model, build_optimizer
from cs336_basics.training.runtime import resolve_device, set_seed
from cs336_basics.training.trainer import train
from cs336_basics.model.funtional import load_checkpoint
from cs336_basics.data_pipeline import load_tokenizer_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified training and data pipeline entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a Transformer language model.")
    add_train_arguments(train_parser)
    train_parser.set_defaults(func=run_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate validation loss for a saved checkpoint.")
    add_eval_arguments(eval_parser)
    eval_parser.set_defaults(func=run_eval)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a saved checkpoint.")
    add_generate_arguments(generate_parser)
    generate_parser.set_defaults(func=run_generate)

    preprocess_parser = subparsers.add_parser("preprocess-tinystories", help="Train a BPE tokenizer and encode data.")
    add_preprocess_tinystories_arguments(preprocess_parser)
    preprocess_parser.set_defaults(func=run_preprocess_tinystories)

    encode_parser = subparsers.add_parser("encode-validation", help="Encode validation text with an existing tokenizer.")
    add_encode_validation_arguments(encode_parser)
    encode_parser.set_defaults(func=run_encode_validation)
    return parser


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="train_config.json", help="Path to a JSON training config.")
    parser.add_argument("--print-example-config", action="store_true", help="Print a complete example config and exit.")


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="train_config.json", help="Path to a JSON training config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--valid-path", type=str, default=None, help="Validation token dataset path override.")
    parser.add_argument("--mode", choices=("sampled", "full"), default=None, help="Validation mode override.")
    parser.add_argument("--num-batches", type=int, default=None, help="Number of sampled validation batches.")
    parser.add_argument("--batch-size", type=int, default=None, help="Validation batch size override.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="train_config.json", help="Path to a JSON training config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path to load.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="Directory containing vocab.base64.json, merges.base64.json, and metadata.json.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to encode and continue.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument("--eos-token-id", type=int, default=0, help="Token id that stops generation.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling p value.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")


def add_preprocess_tinystories_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("tests/fixtures/tinystories_sample.txt"), help="Input text file.")
    parser.add_argument("--output", type=Path, default=Path("data/tinystories_sample_tokens.npy"), help="Output .npy file.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("data/tinystories_tokenizer"),
        help="Directory for saved tokenizer artifacts.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Maximum BPE vocabulary size.")
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=None,
        help="Special token to reserve. Can be passed multiple times.",
    )
    parser.add_argument("--dtype", default="int64", help="NumPy dtype for token ids, for example int64 or uint16.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and stage output.")
    parser.add_argument(
        "--train-config-out",
        type=Path,
        default=None,
        help="Optional path to write a training config that points at the generated dataset.",
    )


def add_encode_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"), help="Input text file.")
    parser.add_argument("--output", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.npy"), help="Output .npy file.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("data/tinystories_train_tokenizer"),
        help="Directory containing tokenizer artifacts.",
    )
    parser.add_argument("--dtype", default="int64", help="NumPy dtype for token ids, for example int64 or uint16.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and stage output.")


def run_train(args: argparse.Namespace) -> None:
    if args.print_example_config:
        print_example_config()
        return
    train(load_config(args.config))


def run_eval(args: argparse.Namespace) -> None:
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


def run_generate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = resolve_device(args.device or config.device)
    set_seed(config.seed)

    model = build_model(config.model, device)
    optimizer = build_optimizer(model, config.optimizer)
    load_checkpoint(args.checkpoint, model, optimizer)

    tokenizer = load_tokenizer_artifacts(args.tokenizer_dir)
    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        raise ValueError("Prompt must encode to at least one token")

    prompt_ids = torch.tensor(prompt_tokens, dtype=torch.int64, device=device)
    prompt_ids = rearrange(prompt_ids, "sequence -> 1 sequence")

    model.eval()
    with torch.no_grad():
        output_ids = generate_tokens(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=args.eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    output_ids = rearrange(output_ids, "1 sequence -> sequence")
    print(tokenizer.decode(output_ids.tolist()))


def run_preprocess_tinystories(args: argparse.Namespace) -> None:
    preprocess_tinystories(
        input_path=args.input,
        output_path=args.output,
        tokenizer_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens or DEFAULT_SPECIAL_TOKENS,
        dtype=args.dtype,
        train_config_out=args.train_config_out,
        show_progress=not args.no_progress,
    )


def run_encode_validation(args: argparse.Namespace) -> None:
    encode_validation_dataset(
        input_path=args.input,
        output_path=args.output,
        tokenizer_dir=args.tokenizer_dir,
        dtype=args.dtype,
        show_progress=not args.no_progress,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
