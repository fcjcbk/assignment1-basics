from __future__ import annotations

import argparse
from pathlib import Path

from cs336_basics.data_pipeline import (
    DEFAULT_SPECIAL_TOKENS,
    DEFAULT_TOKEN_CHUNK_SIZE,
    _count_encoded_tokens,
    _encode_bytes,
    _iter_encoded_token_chunks,
    _print_stage,
    _resolve_token_dtype,
    _write_encoded_tokens,
    preprocess_tinystories,
    save_tokenizer_artifacts,
    save_training_config,
)


DEFAULT_INPUT_PATH = Path("tests/fixtures/tinystories_sample.txt")
DEFAULT_OUTPUT_PATH = Path("data/tinystories_sample_tokens.npy")
DEFAULT_TOKENIZER_DIR = Path("data/tinystories_tokenizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize TinyStories text into a 1D .npy token-id dataset for train_model.py."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input TinyStories text file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output .npy token-id dataset.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
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
        help="Optional path to write a train_model.py JSON config that points at the generated dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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


__all__ = [
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_SPECIAL_TOKENS",
    "DEFAULT_TOKEN_CHUNK_SIZE",
    "DEFAULT_TOKENIZER_DIR",
    "_count_encoded_tokens",
    "_encode_bytes",
    "_iter_encoded_token_chunks",
    "_print_stage",
    "_resolve_token_dtype",
    "_write_encoded_tokens",
    "main",
    "parse_args",
    "preprocess_tinystories",
    "save_tokenizer_artifacts",
    "save_training_config",
]


if __name__ == "__main__":
    main()
