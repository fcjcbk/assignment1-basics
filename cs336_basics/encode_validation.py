from __future__ import annotations

import argparse
from pathlib import Path

from cs336_basics.data_pipeline import (
    _check_tokenizer_artifacts_exist,
    _tokenizer_artifact_paths,
    encode_validation_dataset,
    load_tokenizer_artifacts,
)


DEFAULT_INPUT_PATH = Path("data/TinyStoriesV2-GPT4-valid.txt")
DEFAULT_OUTPUT_PATH = Path("data/TinyStoriesV2-GPT4-valid.npy")
DEFAULT_TOKENIZER_DIR = Path("data/tinystories_train_tokenizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a validation text dataset with an existing tokenizer and write a 1D .npy token dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input validation text file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output .npy token-id dataset.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help="Directory containing vocab.base64.json, merges.base64.json, and metadata.json.",
    )
    parser.add_argument("--dtype", default="int64", help="NumPy dtype for token ids, for example int64 or uint16.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and stage output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encode_validation_dataset(
        input_path=args.input,
        output_path=args.output,
        tokenizer_dir=args.tokenizer_dir,
        dtype=args.dtype,
        show_progress=not args.no_progress,
    )


__all__ = [
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TOKENIZER_DIR",
    "_check_tokenizer_artifacts_exist",
    "_tokenizer_artifact_paths",
    "encode_validation_dataset",
    "load_tokenizer_artifacts",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
