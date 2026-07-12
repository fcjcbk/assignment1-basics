from __future__ import annotations

import argparse
from pathlib import Path

from cs336_basics.preprocess_tinystories import _count_encoded_tokens, _resolve_token_dtype, _write_encoded_tokens
from cs336_basics.tokenizer.tokenizer import Tokenizer


DEFAULT_INPUT_PATH = Path("data/TinyStoriesV2-GPT4-valid.txt")
DEFAULT_OUTPUT_PATH = Path("data/TinyStoriesV2-GPT4-valid.npy")
DEFAULT_TOKENIZER_DIR = Path("data/tinystories_train_tokenizer")


def _tokenizer_artifact_paths(tokenizer_dir: Path) -> tuple[Path, Path, Path]:
    vocab_path = tokenizer_dir / "vocab.base64.json"
    merges_path = tokenizer_dir / "merges.base64.json"
    metadata_path = tokenizer_dir / "metadata.json"
    return vocab_path, merges_path, metadata_path


def _check_tokenizer_artifacts_exist(tokenizer_dir: Path) -> None:
    vocab_path, merges_path, metadata_path = _tokenizer_artifact_paths(tokenizer_dir)

    missing_paths = [path for path in (vocab_path, merges_path, metadata_path) if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing tokenizer artifact(s): {missing}")


def load_tokenizer_artifacts(tokenizer_dir: Path) -> Tokenizer:
    _check_tokenizer_artifacts_exist(tokenizer_dir)
    vocab_path, merges_path, metadata_path = _tokenizer_artifact_paths(tokenizer_dir)

    return Tokenizer.from_files(vocab_path, merges_path, metadata_path=metadata_path)


def encode_validation_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer_dir: Path,
    dtype: str,
    show_progress: bool = True,
) -> None:
    if output_path.suffix != ".npy":
        raise ValueError("output path must end with .npy so train_model.py can load it with np.load")

    _check_tokenizer_artifacts_exist(tokenizer_dir)
    vocab_path, merges_path, metadata_path = _tokenizer_artifact_paths(tokenizer_dir)
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        metadata_path=metadata_path,
    )
    token_dtype = _resolve_token_dtype(dtype, len(tokenizer.vocab))

    if show_progress:
        print("[Counting token ids]")
    token_count, max_token_id = _count_encoded_tokens(input_path, tokenizer, token_dtype, show_progress)
    if token_count < 2:
        raise ValueError(f"Expected at least two token ids, got {token_count}")
    if max_token_id >= len(tokenizer.vocab):
        raise ValueError(f"Encoded token id {max_token_id} is outside vocab size {len(tokenizer.vocab)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if show_progress:
        print("[Writing token ids]")
    _write_encoded_tokens(input_path, output_path, tokenizer, token_dtype, token_count, show_progress)

    print(f"Wrote {token_count} token ids to {output_path}")


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


if __name__ == "__main__":
    main()
