from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer.bpe import train_bpe
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.train_model import TrainingConfig


DEFAULT_INPUT_PATH = Path("tests/fixtures/tinystories_sample.txt")
DEFAULT_OUTPUT_PATH = Path("data/tinystories_sample_tokens.npy")
DEFAULT_TOKENIZER_DIR = Path("data/tinystories_tokenizer")
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def save_tokenizer_artifacts(
    tokenizer_dir: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    vocab_json = {str(token_id): _encode_bytes(token_bytes) for token_id, token_bytes in vocab.items()}
    merges_json = [[_encode_bytes(left), _encode_bytes(right)] for left, right in merges]

    (tokenizer_dir / "vocab.base64.json").write_text(json.dumps(vocab_json, indent=2), encoding="utf-8")
    (tokenizer_dir / "merges.base64.json").write_text(json.dumps(merges_json, indent=2), encoding="utf-8")
    (tokenizer_dir / "metadata.json").write_text(
        json.dumps(
            {
                "vocab_size": len(vocab),
                "num_merges": len(merges),
                "special_tokens": special_tokens,
                "encoding": "base64",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_training_config(config_path: Path, token_path: Path, vocab_size: int) -> None:
    config = TrainingConfig()
    config.model.vocab_size = vocab_size
    config.data.train_path = str(token_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def preprocess_tinystories(
    input_path: Path,
    output_path: Path,
    tokenizer_dir: Path,
    vocab_size: int,
    special_tokens: list[str],
    dtype: str,
    train_config_out: Path | None,
) -> None:
    if output_path.suffix != ".npy":
        raise ValueError("output path must end with .npy so train_model.py can load it with np.load")

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    text = input_path.read_text(encoding="utf-8")
    token_ids = np.asarray(tokenizer.encode(text), dtype=np.dtype(dtype))
    if token_ids.ndim != 1 or len(token_ids) < 2:
        raise ValueError(f"Expected at least two token ids, got shape {token_ids.shape}")

    max_token_id = int(token_ids.max())
    if max_token_id >= len(vocab):
        raise ValueError(f"Encoded token id {max_token_id} is outside vocab size {len(vocab)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, token_ids)
    save_tokenizer_artifacts(tokenizer_dir, vocab, merges, special_tokens)

    if train_config_out is not None:
        save_training_config(train_config_out, output_path, len(vocab))

    print(f"Wrote {len(token_ids)} token ids to {output_path}")
    print(f"Wrote tokenizer artifacts to {tokenizer_dir}")
    if train_config_out is not None:
        print(f"Wrote train config to {train_config_out}")


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
    )


if __name__ == "__main__":
    main()
