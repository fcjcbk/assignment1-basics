from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer.bpe import train_bpe
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.training.config import TrainingConfig


DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
DEFAULT_TOKEN_CHUNK_SIZE = 1_000_000


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


def load_tokenizer_artifacts(tokenizer_dir: Path) -> Tokenizer:
    _check_tokenizer_artifacts_exist(tokenizer_dir)
    vocab_path, merges_path, metadata_path = _tokenizer_artifact_paths(tokenizer_dir)
    return Tokenizer.from_files(vocab_path, merges_path, metadata_path=metadata_path)


def preprocess_tinystories(
    input_path: Path,
    output_path: Path,
    tokenizer_dir: Path,
    vocab_size: int,
    special_tokens: list[str],
    dtype: str,
    train_config_out: Path | None,
    show_progress: bool = True,
) -> None:
    if output_path.suffix != ".npy":
        raise ValueError("output path must end with .npy so train_model.py can load it with np.load")

    _print_stage("Training BPE tokenizer", show_progress)
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=show_progress,
    )
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    token_dtype = _resolve_token_dtype(dtype, len(vocab))

    _print_stage("Counting token ids", show_progress)
    token_count, max_token_id = _count_encoded_tokens(input_path, tokenizer, token_dtype, show_progress)
    _validate_encoded_tokens(token_count, max_token_id, len(vocab))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _print_stage("Writing token ids", show_progress)
    _write_encoded_tokens(input_path, output_path, tokenizer, token_dtype, token_count, show_progress)

    _print_stage("Saving tokenizer artifacts", show_progress)
    save_tokenizer_artifacts(tokenizer_dir, vocab, merges, special_tokens)

    if train_config_out is not None:
        _print_stage("Saving training config", show_progress)
        save_training_config(train_config_out, output_path, len(vocab))

    print(f"Wrote {token_count} token ids to {output_path}")
    print(f"Wrote tokenizer artifacts to {tokenizer_dir}")
    if train_config_out is not None:
        print(f"Wrote train config to {train_config_out}")


def encode_validation_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer_dir: Path,
    dtype: str,
    show_progress: bool = True,
) -> None:
    if output_path.suffix != ".npy":
        raise ValueError("output path must end with .npy so train_model.py can load it with np.load")

    tokenizer = load_tokenizer_artifacts(tokenizer_dir)
    token_dtype = _resolve_token_dtype(dtype, len(tokenizer.vocab))

    _print_stage("Counting token ids", show_progress)
    token_count, max_token_id = _count_encoded_tokens(input_path, tokenizer, token_dtype, show_progress)
    _validate_encoded_tokens(token_count, max_token_id, len(tokenizer.vocab))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _print_stage("Writing token ids", show_progress)
    _write_encoded_tokens(input_path, output_path, tokenizer, token_dtype, token_count, show_progress)

    print(f"Wrote {token_count} token ids to {output_path}")


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


def _resolve_token_dtype(dtype: str, vocab_size: int) -> np.dtype:
    token_dtype = np.dtype(dtype)
    if not np.issubdtype(token_dtype, np.integer):
        raise ValueError(f"token dtype must be an integer dtype, got {token_dtype}")

    max_storable_token_id = np.iinfo(token_dtype).max
    max_vocab_token_id = vocab_size - 1
    if max_storable_token_id < max_vocab_token_id:
        raise ValueError(
            f"token dtype {token_dtype} can store token ids up to {max_storable_token_id}, "
            f"but vocab size {vocab_size} requires ids up to {max_vocab_token_id}"
        )
    return token_dtype


def _iter_encoded_token_chunks(
    tokenizer: Tokenizer,
    text_iterable: Iterable[str],
    dtype: np.dtype,
    chunk_size: int = DEFAULT_TOKEN_CHUNK_SIZE,
) -> Iterator[np.ndarray]:
    token_buffer: list[int] = []
    for token_id in tokenizer.encode_iterable(text_iterable):
        token_buffer.append(token_id)
        if len(token_buffer) >= chunk_size:
            yield np.asarray(token_buffer, dtype=dtype)
            token_buffer.clear()

    if token_buffer:
        yield np.asarray(token_buffer, dtype=dtype)


def _count_encoded_tokens(
    input_path: Path,
    tokenizer: Tokenizer,
    dtype: np.dtype,
    show_progress: bool = False,
) -> tuple[int, int]:
    token_count = 0
    max_token_id = -1
    progress = tqdm(desc="Counting token ids", unit="token", disable=not show_progress)
    try:
        with input_path.open(encoding="utf-8") as input_file:
            for chunk in _iter_encoded_token_chunks(tokenizer, input_file, dtype):
                token_count += len(chunk)
                max_token_id = max(max_token_id, int(chunk.max()))
                progress.update(len(chunk))
    finally:
        progress.close()
    return token_count, max_token_id


def _write_encoded_tokens(
    input_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
    dtype: np.dtype,
    token_count: int,
    show_progress: bool = False,
) -> None:
    token_ids = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=(token_count,))
    offset = 0
    progress = tqdm(total=token_count, desc="Writing token ids", unit="token", disable=not show_progress)
    try:
        with input_path.open(encoding="utf-8") as input_file:
            for chunk in _iter_encoded_token_chunks(tokenizer, input_file, dtype):
                next_offset = offset + len(chunk)
                token_ids[offset:next_offset] = chunk
                offset = next_offset
                progress.update(len(chunk))
        token_ids.flush()
    finally:
        progress.close()


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _validate_encoded_tokens(token_count: int, max_token_id: int, vocab_size: int) -> None:
    if token_count < 2:
        raise ValueError(f"Expected at least two token ids, got {token_count}")
    if max_token_id >= vocab_size:
        raise ValueError(f"Encoded token id {max_token_id} is outside vocab size {vocab_size}")


def _print_stage(message: str, show_progress: bool) -> None:
    if show_progress:
        print(f"[{message}]")
