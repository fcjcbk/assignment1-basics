from pathlib import Path

import numpy as np

from cs336_basics.encode_validation import encode_validation_dataset, load_tokenizer_artifacts
from cs336_basics.preprocess_tinystories import save_tokenizer_artifacts


def test_load_tokenizer_artifacts_restores_base64_vocab_merges_and_special_tokens(tmp_path: Path):
    tokenizer_dir = tmp_path / "tokenizer"
    vocab = {0: b"a", 1: b"b", 2: b"ab", 3: b"<|endoftext|>"}
    merges = [(b"a", b"b")]
    special_tokens = ["<|endoftext|>"]
    save_tokenizer_artifacts(tokenizer_dir, vocab, merges, special_tokens)

    tokenizer = load_tokenizer_artifacts(tokenizer_dir)

    assert tokenizer.vocab == vocab
    assert tokenizer.merges == merges
    assert tokenizer.special_tokens == special_tokens


def test_encode_validation_dataset_writes_npy_with_existing_tokenizer(tmp_path: Path):
    tokenizer_dir = tmp_path / "tokenizer"
    input_path = tmp_path / "valid.txt"
    output_path = tmp_path / "valid.npy"
    save_tokenizer_artifacts(
        tokenizer_dir,
        vocab={0: b"a", 1: b"b", 2: b"ab", 3: b"<|endoftext|>"},
        merges=[(b"a", b"b")],
        special_tokens=["<|endoftext|>"],
    )
    input_path.write_text("abab", encoding="utf-8")

    encode_validation_dataset(
        input_path=input_path,
        output_path=output_path,
        tokenizer_dir=tokenizer_dir,
        dtype="uint16",
        show_progress=False,
    )

    tokens = np.load(output_path)
    np.testing.assert_array_equal(tokens, np.array([2, 2], dtype=np.uint16))
