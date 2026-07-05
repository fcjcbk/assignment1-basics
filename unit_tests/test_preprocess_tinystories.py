import json
from pathlib import Path

import numpy as np

from cs336_basics.preprocess_tinystories import preprocess_tinystories


def test_preprocess_tinystories_writes_tokens_and_config(tmp_path: Path):
    input_path = tmp_path / "tiny.txt"
    output_path = tmp_path / "tokens.npy"
    tokenizer_dir = tmp_path / "tokenizer"
    config_path = tmp_path / "train_config.json"
    input_path.write_text("Once upon a time\nOnce again", encoding="utf-8")

    preprocess_tinystories(
        input_path=input_path,
        output_path=output_path,
        tokenizer_dir=tokenizer_dir,
        vocab_size=270,
        special_tokens=["<|endoftext|>"],
        dtype="int64",
        train_config_out=config_path,
    )

    tokens = np.load(output_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert tokens.ndim == 1
    assert tokens.dtype == np.int64
    assert len(tokens) > 2
    assert config["data"]["train_path"] == str(output_path)
    assert config["model"]["vocab_size"] == 270
    assert (tokenizer_dir / "vocab.base64.json").exists()
    assert (tokenizer_dir / "merges.base64.json").exists()


def test_preprocess_tinystories_streams_input_when_encoding(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "tiny.txt"
    output_path = tmp_path / "tokens.npy"
    tokenizer_dir = tmp_path / "tokenizer"
    input_path.write_text("Once upon a time\nOnce again\n", encoding="utf-8")

    original_read_text = Path.read_text

    def fail_on_input_read_text(self: Path, *args, **kwargs):
        if self == input_path:
            raise AssertionError("preprocess_tinystories should stream input text when encoding")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_on_input_read_text)

    preprocess_tinystories(
        input_path=input_path,
        output_path=output_path,
        tokenizer_dir=tokenizer_dir,
        vocab_size=270,
        special_tokens=["<|endoftext|>"],
        dtype="uint16",
        train_config_out=None,
    )

    tokens = np.load(output_path)

    assert tokens.ndim == 1
    assert tokens.dtype == np.uint16
    assert len(tokens) > 2
