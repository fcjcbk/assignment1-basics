import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cs336_basics import cli
from cs336_basics.data_pipeline import preprocess_tinystories as pipeline_preprocess_tinystories
from cs336_basics.data_pipeline import save_tokenizer_artifacts
from cs336_basics.model.funtional import save_checkpoint
from cs336_basics.training.config import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    TrainingConfig,
)
from cs336_basics.training.factory import build_model, build_optimizer


def _tiny_config(tmp_path: Path, *, valid_path: Path | None = None) -> TrainingConfig:
    return TrainingConfig(
        model=ModelConfig(
            vocab_size=32,
            context_length=8,
            num_layers=1,
            d_model=16,
            num_heads=4,
            max_seq_len=8,
            d_ff=32,
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_iters=1,
            max_grad_norm=1.0,
        ),
        data=DataConfig(synthetic_num_tokens=128),
        checkpoint=CheckpointConfig(path=str(tmp_path / "checkpoint.pt"), save_interval=1),
        logging=LoggingConfig(log_interval=1),
        run=RunConfig(name="cli-run"),
        eval=EvalConfig(valid_path=str(valid_path), interval=1, mode="sampled", num_batches=1, batch_size=2)
        if valid_path is not None
        else EvalConfig(),
        batch_size=2,
        total_steps=2,
        device="cpu",
        seed=1337,
    )


def _write_config(tmp_path: Path, config: TrainingConfig) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    return config_path


def test_cli_train_runs_tiny_synthetic_smoke_config(tmp_path: Path):
    config_path = _write_config(tmp_path, _tiny_config(tmp_path))

    cli.main(["train", "--config", str(config_path)])

    assert (tmp_path / "cli-run" / "checkpoint_step_2.pt").exists()


def test_cli_eval_reports_checkpoint_validation_loss(tmp_path: Path, capsys):
    valid_path = tmp_path / "valid.npy"
    np.save(valid_path, np.arange(128, dtype=np.int64) % 32)
    config = _tiny_config(tmp_path, valid_path=valid_path)
    config_path = _write_config(tmp_path, config)

    model = build_model(config.model, "cpu")
    optimizer = build_optimizer(model, config.optimizer)
    checkpoint_path = tmp_path / "eval_checkpoint.pt"
    save_checkpoint(model, optimizer, 3, checkpoint_path)

    cli.main(["eval", "--config", str(config_path), "--checkpoint", str(checkpoint_path)])

    output = capsys.readouterr().out
    assert "checkpoint_iteration=3" in output
    assert "val_loss=" in output


def test_cli_generate_requires_explicit_assets_and_uses_prompt(tmp_path: Path, capsys):
    tokenizer_dir = tmp_path / "tokenizer"
    save_tokenizer_artifacts(
        tokenizer_dir,
        vocab={0: b"a", 1: b"b", 2: b"ab", 3: b"<|endoftext|>"},
        merges=[(b"a", b"b")],
        special_tokens=["<|endoftext|>"],
    )

    config = _tiny_config(tmp_path)
    config.model.vocab_size = 4
    config_path = _write_config(tmp_path, config)
    model = build_model(config.model, "cpu")
    optimizer = build_optimizer(model, config.optimizer)
    checkpoint_path = tmp_path / "generate_checkpoint.pt"
    save_checkpoint(model, optimizer, 5, checkpoint_path)

    cli.main(
        [
            "generate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--prompt",
            "ab",
            "--max-new-tokens",
            "0",
        ]
    )

    assert capsys.readouterr().out == "ab\n"


def test_legacy_wrappers_still_expose_training_and_data_public_surface():
    import cs336_basics.preprocess_tinystories as preprocess_module
    import cs336_basics.train_model as train_model

    assert train_model.TrainingConfig is TrainingConfig
    assert train_model.RunConfig is RunConfig
    assert callable(train_model.train)
    assert callable(train_model.build_model)
    assert callable(train_model.evaluate_checkpoint)
    assert preprocess_module.preprocess_tinystories is pipeline_preprocess_tinystories
