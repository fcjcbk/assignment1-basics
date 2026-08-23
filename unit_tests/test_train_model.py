import json
from dataclasses import replace
from pathlib import Path
import os
import signal

import torch
import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchinfo import summary as torchinfo_summary

import cs336_basics.train_model as train_model
from cs336_basics.model.funtional import save_checkpoint
from cs336_basics.train_model import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    LossPlotConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    TensorBoardConfig,
    TrainingConfig,
)


def _tiny_training_config(
    tmp_path: Path,
    *,
    checkpoint_path: Path | None = None,
    log_path: Path | None = None,
    plot_path: Path | None = None,
    tensorboard_log_dir: Path | None = None,
    run_name: str | None = None,
) -> TrainingConfig:
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
        checkpoint=CheckpointConfig(path=str(checkpoint_path or tmp_path / "checkpoint.pt"), save_interval=1),
        logging=LoggingConfig(log_interval=1, log_file=str(log_path or tmp_path / "train.log")),
        plot=LossPlotConfig(
            enabled=plot_path is not None,
            path=str(plot_path or tmp_path / "loss_curve.png"),
            interval=1,
        ),
        tensorboard=TensorBoardConfig(
            enabled=tensorboard_log_dir is not None,
            log_dir=str(tensorboard_log_dir or tmp_path / "tensorboard"),
            interval=1,
            flush_secs=1,
        ),
        run=RunConfig(name=run_name),
        batch_size=2,
        total_steps=2,
        device="cpu",
        seed=1337,
    )


def test_torchinfo_summary_outputs_current_model_structure():
    config_path = Path(__file__).resolve().parents[1] / "train_config.json"
    config = train_model.load_config(config_path)
    model = train_model.build_model(config.model, "cpu")

    model_summary = torchinfo_summary(
        model,
        input_size=(1, config.model.context_length),
        dtypes=[torch.long],
        device="cpu",
        depth=4,
        verbose=1,
    )

    summary_text = str(model_summary)
    assert "TransformerLanguageModel" in summary_text
    assert "MultiHeadSelfAttentionWithRoPE" in summary_text
    assert "SwiGLu" in summary_text
    assert model_summary.total_params == sum(parameter.numel() for parameter in model.parameters())


def test_train_writes_step_specific_checkpoints_and_log(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    log_path = tmp_path / "train.log"

    config = TrainingConfig(
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
        checkpoint=CheckpointConfig(path=str(checkpoint_path), save_interval=1),
        logging=LoggingConfig(log_interval=1, log_file=str(log_path)),
        run=RunConfig(name="checkpoint-log-run"),
        batch_size=2,
        total_steps=2,
        device="cpu",
        seed=1337,
    )

    train_model.train(config)

    run_dir = tmp_path / "checkpoint-log-run"
    step_1_checkpoint = torch.load(run_dir / "checkpoint_step_1.pt")
    step_2_checkpoint = torch.load(run_dir / "checkpoint_step_2.pt")
    assert step_1_checkpoint["iteration"] == 1
    assert step_2_checkpoint["iteration"] == 2
    assert "model_state" in step_2_checkpoint
    assert "optimizer_state" in step_2_checkpoint
    assert "step=2/2" in (run_dir / "train.log").read_text()


def test_train_saves_current_step_checkpoint_before_exiting_on_sigint(tmp_path: Path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    original_sample_train_data = train_model.sample_train_data
    sent_signal = False

    def interrupt_after_sampling_current_step(*args, **kwargs):
        nonlocal sent_signal
        batch = original_sample_train_data(*args, **kwargs)
        if not sent_signal:
            sent_signal = True
            os.kill(os.getpid(), signal.SIGINT)
        return batch

    monkeypatch.setattr(train_model, "sample_train_data", interrupt_after_sampling_current_step)

    config = TrainingConfig(
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
        checkpoint=CheckpointConfig(path=str(checkpoint_path), save_interval=0),
        logging=LoggingConfig(log_interval=1),
        run=RunConfig(name="sigint-run"),
        batch_size=2,
        total_steps=4,
        device="cpu",
        seed=1337,
    )

    train_model.train(config)

    checkpoint = torch.load(tmp_path / "sigint-run" / "checkpoint_step_1.pt")
    assert checkpoint["iteration"] == 1
    assert not (tmp_path / "sigint-run" / "checkpoint_step_2.pt").exists()


def test_train_logs_validation_loss_when_eval_config_is_enabled(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    log_path = tmp_path / "train.log"
    valid_path = tmp_path / "valid.npy"
    np.save(valid_path, np.arange(128, dtype=np.int64) % 32)

    config = TrainingConfig(
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
        checkpoint=CheckpointConfig(path=str(checkpoint_path), save_interval=0),
        logging=LoggingConfig(log_interval=1, log_file=str(log_path)),
        eval=EvalConfig(valid_path=str(valid_path), interval=1, mode="sampled", num_batches=1, batch_size=2),
        run=RunConfig(name="validation-log-run"),
        batch_size=2,
        total_steps=2,
        device="cpu",
        seed=1337,
    )

    train_model.train(config)

    log_text = (tmp_path / "validation-log-run" / "train.log").read_text()
    assert "val_loss=" in log_text
    assert "val_ppl=" in log_text
    assert "eval_tokens=16" in log_text


def test_train_writes_matplotlib_loss_curve_when_plot_config_is_enabled(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    plot_path = tmp_path / "loss_curve.png"
    valid_path = tmp_path / "valid.npy"
    np.save(valid_path, np.arange(128, dtype=np.int64) % 32)

    config = TrainingConfig(
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
        checkpoint=CheckpointConfig(path=str(checkpoint_path), save_interval=0),
        logging=LoggingConfig(log_interval=1),
        eval=EvalConfig(valid_path=str(valid_path), interval=1, mode="sampled", num_batches=1, batch_size=2),
        plot=LossPlotConfig(enabled=True, path=str(plot_path), interval=1, width=480, height=320, dpi=100),
        run=RunConfig(name="plot-run"),
        batch_size=2,
        total_steps=2,
        device="cpu",
        seed=1337,
    )

    train_model.train(config)

    assert (tmp_path / "plot-run" / "loss_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_train_writes_tensorboard_metrics_to_run_directory(tmp_path: Path):
    valid_path = tmp_path / "valid.npy"
    tensorboard_log_dir = tmp_path / "tensorboard"
    np.save(valid_path, np.arange(128, dtype=np.int64) % 32)
    config = _tiny_training_config(
        tmp_path,
        tensorboard_log_dir=tensorboard_log_dir,
        run_name="tensorboard-run",
    )
    config = replace(
        config,
        eval=EvalConfig(valid_path=str(valid_path), interval=1, mode="sampled", num_batches=1, batch_size=2),
    )

    train_model.train(config)

    events = EventAccumulator(str(tensorboard_log_dir / "tensorboard-run"))
    events.Reload()
    scalar_tags = set(events.Tags()["scalars"])
    assert {
        "Loss/train",
        "Loss/validation",
        "Optimization/learning_rate",
        "Performance/steps_per_second",
        "Performance/elapsed_seconds",
    } <= scalar_tags
    assert [event.step for event in events.Scalars("Loss/train")] == [1, 2]
    assert [event.step for event in events.Scalars("Loss/validation")] == [1, 2]


def test_resumed_training_purges_overlapping_tensorboard_steps(tmp_path: Path):
    tensorboard_log_dir = tmp_path / "tensorboard"
    first_config = _tiny_training_config(
        tmp_path,
        tensorboard_log_dir=tensorboard_log_dir,
        run_name="resume-run",
    )
    train_model.train(replace(first_config, total_steps=3))

    resumed_config = replace(
        first_config,
        checkpoint=replace(
            first_config.checkpoint,
            resume_from=str(tmp_path / "resume-run" / "checkpoint_step_2.pt"),
        ),
        total_steps=4,
    )
    train_model.train(resumed_config)

    events = EventAccumulator(str(tensorboard_log_dir / "resume-run"))
    events.Reload()
    assert [event.step for event in events.Scalars("Loss/train")] == [1, 2, 3, 4]


def test_train_uses_manual_run_name_for_all_training_artifacts(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoints" / "model.pt"
    log_path = tmp_path / "logs" / "train.log"
    plot_path = tmp_path / "plots" / "loss_curve.png"

    config = _tiny_training_config(
        tmp_path,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        plot_path=plot_path,
        run_name="smoke-run",
    )

    train_model.train(config)

    assert (tmp_path / "checkpoints" / "smoke-run" / "model_step_1.pt").exists()
    assert "run=smoke-run" in (tmp_path / "logs" / "smoke-run" / "train.log").read_text()
    assert (tmp_path / "plots" / "smoke-run" / "loss_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_train_generates_unique_run_names_when_not_configured(tmp_path: Path):
    train_model.train(_tiny_training_config(tmp_path))
    train_model.train(_tiny_training_config(tmp_path))

    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())

    assert len(run_dirs) == 2
    assert run_dirs[0].name != run_dirs[1].name
    assert all(path.name.startswith("train-") for path in run_dirs)
    assert all((path / "checkpoint_step_1.pt").exists() for path in run_dirs)
    assert all((path / "train.log").exists() for path in run_dirs)


def test_load_config_loads_manual_run_name(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run": {"name": "smoke-run"}}), encoding="utf-8")

    config = train_model.load_config(config_path)

    assert config.run.name == "smoke-run"


def test_load_config_loads_tensorboard_settings(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tensorboard": {
                    "enabled": True,
                    "log_dir": "custom/events",
                    "interval": 25,
                    "flush_secs": 5,
                }
            }
        ),
        encoding="utf-8",
    )

    config = train_model.load_config(config_path)

    assert config.tensorboard == TensorBoardConfig(
        enabled=True,
        log_dir="custom/events",
        interval=25,
        flush_secs=5,
    )


@pytest.mark.parametrize("run_name", ["", ".", "..", "nested/name", r"nested\name", "bad..name", "bad name"])
def test_load_config_rejects_unsafe_run_names(tmp_path: Path, run_name: str):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run": {"name": run_name}}), encoding="utf-8")

    with pytest.raises(ValueError, match="run.name"):
        train_model.load_config(config_path)


def test_evaluate_checkpoint_loads_checkpoint_and_returns_validation_loss(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    valid_path = tmp_path / "valid.npy"
    np.save(valid_path, np.arange(128, dtype=np.int64) % 32)

    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=32,
            context_length=8,
            num_layers=1,
            d_model=16,
            num_heads=4,
            max_seq_len=8,
            d_ff=32,
        ),
        optimizer=OptimizerConfig(),
        eval=EvalConfig(valid_path=str(valid_path), mode="sampled", num_batches=1, batch_size=2),
        device="cpu",
        seed=1337,
    )
    model = train_model.build_model(config.model, "cpu")
    optimizer = train_model.build_optimizer(model, config.optimizer)
    save_checkpoint(model, optimizer, 7, checkpoint_path)

    result = train_model.evaluate_checkpoint(config, checkpoint_path)

    assert result.checkpoint_iteration == 7
    assert result.mode == "sampled"
    assert result.token_count == 16
    assert result.loss > 0
