import json
from dataclasses import replace
from pathlib import Path
import os
import signal

import torch
import numpy as np
import pytest
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
    TrainingConfig,
    WandbConfig,
)
from cs336_basics.training.plotting import LossCurvePlotter


class _FakeWandbRun:
    def __init__(self, init_kwargs):
        self.init_kwargs = init_kwargs
        self.url = "https://wandb.example/runs/test"
        self.defined_metrics = []
        self.logged_metrics = []
        self.finished = False

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, metrics):
        self.logged_metrics.append(metrics)

    def finish(self):
        self.finished = True


def _capture_wandb_runs(monkeypatch):
    runs = []

    def fake_init(**kwargs):
        run = _FakeWandbRun(kwargs)
        runs.append(run)
        return run

    monkeypatch.setattr("cs336_basics.training.wandb_monitor.wandb.init", fake_init)
    return runs


def _tiny_training_config(
    tmp_path: Path,
    *,
    checkpoint_path: Path | None = None,
    log_path: Path | None = None,
    plot_path: Path | None = None,
    wandb_log_dir: Path | None = None,
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
        ),
        wandb=WandbConfig(
            enabled=wandb_log_dir is not None,
            project="assignment1-tests",
            mode="offline",
            log_dir=str(wandb_log_dir or tmp_path / "wandb"),
            interval=1,
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
        plot=LossPlotConfig(enabled=True, path=str(plot_path), width=480, height=320, dpi=100),
        run=RunConfig(name="plot-run"),
        batch_size=2,
        total_steps=2,
        device="cpu",
        seed=1337,
    )

    train_model.train(config)

    assert (tmp_path / "plot-run" / "loss_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_train_renders_matplotlib_loss_curve_only_once_at_end(tmp_path: Path, monkeypatch):
    render_train_point_counts: list[int] = []
    original_render = LossCurvePlotter.render

    def track_render(plotter: LossCurvePlotter) -> None:
        render_train_point_counts.append(len(plotter.train_points))
        original_render(plotter)

    monkeypatch.setattr(LossCurvePlotter, "render", track_render)

    train_model.train(
        _tiny_training_config(
            tmp_path,
            plot_path=tmp_path / "loss_curve.png",
            run_name="plot-once-run",
        )
    )

    assert render_train_point_counts == [2]


def test_train_reports_metrics_and_config_to_wandb(tmp_path: Path, monkeypatch):
    valid_path = tmp_path / "valid.npy"
    wandb_log_dir = tmp_path / "wandb"
    runs = _capture_wandb_runs(monkeypatch)
    np.save(valid_path, np.arange(128, dtype=np.int64) % 32)
    config = _tiny_training_config(
        tmp_path,
        wandb_log_dir=wandb_log_dir,
        run_name="wandb-run",
    )
    config = replace(
        config,
        eval=EvalConfig(valid_path=str(valid_path), interval=1, mode="sampled", num_batches=1, batch_size=2),
    )

    train_model.train(config)

    assert len(runs) == 1
    run = runs[0]
    assert run.init_kwargs["project"] == "assignment1-tests"
    assert run.init_kwargs["name"] == "wandb-run"
    assert run.init_kwargs["mode"] == "offline"
    assert run.init_kwargs["dir"] == str(wandb_log_dir / "wandb-run")
    assert run.init_kwargs["config"]["run"]["name"] == "wandb-run"
    assert (("global_step",), {}) in run.defined_metrics
    assert (("*",), {"step_metric": "global_step"}) in run.defined_metrics

    reported_metrics = {name for log in run.logged_metrics for name in log}
    assert {
        "Loss/train",
        "Loss/validation",
        "Optimization/learning_rate",
        "Performance/steps_per_second",
        "Performance/elapsed_seconds",
    } <= reported_metrics
    assert [log["global_step"] for log in run.logged_metrics if "Loss/train" in log] == [1, 2]
    assert [log["global_step"] for log in run.logged_metrics if "Loss/validation" in log] == [1, 2]
    assert run.finished


def test_resumed_training_starts_a_new_wandb_run_at_the_resumed_step(tmp_path: Path, monkeypatch):
    wandb_log_dir = tmp_path / "wandb"
    runs = _capture_wandb_runs(monkeypatch)
    first_config = _tiny_training_config(
        tmp_path,
        wandb_log_dir=wandb_log_dir,
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

    assert len(runs) == 2
    assert [log["global_step"] for log in runs[0].logged_metrics if "Loss/train" in log] == [1, 2, 3]
    assert [log["global_step"] for log in runs[1].logged_metrics if "Loss/train" in log] == [3, 4]
    assert all(run.finished for run in runs)


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


def test_load_config_loads_wandb_settings(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "wandb": {
                    "enabled": True,
                    "project": "custom-project",
                    "entity": "custom-team",
                    "mode": "offline",
                    "log_dir": "custom/wandb",
                    "interval": 25,
                    "tags": ["tiny", "smoke"],
                }
            }
        ),
        encoding="utf-8",
    )

    config = train_model.load_config(config_path)

    assert config.wandb == WandbConfig(
        enabled=True,
        project="custom-project",
        entity="custom-team",
        mode="offline",
        log_dir="custom/wandb",
        interval=25,
        tags=["tiny", "smoke"],
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
