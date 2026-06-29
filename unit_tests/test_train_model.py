from pathlib import Path
import os
import signal

import torch

import cs336_basics.train_model as train_model
from cs336_basics.train_model import (
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


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
        batch_size=2,
        total_steps=2,
        device="cpu",
        seed=1337,
    )

    train_model.train(config)

    step_1_checkpoint = torch.load(tmp_path / "checkpoint_step_1.pt")
    step_2_checkpoint = torch.load(tmp_path / "checkpoint_step_2.pt")
    assert step_1_checkpoint["iteration"] == 1
    assert step_2_checkpoint["iteration"] == 2
    assert "model_state" in step_2_checkpoint
    assert "optimizer_state" in step_2_checkpoint
    assert "step=2/2" in log_path.read_text()


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
        batch_size=2,
        total_steps=4,
        device="cpu",
        seed=1337,
    )

    train_model.train(config)

    checkpoint = torch.load(tmp_path / "checkpoint_step_1.pt")
    assert checkpoint["iteration"] == 1
    assert not (tmp_path / "checkpoint_step_2.pt").exists()
