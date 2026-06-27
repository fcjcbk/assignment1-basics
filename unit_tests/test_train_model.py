from pathlib import Path

import torch

from cs336_basics.train_model import (
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    train,
)


def test_train_writes_checkpoint_and_log(tmp_path: Path):
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

    train(config)

    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["iteration"] == 2
    assert "model_state" in checkpoint
    assert "optimizer_state" in checkpoint
    assert "step=2/2" in log_path.read_text()
