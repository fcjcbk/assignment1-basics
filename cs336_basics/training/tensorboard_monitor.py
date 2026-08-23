from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from cs336_basics.training.config import TensorBoardConfig, TrainingConfig


class TensorBoardMonitor:
    def __init__(self, config: TensorBoardConfig, *, purge_step: int | None = None):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(
            log_dir=str(self.log_dir),
            purge_step=purge_step,
            flush_secs=config.flush_secs,
        )

    def record_config(self, config: TrainingConfig, *, step: int = 0) -> None:
        config_json = json.dumps(asdict(config), indent=2)
        self._writer.add_text("Run/config", f"```json\n{config_json}\n```", global_step=step)

    def record_train_metrics(
        self,
        step: int,
        loss: float,
        *,
        learning_rate: float,
        steps_per_second: float,
        elapsed_seconds: float,
        force: bool = False,
    ) -> None:
        if not (force or step == 1 or step % self.config.interval == 0):
            return

        self._add_scalar("Loss/train", loss, step)
        self._add_scalar("Optimization/learning_rate", learning_rate, step)
        self._add_scalar("Performance/steps_per_second", steps_per_second, step)
        self._add_scalar("Performance/elapsed_seconds", elapsed_seconds, step)

    def record_validation_loss(self, step: int, loss: float) -> None:
        self._add_scalar("Loss/validation", loss, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    def _add_scalar(self, tag: str, value: float, step: int) -> None:
        if math.isfinite(value):
            self._writer.add_scalar(tag, value, global_step=step)
