from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import wandb

from cs336_basics.training.config import TrainingConfig, WandbConfig


class WandbMonitor:
    def __init__(self, config: WandbConfig, training_config: TrainingConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            project=config.project,
            entity=config.entity,
            name=training_config.run.name,
            config=asdict(training_config),
            tags=config.tags or None,
            mode=config.mode,
            dir=str(self.log_dir),
            reinit="finish_previous",
        )
        self._run.define_metric("global_step")
        self._run.define_metric("*", step_metric="global_step")

    @property
    def url(self) -> str | None:
        return self._run.url

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

        self._log(
            step,
            {
                "Loss/train": loss,
                "Optimization/learning_rate": learning_rate,
                "Performance/steps_per_second": steps_per_second,
                "Performance/elapsed_seconds": elapsed_seconds,
            },
        )

    def record_validation_loss(self, step: int, loss: float) -> None:
        self._log(step, {"Loss/validation": loss})

    def close(self) -> None:
        self._run.finish()

    def _log(self, step: int, metrics: dict[str, float]) -> None:
        finite_metrics: dict[str, Any] = {name: value for name, value in metrics.items() if math.isfinite(value)}
        if finite_metrics:
            self._run.log({"global_step": step, **finite_metrics})
