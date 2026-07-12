from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from cs336_basics.training.config import LossPlotConfig


class LossCurvePlotter:
    def __init__(self, config: LossPlotConfig, total_steps: int):
        if config.interval <= 0:
            raise ValueError("plot.interval must be positive")
        if config.width < 320:
            raise ValueError("plot.width must be at least 320")
        if config.height < 240:
            raise ValueError("plot.height must be at least 240")
        if config.dpi <= 0:
            raise ValueError("plot.dpi must be positive")

        self.config = config
        self.total_steps = total_steps
        self.path = Path(config.path)
        self.train_points: list[tuple[int, float]] = []
        self.validation_points: list[tuple[int, float]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._plt = _load_matplotlib_pyplot()

    def record_train_loss(self, step: int, loss: float) -> None:
        self._append_point(self.train_points, step, loss)

    def record_validation_loss(self, step: int, loss: float) -> None:
        self._append_point(self.validation_points, step, loss)

    def maybe_render(self, step: int, force: bool = False) -> None:
        if force or step == 1 or step % self.config.interval == 0:
            self.render()

    def render(self) -> None:
        render_loss_curve_png(
            plt=self._plt,
            train_points=self.train_points,
            validation_points=self.validation_points,
            total_steps=self.total_steps,
            path=self.path,
            width=self.config.width,
            height=self.config.height,
            dpi=self.config.dpi,
        )

    @staticmethod
    def _append_point(points: list[tuple[int, float]], step: int, loss: float) -> None:
        if math.isfinite(loss):
            points.append((step, loss))


def render_loss_curve_png(
    plt: Any,
    train_points: list[tuple[int, float]],
    validation_points: list[tuple[int, float]],
    total_steps: int,
    path: Path,
    width: int = 960,
    height: int = 540,
    dpi: int = 120,
) -> None:
    figure_size = (width / dpi, height / dpi)
    figure, axis = plt.subplots(figsize=figure_size, dpi=dpi)
    tmp_path = path.with_suffix(".tmp.png")
    all_points = [*train_points, *validation_points]

    try:
        if train_points:
            train_steps, train_losses = zip(*train_points, strict=True)
            axis.plot(train_steps, train_losses, color="#2563eb", linewidth=2.0, label="train_loss")
            axis.scatter(train_steps[-1], train_losses[-1], color="#1d4ed8", s=26, zorder=3)

        if validation_points:
            validation_steps, validation_losses = zip(*validation_points, strict=True)
            axis.plot(
                validation_steps,
                validation_losses,
                color="#dc2626",
                linewidth=2.0,
                marker="o",
                markersize=4,
                label="val_loss",
            )
            axis.scatter(validation_steps[-1], validation_losses[-1], color="#b91c1c", s=32, zorder=3)

        axis.set_title("Training and validation loss")
        axis.set_xlabel("step")
        axis.set_ylabel("loss")
        max_observed_step = max((step for step, _ in all_points), default=1)
        axis.set_xlim(left=0, right=max(total_steps, max_observed_step, 1))
        axis.grid(True, alpha=0.25)

        if all_points:
            latest = " | ".join(
                piece
                for piece in (
                    _format_latest_loss("train_loss", train_points),
                    _format_latest_loss("val_loss", validation_points),
                )
                if piece
            )
            axis.text(
                0.5,
                1.02,
                latest,
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#4b5563",
            )
            axis.legend(loc="best")
        else:
            axis.text(0.5, 0.5, "Waiting for loss data", transform=axis.transAxes, ha="center", va="center")

        figure.tight_layout()
        figure.savefig(tmp_path, format="png")
        tmp_path.replace(path)
    finally:
        plt.close(figure)


def _load_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required when plot.enabled is true") from exc
    return plt


def _format_latest_loss(label: str, points: list[tuple[int, float]]) -> str:
    if not points:
        return ""
    step, loss = points[-1]
    return f"{label}={loss:.6f} @ step {step}"
