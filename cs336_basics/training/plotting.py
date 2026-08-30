from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from cs336_basics.training.config import LossPlotConfig


class LossCurvePlotter:
    def __init__(self, config: LossPlotConfig, total_steps: int):
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
        self.learning_rate_points: list[tuple[int, float]] = []
        self.steps_per_second_points: list[tuple[int, float]] = []
        self.elapsed_seconds: float | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._plt = _load_matplotlib_pyplot()

    def record_train_loss(
        self,
        step: int,
        loss: float,
        *,
        learning_rate: float | None = None,
        steps_per_second: float | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        self._append_point(self.train_points, step, loss)
        if learning_rate is not None:
            self._append_point(self.learning_rate_points, step, learning_rate)
        if steps_per_second is not None:
            self._append_point(self.steps_per_second_points, step, steps_per_second)
        if elapsed_seconds is not None and math.isfinite(elapsed_seconds):
            self.elapsed_seconds = elapsed_seconds

    def record_validation_loss(self, step: int, loss: float) -> None:
        self._append_point(self.validation_points, step, loss)

    def render(self) -> None:
        render_loss_curve_png(
            plt=self._plt,
            train_points=self.train_points,
            validation_points=self.validation_points,
            learning_rate_points=self.learning_rate_points,
            steps_per_second_points=self.steps_per_second_points,
            elapsed_seconds=self.elapsed_seconds,
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
    width: int = 1000,
    height: int = 720,
    dpi: int = 120,
    learning_rate_points: list[tuple[int, float]] | None = None,
    steps_per_second_points: list[tuple[int, float]] | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    figure_size = (width / dpi, height / dpi)
    figure = plt.figure(figsize=figure_size, dpi=dpi)
    try:
        _draw_training_monitor(
            figure=figure,
            train_points=train_points,
            validation_points=validation_points,
            learning_rate_points=learning_rate_points or [],
            steps_per_second_points=steps_per_second_points or [],
            elapsed_seconds=elapsed_seconds,
            total_steps=total_steps,
        )
        _save_figure_png(figure, path)
    finally:
        plt.close(figure)


def _draw_training_monitor(
    *,
    figure: Any,
    train_points: list[tuple[int, float]],
    validation_points: list[tuple[int, float]],
    learning_rate_points: list[tuple[int, float]],
    steps_per_second_points: list[tuple[int, float]],
    elapsed_seconds: float | None,
    total_steps: int,
) -> None:
    figure.clear()
    grid = figure.add_gridspec(nrows=3, ncols=1, height_ratios=(3.4, 1.2, 1.0), hspace=0.42)
    loss_axis = figure.add_subplot(grid[0])
    lr_axis = figure.add_subplot(grid[1], sharex=loss_axis)
    status_axis = figure.add_subplot(grid[2])
    all_step_points = [*train_points, *validation_points, *learning_rate_points, *steps_per_second_points]

    if train_points:
        train_steps, train_losses = zip(*train_points, strict=True)
        loss_axis.plot(
            train_steps,
            train_losses,
            color="#2563eb",
            linewidth=1.1,
            alpha=0.35,
            label="train loss raw",
        )
        smoothed_train_points = _smooth_points(train_points, window=min(25, max(3, len(train_points) // 12)))
        if len(smoothed_train_points) >= 2:
            smooth_steps, smooth_losses = zip(*smoothed_train_points, strict=True)
            loss_axis.plot(smooth_steps, smooth_losses, color="#1d4ed8", linewidth=2.2, label="train loss trend")
        loss_axis.scatter(train_steps[-1], train_losses[-1], color="#1d4ed8", s=28, zorder=4)

    if validation_points:
        validation_steps, validation_losses = zip(*validation_points, strict=True)
        loss_axis.plot(
            validation_steps,
            validation_losses,
            color="#dc2626",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="validation loss",
        )
        latest_validation_point = validation_points[-1]
        best_validation_point = min(validation_points, key=lambda point: point[1])
        loss_axis.scatter(*latest_validation_point, color="#b91c1c", s=32, zorder=4)
        loss_axis.scatter(
            *best_validation_point,
            color="#f59e0b",
            edgecolors="#92400e",
            marker="*",
            s=96,
            zorder=5,
            label="best validation",
        )

    loss_axis.set_ylabel("loss")
    loss_axis.grid(True, alpha=0.25)

    if learning_rate_points:
        lr_steps, learning_rates = zip(*learning_rate_points, strict=True)
        lr_axis.plot(lr_steps, learning_rates, color="#059669", linewidth=1.8, label="learning rate")
        lr_axis.fill_between(lr_steps, learning_rates, color="#10b981", alpha=0.14)
        lr_axis.scatter(lr_steps[-1], learning_rates[-1], color="#047857", s=20, zorder=3)
    else:
        lr_axis.text(
            0.5,
            0.5,
            "Learning rate will appear after the first training step",
            transform=lr_axis.transAxes,
            ha="center",
            va="center",
            color="#6b7280",
        )

    max_observed_step = max((step for step, _ in all_step_points), default=0)
    x_right = _observed_step_axis_right(max_observed_step)
    loss_axis.set_xlim(left=0, right=x_right)
    lr_axis.set_xlim(left=0, right=x_right)
    lr_axis.set_xlabel("step")
    lr_axis.set_ylabel("lr")
    lr_axis.grid(True, alpha=0.2)

    if train_points or validation_points:
        loss_axis.legend(loc="best", fontsize=8)
    else:
        loss_axis.text(0.5, 0.5, "Waiting for loss data", transform=loss_axis.transAxes, ha="center", va="center")

    if learning_rate_points:
        lr_axis.legend(loc="best", fontsize=8)

    _render_status_panel(
        status_axis=status_axis,
        train_points=train_points,
        validation_points=validation_points,
        learning_rate_points=learning_rate_points,
        steps_per_second_points=steps_per_second_points,
        elapsed_seconds=elapsed_seconds,
        total_steps=total_steps,
        current_step=max_observed_step,
    )

    figure.suptitle("Training monitor", fontsize=12, fontweight="bold")
    figure.subplots_adjust(top=0.88, left=0.08, right=0.98, bottom=0.08)


def _save_figure_png(figure: Any, path: Path) -> None:
    tmp_path = path.with_suffix(".tmp.png")
    figure.savefig(tmp_path, format="png")
    tmp_path.replace(path)


def _load_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required when plot.enabled is true") from exc
    return plt


def _format_latest_loss(label: str, points: list[tuple[int, float]], *, precision: int = 6) -> str:
    if not points:
        return ""
    step, loss = points[-1]
    return f"{label}={loss:.{precision}f} @ {step}"


def _smooth_points(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1 or len(points) < 3:
        return points[:]

    smoothed_points: list[tuple[int, float]] = []
    losses: list[float] = []
    for step, loss in points:
        losses.append(loss)
        start = max(0, len(losses) - window)
        smoothed_points.append((step, sum(losses[start:]) / (len(losses) - start)))
    return smoothed_points


def _observed_step_axis_right(max_observed_step: int) -> int:
    if max_observed_step <= 10:
        return 10
    return math.ceil(max_observed_step * 1.05)


def _render_status_panel(
    *,
    status_axis: Any,
    train_points: list[tuple[int, float]],
    validation_points: list[tuple[int, float]],
    learning_rate_points: list[tuple[int, float]],
    steps_per_second_points: list[tuple[int, float]],
    elapsed_seconds: float | None,
    total_steps: int,
    current_step: int,
) -> None:
    from matplotlib.patches import Rectangle

    status_axis.set_axis_off()
    progress = current_step / total_steps if total_steps > 0 else 0.0
    progress = min(max(progress, 0.0), 1.0)

    status_axis.add_patch(
        Rectangle((0.0, 0.06), 1.0, 0.22, transform=status_axis.transAxes, color="#e5e7eb", linewidth=0)
    )
    status_axis.add_patch(
        Rectangle((0.0, 0.06), progress, 0.22, transform=status_axis.transAxes, color="#2563eb", linewidth=0)
    )

    primary_status_text = " | ".join(
        piece
        for piece in (
            _format_latest_loss("train", train_points, precision=4),
            _format_latest_loss("val", validation_points, precision=4),
            _format_best_validation_loss(validation_points),
        )
        if piece
    )
    secondary_status_text = " | ".join(
        piece
        for piece in (
            _format_latest_metric("lr", learning_rate_points, precision=3),
            _format_latest_metric("steps/s", steps_per_second_points, precision=2),
            _format_elapsed_and_eta(elapsed_seconds, current_step, total_steps),
        )
        if piece
    )
    if not primary_status_text and not secondary_status_text:
        primary_status_text = "Waiting for training metrics"

    status_axis.text(
        0.0,
        0.88,
        primary_status_text,
        transform=status_axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.4,
        color="#111827",
        clip_on=True,
    )
    status_axis.text(
        0.0,
        0.58,
        secondary_status_text,
        transform=status_axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.4,
        color="#374151",
        clip_on=True,
    )
    status_axis.text(
        0.5,
        0.17,
        f"progress {current_step:,}/{total_steps:,} ({progress:.1%})",
        transform=status_axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.4,
        color="#111827" if progress < 0.55 else "#ffffff",
    )


def _format_best_validation_loss(points: list[tuple[int, float]]) -> str:
    if not points:
        return ""
    step, loss = min(points, key=lambda point: point[1])
    return f"best={loss:.4f} @ {step}"


def _format_latest_metric(label: str, points: list[tuple[int, float]], *, precision: int) -> str:
    if not points:
        return ""
    _, value = points[-1]
    return f"{label}={value:.{precision}g}"


def _format_elapsed_and_eta(elapsed_seconds: float | None, current_step: int, total_steps: int) -> str:
    if elapsed_seconds is None or elapsed_seconds <= 0 or current_step <= 0:
        return ""
    remaining_steps = max(total_steps - current_step, 0)
    eta_seconds = elapsed_seconds * remaining_steps / current_step
    return f"elapsed={_format_duration(elapsed_seconds)} eta={_format_duration(eta_seconds)}"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"
