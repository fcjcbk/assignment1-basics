from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 32
    num_layers: int = 2
    d_model: int = 64
    num_heads: int = 4
    max_seq_len: int = 32
    theta: float = 10_000.0
    d_ff: int | None = None


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 10
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float | None = 1.0


@dataclass
class DataConfig:
    train_path: str | None = None
    dtype: str = "int64"
    use_memmap: bool = True
    synthetic_num_tokens: int = 4096


@dataclass
class CheckpointConfig:
    path: str = "checkpoints/latest.pt"
    save_interval: int = 100
    resume_from: str | None = None


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_interval: int = 10
    log_file: str | None = None


@dataclass
class EvalConfig:
    valid_path: str | None = None
    interval: int = 1000
    mode: str = "sampled"
    num_batches: int = 50
    batch_size: int | None = None


@dataclass
class LossPlotConfig:
    enabled: bool = False
    path: str = "log/loss_curve.png"
    interval: int = 10
    width: int = 960
    height: int = 540
    dpi: int = 120


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    plot: LossPlotConfig = field(default_factory=LossPlotConfig)
    batch_size: int = 8
    total_steps: int = 100
    device: str = "auto"
    seed: int = 1337


def load_config(config_path: str | Path | None) -> TrainingConfig:
    config = TrainingConfig()
    if config_path is None:
        return config

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        overrides = json.load(f)
    if not isinstance(overrides, dict):
        raise ValueError("Training config JSON must contain an object at the top level")

    return _merge_dataclass(config, overrides)


def example_config_json() -> str:
    return json.dumps(asdict(TrainingConfig()), indent=2)


def print_example_config() -> None:
    print(example_config_json())


def _merge_dataclass(default: Any, overrides: dict[str, Any]) -> Any:
    if not is_dataclass(default):
        return overrides

    values = {item.name: getattr(default, item.name) for item in fields(default)}
    field_types = {item.name: item.type for item in fields(default)}

    for key, value in overrides.items():
        if key not in values:
            raise ValueError(f"Unknown config field: {key}")
        current_value = getattr(default, key)
        if is_dataclass(current_value):
            if not isinstance(value, dict):
                raise ValueError(f"Config field '{key}' must be an object")
            values[key] = _merge_dataclass(current_value, value)
        else:
            values[key] = value

    return type(default)(**{key: _coerce_optional_none(value, field_types[key]) for key, value in values.items()})


def _coerce_optional_none(value: Any, field_type: Any) -> Any:
    if value != "none":
        return value
    if "NoneType" in str(field_type) or "None" in str(field_type):
        return None
    return value
