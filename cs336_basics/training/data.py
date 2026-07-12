from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cs336_basics.training.config import DataConfig, TrainingConfig


def load_dataset(config: DataConfig, vocab_size: int, logger: logging.Logger) -> np.ndarray:
    if config.train_path is None:
        logger.warning("No data.train_path provided; using a tiny synthetic token dataset for smoke testing.")
        return (np.arange(config.synthetic_num_tokens, dtype=np.int64) % vocab_size).astype(np.int64)

    path = Path(config.train_path)
    if path.suffix == ".npy":
        mmap_mode = "r" if config.use_memmap else None
        dataset = np.load(path, mmap_mode=mmap_mode)
    else:
        dataset = np.fromfile(path, dtype=np.dtype(config.dtype))

    if dataset.ndim != 1:
        raise ValueError(f"Expected a 1D token dataset, got shape {dataset.shape}")
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two tokens")

    logger.info("Loaded dataset from %s with %d tokens", path, len(dataset))
    return dataset


def load_validation_dataset(config: TrainingConfig, valid_path: str, logger: logging.Logger) -> np.ndarray:
    return load_dataset(
        DataConfig(train_path=valid_path, dtype=config.data.dtype, use_memmap=config.data.use_memmap),
        config.model.vocab_size,
        logger,
    )
