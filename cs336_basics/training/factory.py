from __future__ import annotations

import torch

from cs336_basics.model.optimizer import AdamW
from cs336_basics.model.transformer_language_model import TransformerLanguageModel
from cs336_basics.training.config import ModelConfig, OptimizerConfig


def build_model(config: ModelConfig, device: str) -> TransformerLanguageModel:
    return TransformerLanguageModel(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        theta=config.theta,
        d_ff=config.d_ff,
        device=torch.device(device),
    )


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
