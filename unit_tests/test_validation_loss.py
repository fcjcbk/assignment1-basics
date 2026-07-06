import math

import numpy as np
import torch

from cs336_basics.evaluation import evaluate_loss_full, evaluate_loss_sampled


class UniformLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*inputs.shape, self.vocab_size, device=inputs.device)


def test_sampled_validation_loss_reports_token_count_and_perplexity():
    vocab_size = 8
    model = UniformLogitModel(vocab_size)
    dataset = np.arange(64, dtype=np.int64) % vocab_size

    result = evaluate_loss_sampled(
        model=model,
        dataset=dataset,
        batch_size=3,
        context_length=5,
        num_batches=4,
        device="cpu",
    )

    assert result.mode == "sampled"
    assert result.token_count == 3 * 5 * 4
    assert math.isclose(result.loss, math.log(vocab_size), rel_tol=1e-5)
    assert math.isclose(result.perplexity, vocab_size, rel_tol=1e-5)


def test_full_validation_loss_drops_incomplete_final_chunk():
    vocab_size = 8
    model = UniformLogitModel(vocab_size)
    dataset = np.arange(23, dtype=np.int64) % vocab_size

    result = evaluate_loss_full(
        model=model,
        dataset=dataset,
        batch_size=3,
        context_length=4,
        device="cpu",
    )

    assert result.mode == "full"
    assert result.token_count == 20
    assert math.isclose(result.loss, math.log(vocab_size), rel_tol=1e-5)
