import torch

from cs336_basics.model.linear import Linear


def test_linear_matches_matrix_multiply_for_batched_inputs():
    weights = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    x = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]],
            [[0.5, 0.5, 0.5], [-1.0, 2.0, 3.0]],
        ]
    )

    layer = Linear(d_in=3, d_out=2, weights=weights)
    actual = layer.forward(x)
    expected = x @ weights.T

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 2, 2)
