import torch
from torch import nn
import einx
import numpy
import torch.nn.functional as F

from cs336_basics.model.linear import Linear
from cs336_basics.model import funtional


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

    layer = Linear(3, 2)
    layer.weight = nn.Parameter(weights)
       
    actual = layer(x)
    expected = x @ weights.T

    # print("actual:", actual)
    # print("expected:", expected)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 2, 2)

def test_einx():
    a = torch.tensor(
        [[1, 2, 3, 4],[5,6,7,8]])
    # double_theta = y = torch.repeat_interleave(a, repeats=2, dim=-1)
    # print(double_theta)

    y = reverse_pairs_last_dim(a)
    print(y)


def reverse_pairs_last_dim(x):
    *batch, L = x.shape
    assert L % 2 == 0, "最后一维长度必须是偶数"
    return x.view(*batch, L // 2, 2).flip(-1).reshape(*batch, L)


def test_soft_max():
    t = torch.tensor([
        [[1, 2, 3], [3, 4, 5]],
        [[5, 6, 7], [7, 8, 9]],
        [[9, 10, 11], [11, 12, 13]]
        ]).float()
    m = nn.Softmax(dim=1)
    # print("expected: ", m(t))
    print("actual: ", funtional.softmax(t, 0))
    expected = m(t)
    actual = funtional.softmax(t, 1)
    torch.testing.assert_close(actual, expected)

    # torch.max()
    print("shape: ", t.shape)
    print("max: ", t.max(0, keepdim=True))
    print("sum: ", t.sum(0, keepdim=True))
    # print("max1: ", t.max(1, keepdim=True))
    # print("max2: ", t.max(2, keepdim=True))


def test_attention():
    nn.MultiheadAttention
    ...

def test_cross_entropy():
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    # print("expected_soft: ", F.softmax(inputs.view(-1, inputs.size(-1)), -1))
    numpy.testing.assert_allclose(
        funtional.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1)).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-4,
    )


def test_cross_entropy_is_stable_for_large_logits():
    inputs = torch.tensor(
        [
            [10000.0, 9999.0, 9998.0],
            [-10000.0, -9999.0, -10001.0],
        ]
    )
    targets = torch.tensor([0, 1])

    actual = funtional.cross_entropy(inputs, targets)

    assert torch.isfinite(actual)
    expected = F.cross_entropy(inputs, targets)
    torch.testing.assert_close(actual, expected)
