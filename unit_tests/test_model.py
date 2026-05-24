import torch
from torch import nn
import einx

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
    ...