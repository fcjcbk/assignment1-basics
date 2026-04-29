from torch import Tensor
import torch
import einx
from jaxtyping import Float


def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)
