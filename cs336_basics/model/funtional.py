from torch import Tensor
import torch
import einx
from jaxtyping import Float, Bool, Int
import math
import cs336_basics.model.funtional as functional


def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)


def softmax(
    in_features: Float[Tensor, " ..."],
    dim: int
) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    shifted = in_features - in_features.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)

def attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    qk = einx.dot("... queries [d_k],  ... keys [d_k] -> ... queries keys", Q, K) / math.sqrt(Q.shape[-1])

    if mask is not None:
        qk = qk.masked_fill(~mask, -float('inf'))
    
    soft_max_qk = functional.softmax(qk, -1)
    return einx.dot("... queries [keys], ... [keys] d_v -> ... queries d_v", soft_max_qk, V)

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Stabilize the logits before taking exponentials so large positive values
    # do not overflow and very small probabilities do not underflow to zero.
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    nll = log_sum_exp - shifted.gather(1, targets.unsqueeze(1)).squeeze(1)
    return nll.mean()
