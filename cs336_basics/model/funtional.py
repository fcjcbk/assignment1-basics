from torch import Tensor
import torch
import einx
from jaxtyping import Float, Bool, Int
import math
from collections.abc import Iterable
import cs336_basics.model.funtional as functional
import numpy.typing as npt
import random
import numpy as np


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


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    
    lr = 0
    
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate
    
    return lr


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = []
    
    for param in parameters:
        if param.grad is None:
            continue
        grads.append(param.grad.view(-1))
        
    if grads is None:
        return
    all_grads = torch.cat(grads)
    
    l2 = torch.sqrt(torch.sum(all_grads ** 2))
    
    if l2 > max_l2_norm:
        scal = max_l2_norm / (l2 + eps)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.data.mul_(scal)

def sample_train_data(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    
    inputs = []
    labels = []

    for _ in range(batch_size):
        start_index = random.randint(0, len(dataset) - context_length - 1)
        inputs.append(dataset[start_index : start_index + context_length])
        labels.append(dataset[start_index + 1 : start_index + 1 + context_length])

    inputs_tensor = torch.tensor(np.stack(inputs), dtype=torch.long, device=device)
    labels_tensor = torch.tensor(np.stack(labels), dtype=torch.long, device=device)
    return inputs_tensor, labels_tensor
