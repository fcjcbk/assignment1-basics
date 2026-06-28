
import torch
from jaxtyping import Float, Int
from torch import nn
from einops import pack, rearrange
from jaxtyping import jaxtyped
from beartype import beartype

from cs336_basics.model.funtional import softmax

@jaxtyped(typechecker=beartype)
def decode(
    input_ids: Float[torch.Tensor, "batch seq_len"],
    model: nn.Module,
) -> Float[torch.Tensor, "batch seq_len vocab_size"]:
    """
    Transformer 前向计算。

    input:
        input_ids: (batch, seq_len)

    output:
        logits: (batch, seq_len, vocab_size)
    """
    return model(input_ids)

@jaxtyped(typechecker=beartype)
def sample(
    logits: Float[torch.Tensor, "batch vocab_size"],
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Int[torch.Tensor, "batch"]:
    """
    根据最后一个位置的 logits 选择下一个 token。

    input:
        logits: (batch, vocab_size)

    output:
        next_token: (batch,)
    """
    caled_logits = logits / max(temperature, 1e-7)
    return __top_p_sampling(caled_logits, top_p)
    

@jaxtyped(typechecker=beartype)
def generate(
    model: nn.Module,
    prompt_ids: Int[torch.Tensor, "batch prompt_len"],
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Int[torch.Tensor, "batch output_len"]:
    """
    循环调用 decode 和 sample, 生成完整序列。

    input:
        prompt_ids: (batch, prompt_len)

    output:
        generated_ids: (batch, prompt_len + max_new_tokens)
    """

    max_tokens = prompt_ids.shape[1] + max_new_tokens
    batch_size = prompt_ids.shape[0]
    
    # (batch)
    finished = torch.zeros(batch_size, dtype=torch.bool)
    
    
    while prompt_ids.shape[1] < max_tokens:
        # shape (batch seq_len vocab_size)
        logits = decode(prompt_ids, model)
        
        # (batch vocab_size) -> (batch)
        next_tokens = sample(logits[:,-1], temperature, top_p)
        next_tokens = torch.where(
            finished,
            eos_token_id,
            next_tokens,
        )
        
        finished |= (next_tokens == eos_token_id)

        next_tokens = rearrange(
            next_tokens.to(dtype=prompt_ids.dtype, device=prompt_ids.device),
            "batch -> batch 1",
        )

        prompt_ids = torch.cat([prompt_ids, next_tokens], dim=1)
        
        if finished.all():
            break
    return prompt_ids
    

@jaxtyped(typechecker=beartype)
def __top_p_sampling(
    logits: Float[torch.Tensor, "batch vocab_size"],
    top_p: float = 1.0,
) -> Int[torch.Tensor, "batch "]:
    # 排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # 移除超出阈值的部分
    sorted_mask = cum_probs > top_p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))

    # 在排序后的分布上采样 (得到的是排序后的索引位置)
    probs = softmax(sorted_logits, dim=-1)
    sampled_index_in_sorted = torch.multinomial(probs, 1)
    
    # 通过 gather 映射回真实的 token id
    next_token = rearrange(torch.gather(sorted_indices, -1, sampled_index_in_sorted), "batch 1 -> batch")
    return next_token
