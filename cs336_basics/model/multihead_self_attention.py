import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn
import einx
from cs336_basics.model.linear import Linear
import cs336_basics.model.rope as rope
import cs336_basics.model.funtional as functional

# def run_multihead_self_attention(
#     d_model: int,
#     num_heads: int,
#     q_proj_weight: Float[Tensor, " d_model d_model"],
#     k_proj_weight: Float[Tensor, " d_model d_model"],
#     v_proj_weight: Float[Tensor, " d_model d_model"],
#     o_proj_weight: Float[Tensor, " d_model d_model"],
#     in_features: Float[Tensor, " ... sequence_length d_model"],
# ) -> Float[Tensor, " ... sequence_length d_model"]:
#     """
#     Given the key, query, and value projection weights of a naive unbatched
#     implementation of multi-head attention, return the output of an optimized batched
#     implementation. This implementation should handle the key, query, and value projections
#     for all heads in a single matrix multiply.
#     This function should not use RoPE.
#     See section 3.2.2 of Vaswani et al., 2017.

#     Args:
#         d_model (int): Dimensionality of the feedforward input and output.
#         num_heads (int): Number of heads to use in multi-headed attention.
#         max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
#         q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
#         k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
#         v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
#         o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
#         in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

#     Returns:
#         Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
#         implementation with the given QKV projection weights and input features.
#     """



#     raise NotImplementedError


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,    # Dimensionality of the Transformer block inputs
        num_heads: int,  # Number of heads to use in multi-head self-attention
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
    
        self.WQ = Linear(d_model, d_model, device, dtype)
        self.WK = Linear(d_model, d_model, device, dtype)
        self.WV = Linear(d_model, d_model, device, dtype)

        self.WO = Linear(d_model, d_model, device, dtype)

        self.rope = rope.RoPE(1000.0, self.d_k, max_seq_len, device)

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        prefix_shape = in_features.shape[:-2]  # 例如 (batch, heads) 或 (batch,)
        seq_len = in_features.shape[-2]

        # 1. get q k v
        Q = self.WQ(in_features)
        K = self.WK(in_features)
        V = self.WV(in_features)

        # 2. splict head
        q_heads = einx.id(
            '... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k',
            Q,
            num_heads=self.num_heads,
            d_k = self.d_k,
            backend='torch'  # 关键：指定后端
        )

        k_heads = einx.id(
            '... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k',
            K,
            num_heads=self.num_heads,
            d_k = self.d_k,
            backend='torch'  # 关键：指定后端
        )

        v_heads = einx.id(
            '... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k',
            V,
            num_heads=self.num_heads,
            d_k = self.d_k,
            backend='torch'  # 关键：指定后端
        )

        causal_mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        target_shape = prefix_shape + (seq_len, seq_len)
        causal_mask = causal_mask.broadcast_to(target_shape)

        attention_val = functional.attention(q_heads, k_heads, v_heads, causal_mask)
        
        out = einx.id(
            "... num_heads sequence_length d_k -> ... sequence_length (num_heads d_k)",
            attention_val,
            heads=self.num_heads,
            d_k = self.d_k,
            backend='torch',  # 关键：指定后端
        )

        return self.WO(out)