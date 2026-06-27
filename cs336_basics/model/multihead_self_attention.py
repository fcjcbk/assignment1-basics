import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn
import einx
from cs336_basics.model.linear import Linear
import cs336_basics.model.rope as rope
import cs336_basics.model.funtional as functional



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

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
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

        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device).tril()
        causal_mask = causal_mask.view(*([1] * (q_heads.ndim - 2)), seq_len, seq_len)

        attention_val = functional.attention(q_heads, k_heads, v_heads, causal_mask)
        
        out = einx.id(
            "... num_heads sequence_length d_k -> ... sequence_length (num_heads d_k)",
            attention_val,
            heads=self.num_heads,
            d_k = self.d_k,
            backend='torch',  # 关键：指定后端
        )

        return self.WO(out)
    


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,    # Dimensionality of the Transformer block inputs
        num_heads: int,  # Number of heads to use in multi-head self-attention
        max_seq_len: int,
        theta: float,
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

        self.rope = rope.RoPE(theta, self.d_k, max_seq_len, device)

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None, # Optional tensor with the positions of the tokens
    ) -> Float[Tensor, " ... sequence_length d_model"]:
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

        q_heads = self.rope(q_heads, token_positions)
        k_heads = self.rope(k_heads, token_positions)

        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device).tril()
        causal_mask = causal_mask.view(*([1] * (q_heads.ndim - 2)), seq_len, seq_len)

        attention_val = functional.attention(q_heads, k_heads, v_heads, causal_mask)
        
        out = einx.id(
            "... num_heads sequence_length d_k -> ... sequence_length (num_heads d_k)",
            attention_val,
            heads=self.num_heads,
            d_k = self.d_k,
            backend='torch',  # 关键：指定后端
        )

        return self.WO(out)
