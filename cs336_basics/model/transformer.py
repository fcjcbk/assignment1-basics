import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn
import einx
from cs336_basics.model.rms_norm import RMSNorm
from cs336_basics.model.multihead_self_attention import MultiHeadSelfAttentionWithRoPE 
from cs336_basics.model.swi_glu import SwiGLu


class Transformer(nn.Module):
    def __init__(
        self,
        d_model:  int,                  # Dimensionality of the Transformer block inputs
        num_heads: int,                 # Number of heads to use in multi-head self-attention.
        max_seq_len: int,
        theta: float,
        d_ff: int | None = None,        # Dimensionality of the position-wise feed-forward inner layer.
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.rms_1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        self.multi_head_self_attenttion = MultiHeadSelfAttentionWithRoPE(
            d_model,
            num_heads,
            max_seq_len,
            theta,
            device,
            dtype
        )

        self.rms_2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        self.ffn = SwiGLu(
            d_model,
            d_ff,
            device,
            dtype,
        )

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:

        layer_1 = in_features + self.multi_head_self_attenttion(self.rms_1(in_features))
        return layer_1 + self.ffn(self.rms_2(layer_1))



