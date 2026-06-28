import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn
import einx
from jaxtyping import jaxtyped
from beartype import beartype

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.transformer import Transformer 
from cs336_basics.model.rms_norm import RMSNorm
from cs336_basics.model.linear import Linear
from cs336_basics.model.funtional import softmax

class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model:  int,                  # Dimensionality of the Transformer block inputs
        num_heads: int,                 # Number of heads to use in multi-head self-attention.
        max_seq_len: int,
        theta: float,
        d_ff: int | None = None,        # Dimensionality of the position-wise feed-forward inner layer.
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.embedding = Embedding(
            vocab_size,
            d_model,
            device,
            dtype,
        )

        self.tranformer_blocks = nn.ModuleList([Transformer(
            d_model,
            num_heads,
            max_seq_len,
            theta,
            d_ff,
            device,
            dtype,
        ) for _ in range(num_layers)])

        self.norm = RMSNorm(
            d_model,
            device=device,
            dtype=dtype
        )

        self.linear = Linear(
            d_model,
            vocab_size,
            device,
            dtype,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        in_indices: Int[Tensor, " batch sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        
        in_indices = self.embedding(in_indices)

        for block in self.tranformer_blocks:
            in_indices = block(in_indices)
        
        out = self.linear(self.norm(in_indices))

        return out
