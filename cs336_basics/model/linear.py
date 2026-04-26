from jaxtyping import Bool, Float, Int
from torch import nn
from torch import Tensor
import einx

class Linear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        weights: Float[Tensor, " d_out d_in"],
    ):
        super().__init__()
        self.weights = nn.Parameter(weights)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einx.dot("... [d_in], d_out [d_in] -> ... d_out", x, self.weights)
