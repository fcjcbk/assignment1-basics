from torch import nn
from torch import Tensor
import torch
import einx
import math

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        sigma_square = 2 / (in_features + out_features)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=math.sqrt(sigma_square), a=-2.0, b=2.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("... [d_in], d_out [d_in] -> ... d_out", x, self.weight)
