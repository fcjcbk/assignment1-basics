from torch import nn
from torch import Tensor
import torch
import einx

class RMSNorm(nn.Module):
    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        x = x / rms
        result = x * self.weight

        return result.to(in_dtype)
