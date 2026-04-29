from torch import nn
from torch import Tensor
import torch
import einx
from cs336_basics.model.linear import Linear
import cs336_basics.model.funtional as functional

class SwiGLu(nn.Module):
    def __init__(self,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.d_ff = int(((d_model * 8 / 3) + 63) // 64 * 64) 
        self.w_1 = Linear(d_model, self.d_ff, device, dtype)
        self.w_2 = Linear(self.d_ff, d_model, device, dtype)
        self.w_3 = Linear(d_model, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu = functional.silu(self.w_1(x))

        return self.w_2(silu * self.w_3(x))
