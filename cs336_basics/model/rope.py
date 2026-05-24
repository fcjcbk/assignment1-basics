from torch import nn
import torch


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        j = torch.arange(0, d_k, 2, device=device).float()
        freqs = 1.0 / (theta ** (j / d_k))
        self.register_buffer("freqs", freqs)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        positions = token_positions.unsqueeze(-1).float()
        angles = positions * self.freqs
        cos = angles.cos()
        sin = angles.sin()

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_odd * cos + x_even * sin
        return out
