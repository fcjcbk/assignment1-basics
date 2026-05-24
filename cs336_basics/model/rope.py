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


class RoPE_1(nn.Module):
    def __init__(self,
        theta: float,
        d_k: int, # dimension of the key (and query) vectors
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"Expected an even d_k for RoPE, got {d_k}")

        self.max_seq_len = max_seq_len

        j = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (theta ** (2 * j / d_k))
        self.register_buffer("freqs", freqs)

    def forward(
        self,
        x: torch.Tensor,                # shape of (..., seq_len, d_k)
        token_positions: torch.Tensor   # a tensor of shape (..., seq_len)
    ) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"Expected an even last dimension for RoPE, got {x.shape[-1]}")

        positions = token_positions.unsqueeze(-1).to(dtype=self.freqs.dtype)
        angles = positions * self.freqs
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)

        x_pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        out = torch.empty_like(x_pairs)
        out[..., 0] = x_even * cos - x_odd * sin
        out[..., 1] = x_odd * cos + x_even * sin
        return out.flatten(start_dim=-2)


def reverse_pairs_last_dim(x):
    *batch, L = x.shape
    assert L % 2 == 0, "最后一维长度必须是偶数"
    return x.view(*batch, L // 2, 2).flip(-1).reshape(*batch, L)
