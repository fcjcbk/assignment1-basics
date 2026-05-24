from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn

from cs336_basics.model.rope import RoPE


class RoPEInterleaved(nn.Module):
    """A cleaned-up variant matching the math structure of RoPE_1 in rope.py."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        j = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (theta ** (2 * j / d_k))
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        positions = token_positions.unsqueeze(-1).float()
        theta = positions * self.freqs
        doubled_theta = torch.repeat_interleave(theta, repeats=2, dim=-1).to(dtype=x.dtype)

        rotated = reverse_pairs_last_dim(x)
        rotated = rotated.clone()
        rotated[..., 0::2] = -rotated[..., 0::2]

        return x * doubled_theta.cos() + rotated * doubled_theta.sin()


def reverse_pairs_last_dim(x: torch.Tensor) -> torch.Tensor:
    *batch_shape, last_dim = x.shape
    if last_dim % 2 != 0:
        raise ValueError(f"Expected an even last dimension, got {last_dim}")
    return x.view(*batch_shape, last_dim // 2, 2).flip(-1).reshape(*batch_shape, last_dim)


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    num_heads: int
    seq_len: int
    d_k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark two RoPE implementations.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--theta", type=float, default=10_000.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--num-heads", type=int, nargs="+", default=[16])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--d-k-values", type=int, nargs="+", default=[64, 128])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.mps.is_available() else "cpu")
    if name == "mps" and not torch.mps.is_available():
        raise RuntimeError("mps was requested but is not available.")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise RuntimeError("float16 benchmarking is only supported on CUDA in this script.")
    return dtype


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_runtime_ms(
    module: nn.Module,
    x: torch.Tensor,
    token_positions: torch.Tensor,
    warmup: int,
    num_runs: int,
    device: torch.device,
) -> tuple[float, float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            module(x, token_positions)
        synchronize(device)

        samples_ms: list[float] = []
        for _ in range(num_runs):
            start = time.perf_counter()
            module(x, token_positions)
            synchronize(device)
            samples_ms.append((time.perf_counter() - start) * 1_000)

    mean_ms = statistics.mean(samples_ms)
    stdev_ms = statistics.pstdev(samples_ms)
    min_ms = min(samples_ms)
    return mean_ms, stdev_ms, min_ms


def make_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(batch_size=batch_size, num_heads=num_heads, seq_len=seq_len, d_k=d_k)
        for batch_size in args.batch_sizes
        for num_heads in args.num_heads
        for seq_len in args.seq_lens
        for d_k in args.d_k_values
    ]


def verify_outputs(
    baseline: nn.Module,
    candidate: nn.Module,
    x: torch.Tensor,
    token_positions: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    with torch.inference_mode():
        baseline_out = baseline(x, token_positions)
        candidate_out = candidate(x, token_positions)

    atol = 1e-3 if dtype in {torch.float16, torch.bfloat16} else 1e-5
    rtol = 1e-3 if dtype in {torch.float16, torch.bfloat16} else 1e-5
    torch.testing.assert_close(candidate_out, baseline_out, atol=atol, rtol=rtol)


def format_case(case: BenchmarkCase) -> str:
    return f"bs={case.batch_size}, heads={case.num_heads}, seq={case.seq_len}, d_k={case.d_k}"


def print_header(device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> None:
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"theta: {args.theta}")
    print(f"warmup: {args.warmup}")
    print(f"num_runs: {args.num_runs}")
    print()
    print(
        f"{'case':<36} {'RoPE mean(ms)':>14} {'RoPE_1-style mean(ms)':>22} {'speedup':>10} {'winner':>10}"
    )
    print("-" * 96)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    cases = make_cases(args)

    print_header(device, dtype, args)

    for case in cases:
        x = torch.randn(
            case.batch_size,
            case.num_heads,
            case.seq_len,
            case.d_k,
            device=device,
            dtype=dtype,
        )
        token_positions = torch.arange(case.seq_len, device=device)

        baseline = RoPE(
            theta=args.theta,
            d_k=case.d_k,
            max_seq_len=case.seq_len,
            device=device,
        ).to(device=device)
        candidate = RoPEInterleaved(
            theta=args.theta,
            d_k=case.d_k,
            max_seq_len=case.seq_len,
            device=device,
        ).to(device=device)

        verify_outputs(baseline, candidate, x, token_positions, dtype)
        baseline_mean, _, _ = measure_runtime_ms(
            baseline, x, token_positions, args.warmup, args.num_runs, device
        )
        candidate_mean, _, _ = measure_runtime_ms(
            candidate, x, token_positions, args.warmup, args.num_runs, device
        )

        if baseline_mean <= candidate_mean:
            speedup = candidate_mean / baseline_mean
            winner = "RoPE"
        else:
            speedup = baseline_mean / candidate_mean
            winner = "RoPE_1"

        print(
            f"{format_case(case):<36} {baseline_mean:>14.3f} {candidate_mean:>22.3f} "
            f"{speedup:>10.2f}x {winner:>10}"
        )


if __name__ == "__main__":
    main()
