# RoPE Benchmark

Run:

```sh
uv run benchmark/benchmark_rope.py
```

Example with a smaller smoke-test workload:

```sh
uv run benchmark/benchmark_rope.py --device cpu --warmup 2 --num-runs 5 --seq-lens 32 --d-k-values 64
```

Notes:

- `RoPE` is imported directly from `cs336_basics/model/rope.py`.
- `RoPE_1-style` is a no-debug benchmark variant that keeps the same rotate-and-interleave math as `RoPE_1`, so results reflect compute cost instead of `print(...)` overhead.
- The script checks numerical equivalence before timing each case.
