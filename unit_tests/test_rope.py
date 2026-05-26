import torch

from cs336_basics.model.rope import RoPE, RoPE_1


def test_rope_1_matches_rope():
    device = torch.device("cpu")
    x = torch.randn(2, 3, 5, 8, device=device)
    token_positions = torch.arange(5, device=device)

    rope = RoPE(theta=10_000.0, d_k=8, max_seq_len=5, device=device)
    rope_1 = RoPE_1(theta=10_000.0, d_k=8, max_seq_len=5, device=device)

    actual = rope_1(x, token_positions)
    expected = rope(x, token_positions)

    torch.testing.assert_close(actual, expected)


def test_rope_uses_default_positions_when_none():
    device = torch.device("cpu")
    x = torch.randn(2, 3, 5, 8, device=device)
    token_positions = torch.arange(5, device=device).view(1, 1, 5).expand(2, 3, 5)

    rope = RoPE(theta=10_000.0, d_k=8, max_seq_len=5, device=device)
    rope_1 = RoPE_1(theta=10_000.0, d_k=8, max_seq_len=5, device=device)

    torch.testing.assert_close(rope(x, None), rope(x, token_positions))
    torch.testing.assert_close(rope_1(x, None), rope_1(x, token_positions))
