import torch
from torch import nn


def import_decode_module():
    import cs336_basics.decode as decode_module

    return decode_module


class RecordingLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_input_ids = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.seen_input_ids = input_ids
        batch, seq_len = input_ids.shape
        token_offsets = torch.arange(self.vocab_size, dtype=torch.float32)
        return input_ids.float().unsqueeze(-1) + token_offsets.reshape(1, 1, self.vocab_size)


def test_decode_forwards_input_ids_to_model_and_returns_logits():
    decode_module = import_decode_module()

    input_ids = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model = RecordingLanguageModel(vocab_size=4)

    logits = decode_module.decode(input_ids, model)

    assert model.seen_input_ids is input_ids
    assert logits.shape == (2, 3, 4)
    expected = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
            [[4.0, 5.0, 6.0, 7.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]],
        ]
    )
    torch.testing.assert_close(logits, expected)


def test_sample_applies_temperature_before_top_p_sampling(monkeypatch):
    decode_module = import_decode_module()
    logits = torch.tensor([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]])
    seen = {}

    def fake_top_p_sampling(scaled_logits: torch.Tensor, top_p: float) -> torch.Tensor:
        seen["scaled_logits"] = scaled_logits
        seen["top_p"] = top_p
        return torch.tensor([2, 1])

    monkeypatch.setattr(decode_module, "__top_p_sampling", fake_top_p_sampling)

    next_tokens = decode_module.sample(logits, temperature=2.0, top_p=0.75)

    torch.testing.assert_close(next_tokens, torch.tensor([2, 1]))
    torch.testing.assert_close(seen["scaled_logits"], logits / 2.0)
    assert seen["top_p"] == 0.75


def test_top_p_sampling_keeps_at_least_highest_probability_token():
    decode_module = import_decode_module()
    logits = torch.tensor([[10.0, 0.0, -1.0], [-2.0, 8.0, 1.0]])

    next_tokens = decode_module.__top_p_sampling(logits, top_p=0.01)

    torch.testing.assert_close(next_tokens, torch.tensor([0, 1]))
    assert next_tokens.shape == (2,)


def test_generate_appends_sampled_tokens_and_stops_after_eos(monkeypatch):
    decode_module = import_decode_module()
    prompt_ids = torch.tensor([[7, 8]])
    contexts_seen = []
    sampled_tokens = [torch.tensor([4.0]), torch.tensor([9.0])]

    def fake_decode(context: torch.Tensor, model: nn.Module) -> torch.Tensor:
        contexts_seen.append(context.clone())
        batch_size, seq_len = context.shape
        return torch.zeros(batch_size, seq_len, 10)

    def fake_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        assert logits.shape == (1, 10)
        assert temperature == 0.5
        assert top_p == 0.8
        return sampled_tokens.pop(0).to(int)

    monkeypatch.setattr(decode_module, "decode", fake_decode)
    monkeypatch.setattr(decode_module, "sample", fake_sample)

    generated = decode_module.generate(
        model=nn.Identity(),
        prompt_ids=prompt_ids,
        max_new_tokens=5,
        eos_token_id=9,
        temperature=0.5,
        top_p=0.8,
    )

    torch.testing.assert_close(generated, torch.tensor([[7, 8, 4, 9]]))
    assert len(contexts_seen) == 2
    torch.testing.assert_close(contexts_seen[0], torch.tensor([[7, 8]]))
    torch.testing.assert_close(contexts_seen[1], torch.tensor([[7, 8, 4]]))
