from cs336_basics.bpe import _enqueue_available_pairs, train_bpe


def test_enqueue_available_pairs_respects_neighbor_side():
    pair_heap = []
    queued_pairs = set()
    frequency_map = {
        (b"a", b"b"): 5,
        (b"b", b"a"): 99,
        (b"b", b"c"): 4,
        (b"c", b"b"): 88,
    }
    adjacent = {
        b"b": {"left": {b"a"}, "right": {b"c"}},
    }
    visited_token = {b"a", b"b", b"c"}

    _enqueue_available_pairs(
        pair_heap,
        queued_pairs,
        frequency_map,
        adjacent,
        visited_token,
        b"b",
    )

    assert set(pair_heap) == {(-5, b"a", b"b"), (-4, b"b", b"c")}


def test_train_bpe_breaks_frequency_ties_by_lexicographic_pair_order(tmp_path, monkeypatch):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("dummy corpus", encoding="utf-8")

    frequency_map = {
        (b"a", b"c"): 10,
        (b"b", b"a"): 10,
        (b"c", b"b"): 9,
    }
    adjacent = {
        b"a": {"left": set(), "right": {b"c"}},
        b"b": {"left": set(), "right": {b"a"}},
        b"c": {"left": {b"a"}, "right": {b"b"}},
    }

    monkeypatch.setattr(
        "cs336_basics.bpe.tokenize_with_special",
        lambda text, special_tokens: (frequency_map, adjacent),
    )

    _, merges = train_bpe(
        input_path=input_path,
        vocab_size=259,
        special_tokens=["<|endoftext|>"],
    )

    assert merges[0] == (b"a", b"c")


def test_train_bpe_uses_heap_and_skips_seen_merged_tokens(tmp_path, monkeypatch):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("dummy corpus", encoding="utf-8")

    frequency_map = {
        (b"b", b"c"): 11,
        (b"a", b"b"): 10,
        (b"ab", b"c"): 9,
        (b"a", b"bc"): 8,
    }
    adjacent = {
        b"a": {"left": set(), "right": {b"b", b"bc"}},
        b"b": {"left": {b"a"}, "right": {b"c"}},
        b"ab": {"left": set(), "right": {b"c"}},
        b"bc": {"left": {b"a"}, "right": set()},
        b"c": {"left": {b"ab", b"b"}, "right": set()},
        b"x": {"left": set(), "right": {b"y"}},
        b"y": {"left": {b"x"}, "right": set()},
    }

    monkeypatch.setattr(
        "cs336_basics.bpe.tokenize_with_special",
        lambda text, special_tokens: (frequency_map, adjacent),
    )

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=260,
        special_tokens=["<|endoftext|>"],
    )

    assert merges == [(b"b", b"c"), (b"a", b"b"), (b"ab", b"c")]
    assert b"ab" in vocab.values()
    assert b"bc" in vocab.values()
    assert b"abc" in vocab.values()

def test_train_bpe_heap():
    vocab, merges = train_bpe(
        input_path="tests/fixtures/corpus.en",
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    print("vocab:", vocab)
    print("merges:", merges)
