from cs336_basics.tokenizer.bpe import (
    _enqueue_available_pairs,
    _stream_tokenize_with_special,
    tokenize_with_special,
    train_bpe,
)


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


def test_train_bpe_breaks_frequency_ties_by_lexicographic_pair_order(tmp_path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("ba\nac", encoding="utf-8")

    _, merges = train_bpe(
        input_path=input_path,
        vocab_size=257,
        special_tokens=[],
    )

    assert merges[0] == (b"b", b"a")


def test_train_bpe_can_merge_with_previously_created_tokens(tmp_path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("bc\nabc", encoding="utf-8")

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=258,
        special_tokens=[],
    )

    assert merges == [(b"b", b"c"), (b"a", b"bc")]
    assert b"bc" in vocab.values()
    assert b"abc" in vocab.values()

def test_train_bpe_heap():
    vocab, merges = train_bpe(
        input_path="tests/fixtures/corpus.en",
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    assert len(vocab) == 300
    assert len(merges) == 43


def test_train_bpe_retokenizes_words_after_merge(tmp_path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("aaa", encoding="utf-8")

    _, merges = train_bpe(
        input_path=input_path,
        vocab_size=258,
        special_tokens=[],
    )

    assert merges == [(b"a", b"a"), (b"aa", b"a")]


def test_stream_tokenize_with_special_matches_full_text_with_tiny_chunks(tmp_path):
    text = (
        "Once upon a tiny time<|endoftext|>Another-story starts now!\n"
        "lettersCrossChunk 1234567890 ???   spaces\n\n"
        "prefix <|endoftext|> suffix and emoji 🙃 stay intact"
    )
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(text, encoding="utf-8")

    expected = tokenize_with_special(text, ["<|endoftext|>"])
    actual = _stream_tokenize_with_special(
        input_path,
        ["<|endoftext|>"],
        chunk_size=5,
    )

    assert actual == expected


def test_stream_tokenize_without_special_matches_full_text_with_tiny_chunks(tmp_path):
    text = "abc def\n\n12345 !!!   trailing spaces   and emoji 🙃"
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(text, encoding="utf-8")

    expected = tokenize_with_special(text, [])
    actual = _stream_tokenize_with_special(input_path, [], chunk_size=4)

    assert actual == expected


def test_train_bpe_is_quiet_by_default(tmp_path, capsys):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("ba\nac", encoding="utf-8")

    train_bpe(input_path=input_path, vocab_size=257, special_tokens=[])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
