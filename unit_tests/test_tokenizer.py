from cs336_basics.tokenizer import Tokenizer

def test_encode_single_token():
    tokenizer = Tokenizer(
        vocab={
            0: b'',
            1: b'a',
            2: b'c',
            3: b'e',
            4: b'h',
            5: b't',
            6: b'th',
            7: b' c',
            8: b' a',
            9: b'the',
            10: b' at',

        },
        merges=[
            (b't', b'h'),
            (b' ', b'c'),
            (b' ', b'a'),
            (b'th', b'e'),
            (b' a', b't'),
        ],
        special_tokens=None,
    )

    res = tokenizer.encode_single_token("the")
    assert res == [9]


def test_encode_token():
    tokenizer = Tokenizer(
        vocab={
            0: b'',
            1: b'a',
            2: b'c',
            3: b'e',
            4: b'h',
            5: b't',
            6: b'th',
            7: b' c',
            8: b' a',
            9: b'the',
            10: b' at',

        },
        merges=[
            (b't', b'h'),
            (b' ', b'c'),
            (b' ', b'a'),
            (b'th', b'e'),
            (b' a', b't'),
        ],
        special_tokens=None,
    )

    res = tokenizer.encode("the cat ate")
    assert res == [9, 7, 1, 5, 10, 3]


def test_decode_combines_bytes_before_utf8_decoding():
    tokenizer = Tokenizer(
        vocab={
            0: b"\xf0\x9f",
            1: b"\x99",
            2: b"\x83",
        },
        merges=[],
        special_tokens=None,
    )

    assert tokenizer.decode([0, 1, 2]) == "🙃"

def test_encode_token_with_sequence():
    tokenizer = Tokenizer(
        vocab={
            0: b' ',
            1: b'a',
            2: b' a',
            3: b'r',
            4: b't',
            5: b'rt',
            6: b' ar',
            7: b' art',

        },
        merges=[
            (b' ', b'a'),
            (b'r', b't'),
            (b' a', b'rt'),
            (b' a', b'r'),
        ],
        special_tokens=None,
    )

    res = tokenizer.encode(" art")
    assert res == [7]


def test_encode_iterable_yields_token_ids_one_by_one():
    tokenizer = Tokenizer(
        vocab={
            0: b"",
            1: b"a",
            2: b"b",
            3: b"c",
        },
        merges=[],
        special_tokens=None,
    )

    res = list(tokenizer.encode_iterable(["ab", "c"]))
    assert res == [1, 2, 3]
