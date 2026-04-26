import heapq
import os
from dataclasses import dataclass

import regex

TokenPair = tuple[bytes, bytes]
PairHeapItem = tuple[int, "ReverseBytes", "ReverseBytes", bytes, bytes, int]

BASE_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


@dataclass(slots=True)
class WordState:
    frequency: int
    tokens: list[bytes]
    pair_counts: dict[TokenPair, int]


@dataclass(frozen=True, slots=True)
class ReverseBytes:
    value: bytes

    def __lt__(self, other: "ReverseBytes") -> bool:
        return self.value > other.value


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer and return the vocabulary and merge list."""
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    next_vocab_id = 0
    for special_token in special_tokens:
        vocab[next_vocab_id] = special_token.encode("utf-8")
        next_vocab_id += 1

    for token in BYTE_TOKENS:
        vocab[next_vocab_id] = token
        next_vocab_id += 1

    if next_vocab_id >= vocab_size:
        return vocab, merges

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    word_frequencies = tokenize_with_special(text, special_tokens)
    words, pair_counts, pair_to_words = _build_word_states(word_frequencies, set(special_tokens))

    pair_versions: dict[TokenPair, int] = {}
    pair_heap: list[PairHeapItem] = []
    for pair, count in pair_counts.items():
        if count <= 0:
            continue
        pair_versions[pair] = 1
        heapq.heappush(pair_heap, _make_versioned_heap_item(count, pair, 1))

    vocab_tokens = set(vocab.values())
    while next_vocab_id < vocab_size:
        best_pair = _pop_best_pair(pair_heap, pair_counts, pair_versions)
        if best_pair is None:
            break

        left, right = best_pair
        merged_token = left + right
        if merged_token in vocab_tokens:
            continue

        vocab[next_vocab_id] = merged_token
        vocab_tokens.add(merged_token)
        next_vocab_id += 1
        merges.append(best_pair)

        affected_word_ids = list(pair_to_words.get(best_pair, ()))
        changed_pairs: set[TokenPair] = set()
        for word_id in affected_word_ids:
            word = words[word_id]
            new_tokens = _merge_pair_in_word(word.tokens, best_pair, merged_token)
            if new_tokens == word.tokens:
                continue

            old_pair_counts = word.pair_counts
            new_pair_counts = _count_adjacent_pairs(new_tokens)
            _update_pair_statistics(
                word_id=word_id,
                word_frequency=word.frequency,
                old_pair_counts=old_pair_counts,
                new_pair_counts=new_pair_counts,
                pair_counts=pair_counts,
                pair_to_words=pair_to_words,
                changed_pairs=changed_pairs,
            )
            word.tokens = new_tokens
            word.pair_counts = new_pair_counts

        for pair in changed_pairs:
            _bump_pair_version(pair, pair_counts, pair_versions, pair_heap)

    return vocab, merges


def tokenize_with_special(text: str, special_tokens: list[str]) -> dict[str, int]:
    word_frequencies: dict[str, int] = {}
    if not special_tokens:
        _count_base_tokens(text, word_frequencies)
        return word_frequencies

    special_pattern = build_special_token_pattern(special_tokens)
    last_end = 0
    for match in special_pattern.finditer(text):
        _count_base_tokens(text[last_end : match.start()], word_frequencies)
        token = match.group(0)
        word_frequencies[token] = word_frequencies.get(token, 0) + 1
        last_end = match.end()

    _count_base_tokens(text[last_end:], word_frequencies)
    return word_frequencies


def _count_base_tokens(text: str, word_frequencies: dict[str, int]) -> None:
    for match in regex.finditer(BASE_PATTERN, text):
        token = match.group(0)
        if token:
            word_frequencies[token] = word_frequencies.get(token, 0) + 1


def build_special_token_pattern(special_tokens: list[str]) -> regex.Pattern[str]:
    escaped_special_tokens = sorted(
        (regex.escape(token) for token in special_tokens),
        key=len,
        reverse=True,
    )
    return regex.compile("|".join(escaped_special_tokens))


def _build_word_states(
    word_frequencies: dict[str, int],
    special_token_set: set[str],
) -> tuple[list[WordState], dict[TokenPair, int], dict[TokenPair, set[int]]]:
    words: list[WordState] = []
    pair_counts: dict[TokenPair, int] = {}
    pair_to_words: dict[TokenPair, set[int]] = {}

    for word, frequency in word_frequencies.items():
        if word in special_token_set:
            tokens = [word.encode("utf-8")]
        else:
            word_bytes = word.encode("utf-8")
            tokens = [BYTE_TOKENS[byte] for byte in word_bytes]

        local_pair_counts = _count_adjacent_pairs(tokens)
        word_id = len(words)
        words.append(
            WordState(
                frequency=frequency,
                tokens=tokens,
                pair_counts=local_pair_counts,
            )
        )

        if not local_pair_counts:
            continue

        for pair, occurrences in local_pair_counts.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * frequency
            pair_to_words.setdefault(pair, set()).add(word_id)

    return words, pair_counts, pair_to_words


def _count_adjacent_pairs(tokens: list[bytes]) -> dict[TokenPair, int]:
    pair_counts: dict[TokenPair, int] = {}
    for left, right in zip(tokens, tokens[1:]):
        pair = (left, right)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def _merge_pair_in_word(
    tokens: list[bytes],
    pair: TokenPair,
    merged_token: bytes,
) -> list[bytes]:
    merged_tokens: list[bytes] = []
    i = 0
    last_index = len(tokens) - 1
    while i < len(tokens):
        if i < last_index and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged_tokens.append(merged_token)
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens


def _update_pair_statistics(
    *,
    word_id: int,
    word_frequency: int,
    old_pair_counts: dict[TokenPair, int],
    new_pair_counts: dict[TokenPair, int],
    pair_counts: dict[TokenPair, int],
    pair_to_words: dict[TokenPair, set[int]],
    changed_pairs: set[TokenPair],
) -> None:
    for pair in old_pair_counts.keys() | new_pair_counts.keys():
        old_count = old_pair_counts.get(pair, 0)
        new_count = new_pair_counts.get(pair, 0)
        if old_count == new_count:
            continue

        if old_count == 0:
            pair_to_words.setdefault(pair, set()).add(word_id)
        elif new_count == 0:
            linked_words = pair_to_words.get(pair)
            if linked_words is not None:
                linked_words.discard(word_id)
                if not linked_words:
                    pair_to_words.pop(pair, None)

        updated_count = pair_counts.get(pair, 0) + (new_count - old_count) * word_frequency
        if updated_count > 0:
            pair_counts[pair] = updated_count
        else:
            pair_counts.pop(pair, None)

        changed_pairs.add(pair)


def _bump_pair_version(
    pair: TokenPair,
    pair_counts: dict[TokenPair, int],
    pair_versions: dict[TokenPair, int],
    pair_heap: list[PairHeapItem],
) -> None:
    version = pair_versions.get(pair, 0) + 1
    pair_versions[pair] = version
    count = pair_counts.get(pair, 0)
    if count > 0:
        heapq.heappush(pair_heap, _make_versioned_heap_item(count, pair, version))


def _make_versioned_heap_item(
    count: int,
    pair: TokenPair,
    version: int,
) -> PairHeapItem:
    return (-count, ReverseBytes(pair[0]), ReverseBytes(pair[1]), pair[0], pair[1], version)


def _pop_best_pair(
    pair_heap: list[PairHeapItem],
    pair_counts: dict[TokenPair, int],
    pair_versions: dict[TokenPair, int],
) -> TokenPair | None:
    while pair_heap:
        neg_count, _, _, left, right, version = heapq.heappop(pair_heap)
        pair = (left, right)
        current_version = pair_versions.get(pair)
        current_count = pair_counts.get(pair, 0)
        if current_version != version:
            continue
        if current_count <= 0:
            continue
        if -neg_count != current_count:
            continue
        return pair
    return None


def _enqueue_available_pairs(
    pair_heap: list[tuple[int, bytes, bytes]],
    queued_pairs: set[tuple[bytes, bytes]],
    frequency_map: dict[object, int],
    adjacent: dict[bytes, dict[str, set[bytes]]],
    visited_token: set[bytes],
    token: bytes,
) -> None:
    token_neighbors = adjacent.get(token, {})

    for neighbor in token_neighbors.get("left", set()):
        if neighbor not in visited_token:
            continue
        _push_pair_candidate(pair_heap, queued_pairs, frequency_map, neighbor, token)

    for neighbor in token_neighbors.get("right", set()):
        if neighbor not in visited_token:
            continue
        _push_pair_candidate(pair_heap, queued_pairs, frequency_map, token, neighbor)


def _push_pair_candidate(
    pair_heap: list[tuple[int, bytes, bytes]],
    queued_pairs: set[tuple[bytes, bytes]],
    frequency_map: dict[object, int],
    left: bytes,
    right: bytes,
) -> None:
    pair = (left, right)
    if pair in queued_pairs:
        return

    count = frequency_map.get(pair)
    if count is None or count <= 0:
        return

    heapq.heappush(pair_heap, _make_pair_heap_item(count, left, right))
    queued_pairs.add(pair)


def _make_pair_heap_item(count: int, left: bytes, right: bytes) -> tuple[int, bytes, bytes]:
    return (-count, left, right)
