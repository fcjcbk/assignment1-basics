import os
import io
import contextlib

import regex
import heapq
from typing import TypeVar, Generic, List, Optional, Callable, Any

from cs336_basics.pretokenization_example import find_chunk_boundaries

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab: dict[int, bytes] = {}
    merge_list: list[tuple[bytes, bytes]] = []
    vocab_count = 0

    for special_token in special_tokens:
        vocab[vocab_count] = special_token.encode("utf-8")
        vocab_count += 1

    for i in range(256):
        vocab[vocab_count] = bytes([i])
        vocab_count += 1


    if len(vocab) >= vocab_size:
        return vocab, merge_list

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    frequency_map, adjacent = tokenize_with_special(text, special_tokens)
    
    visited_token: set[bytes] = set(vocab.values())
    pair_heap: list[tuple[int, bytes, bytes]] = []
    queued_pairs: set[tuple[bytes, bytes]] = set()

    for token in list(visited_token):
        _enqueue_available_pairs(
            pair_heap,
            queued_pairs,
            frequency_map,
            adjacent,
            visited_token,
            token,
        )

    removed_tokens: set[bytes] = set()
    while len(vocab) < vocab_size and pair_heap:
        token_pair = heapq.heappop(pair_heap)
        print(token_pair)
        _, left, right = token_pair

        if left not in visited_token or right not in visited_token:
            continue
        if left in removed_tokens or right in removed_tokens:
            continue

        merged_token = left + right
        if merged_token in visited_token:
            continue

        print(f"merging {left} and {right} to {merged_token}")

        vocab[vocab_count] = merged_token
        vocab_count += 1
        merge_list.append((left, right))
        visited_token.add(merged_token)

        removed_tokens.add(left)
        removed_tokens.add(right)



        _enqueue_available_pairs(
            pair_heap,
            queued_pairs,
            frequency_map,
            adjacent,
            visited_token,
            merged_token,
        )

    return vocab, merge_list


def _enqueue_available_pairs(
    pair_heap: list[tuple[int, bytes, bytes]],
    queued_pairs: set[tuple[bytes, bytes]],
    frequency_map: dict[Any, int],
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
    frequency_map: dict[Any, int],
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
    # Max frequency first; ties break by lexicographic order of the token pair.
    return (-count, left, right)


def tokenize_with_special(
    text,
    special_tokens: list[str],
) -> tuple[dict[bytes, int], dict[bytes, dict[str, set[bytes]]]]:
    """支持 special token 的分词函数"""
    # PAT = rf"{special_patterns}|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"
    PAT = r"""'(?:[sdmt]|ll|ve|re)|\s?\p{L}+|\s?\p{N}+|\s?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    if len(special_tokens) > 0:
        special_patterns = "|".join(special_tokens)
        PAT = special_patterns + "|" + PAT
    tokens = regex.findall(PAT, text, regex.VERBOSE)

    # print(tokens)

    return get_all_token_pair(tokens, special_tokens)

TokenPair = tuple[bytes, bytes]

def _init_adjacent_entry(adjacent: dict[bytes, dict[str, set[bytes]]], token: bytes) -> None:
    if token not in adjacent:
        adjacent[token] = {"left": set(), "right": set()}


def get_all_token_pair(
    tokens: list[str],
    special_tokens: set[str],
) -> tuple[dict[TokenPair, int], dict[bytes, dict[str, set[bytes]]]]:
    token_map: dict[str, int] = dict()
    res: dict[TokenPair, int] = dict()
    adjacent: dict[bytes, dict[str, set[bytes]]] = dict()

    # 统计token频率
    for token in tokens:
        if len(token) == 0:
            continue
        token_map[token] = token_map.get(token, 0) + 1

    # 处理每个token
    for token, frequency in token_map.items():
        token_bytes = token.encode(encoding="utf-8")
        # 添加整个token        
        if token in special_tokens:
            res[token_bytes] = res.get(token_bytes, 0) + frequency
            continue
        
        # 生成所有可能的子串
        token_len = len(token_bytes)
        for window_size in range(1, token_len + 1):  # 从2到完整长度
            i = 0
            while i + window_size <= token_len:

                j = i + 1
                # 生成所有可能的左右相邻子串对
                while j < i + window_size:
                    # 左子串
                    left = token_bytes[i:j]
                    # 右子串
                    right = token_bytes[j:i+window_size]
                    res[(left, right)] = res.get((left, right), 0) + frequency


                    # print("token:{}, i:{}, j:{}, left:{}, right:{}".format(token, i, j, left, right))
                    
                    # 添加相邻关系
                    _init_adjacent_entry(adjacent, left)
                    _init_adjacent_entry(adjacent, right)
                    adjacent[left]["right"].add(right)
                    adjacent[right]["left"].add(left)
                    
                    j += 1
                i += 1

    print(token_map)

    print("res:", res)
    print("adjacent:", adjacent)

    return [res, adjacent]
