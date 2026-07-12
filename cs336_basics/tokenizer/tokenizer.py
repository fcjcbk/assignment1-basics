import base64
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import regex
import cs336_basics.tokenizer.bpe as bpe


def _decode_base64_bytes(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.reverse_vocab = {v: k for k, v in vocab.items()}
        
        self.special_pattern = None
        if special_tokens is not None:
            self.special_pattern = bpe.build_special_token_pattern(special_tokens)
        
        merge_map: dict[bytes, set[bytes]] = {}
        for merge in merges:
            merge_map.setdefault(merge[0], set()).add(merge[1])
        self.merge_map = merge_map

        merge_priority: dict[tuple[bytes, bytes], int] = {}
        for i, merge in enumerate(merges):
            merge_priority[merge] = i
        self.merge_priority = merge_priority

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
        metadata_path: str | Path | None = None,
    ) -> "Tokenizer":
        """
        Create a tokenizer from files.
        """
        if metadata_path is not None:
            metadata = _read_json(Path(metadata_path))
            if metadata.get("encoding") != "base64":
                raise ValueError(f"Expected base64 tokenizer artifacts, got encoding={metadata.get('encoding')!r}")
            if special_tokens is None:
                special_tokens = metadata.get("special_tokens") or None

        vocab_json = _read_json(Path(vocab_filepath))
        merges_json = _read_json(Path(merges_filepath))

        vocab = {int(token_id): _decode_base64_bytes(token_bytes) for token_id, token_bytes in vocab_json.items()}
        merges = [(_decode_base64_bytes(left), _decode_base64_bytes(right)) for left, right in merges_json]

        return cls(
            vocab,
            merges,
            special_tokens,
        )
    
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterable[int]:
        """
        Encode an iterable of strings and yield token ids one by one.
        """
        for text in iterable:
            yield from self.encode(text)
    
    def decode(
        self,
        tokens: list[int],
    ) -> str:
        """
        Decode a list of tokens to a string.
        """
        token_bytes = b"".join(self.vocab[token] for token in tokens)
        try:
            return token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return token_bytes.decode("utf-8", errors="replace")


    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        Encode a string to a list of tokens.
        """
        # first pretokenize the text
        if not self.special_pattern:
            return self.encode_without_special_tokens(text)

        tokens = []
        last_end = 0
        for match in self.special_pattern.finditer(text):
            t = text[last_end : match.start()]
            # print("encode single token: [{}]".format(t))
            if len(t) > 0:
                tokens.extend(self.encode_without_special_tokens(t))
            token = match.group(0)
            tokens.append(self.reverse_vocab[token.encode("utf-8")])
            last_end = match.end()
        tokens.extend(self.encode_without_special_tokens(text[last_end:]))
        return tokens

    def encode_without_special_tokens(
        self,
        text: str,
    ) -> list[int]:
        """
        Encode a string to a list of tokens, without special tokens.
        """
        tokens = []
        for match in regex.finditer(bpe.BASE_PATTERN, text):
            token = match.group(0)
            if not token:
                continue
            tokens.extend(self.encode_single_token(token))
            
        return tokens
            

    def encode_single_token(
        self,
        token: str,
    ) -> list[int]:
        """
        Encode a single token to a list of tokens.
        """

        # print("token: [{}]".format(token))
        token_bytes = token.encode("utf-8")
        merged_tokens: list[bytes] = [bpe.BYTE_TOKENS[byte] for byte in token_bytes]

        while (True):
            min_priority = len(self.merges)
            min_merge = None
            for i in range(len(merged_tokens) - 1):
                left = merged_tokens[i]
                right = merged_tokens[i + 1]
                # print("left: [{}]".format(left))
                # print("right: [{}]".format(right))
                # print("merge_map left: ", self.merge_map.get(left, set()))

                if not (left in self.merge_map) or not (right in self.merge_map[left]):
                    continue
                
                priority = self.merge_priority[(left, right)]
                if priority < min_priority:
                    min_priority = priority
                    min_merge = (left, right)
            
            # print("min_merge: [{}], min_priority: {}".format(min_merge, min_priority))
            if min_merge is None:
                break
            new_merged_tokens = []
            i = 0
            while i < len(merged_tokens):
                if i< len(merged_tokens) - 1 and merged_tokens[i] == min_merge[0] and merged_tokens[i + 1] == min_merge[1]:
                    new_merged_tokens.append(min_merge[0] + min_merge[1])
                    # print("merge: [{}]".format(min_merge))
                    i += 2
                    continue
                new_merged_tokens.append(merged_tokens[i])
                i += 1
            # print("merged_tokens: [{}]".format(new_merged_tokens))
            merged_tokens = new_merged_tokens
        return [self.reverse_vocab[t] for t in merged_tokens]




    def _pre_tokenize(
        self,
        text: str,
    ) -> list[str]:
        """
        Pretokenize a string.
        """
        return regex.findall(bpe.BASE_PATTERN, text)
