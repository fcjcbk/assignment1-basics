from collections.abc import Iterable
import regex
import cs336_basics.bpe as bpe

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

        

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Create a tokenizer from files.
        """
        # with open(vocab_filepath, "rb") as f:
        #     vocab = pickle.load(f)
        # with open(merges_filepath, "rb") as f:
        #     merges = pickle.load(f)
        # return cls(
        #     vocab,
        #     merges,
        #     special_tokens,
        # )
        pass
    
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterable[int]:
        """
        Encode a list of strings to a list of lists of tokens.
        """
        return (self.encode(text) for text in iterable)
    
    def decode(
        self,
        tokens: list[int],
    ) -> str:
        """
        Decode a list of tokens to a string.
        """
        decoded_text = ""
        for token in tokens:
            decoded_text += self.vocab[token].decode("utf-8")
        return decoded_text


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
            tokens = tokens + self.encode_without_special_tokens(text[last_end : match.start()])
            token = match.group(0)
            tokens = tokens + self.reverse_vocab[token.encode("utf-8")]
            last_end = match.end()
        tokens = tokens + self.encode_without_special_tokens(text[last_end:])
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
            tokens = tokens + self.encode_single_token(token)
            
        return tokens
            

    def encode_single_token(
        self,
        token: str,
    ) -> list[int]:
        """
        Encode a single token to a list of tokens.
        """
        token_bytes = token.encode("utf-8")
        merged_tokens: list[bytes] = [bpe.BYTE_TOKENS[byte] for byte in token_bytes]

        has_merged = True
        while (has_merged):
            has_merged = False
            new_merged_tokens: list[bytes] = []
            i = 1
            left = merged_tokens[0]
            while i < len(merged_tokens):
                right = merged_tokens[i]

                if left in self.merge_map and right in self.merge_map[left]:
                    left = left + right
                else:
                    new_merged_tokens.append(left)
                    left = right
                i += 1
            new_merged_tokens.append(left)
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
