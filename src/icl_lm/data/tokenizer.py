import tiktoken
import re

class Tokenizer:
    def __init__(self):
        base_enc = tiktoken.get_encoding("gpt2")

        special_tokens = {
            "<|pad|>": base_enc.n_vocab,
            "<|bos|>": base_enc.n_vocab + 1,
            "<|eos|>": base_enc.n_vocab + 2,
        }
        
        for i in range(len(special_tokens), 10):
            special_tokens[f"<|reserved_{i}|>"] = base_enc.n_vocab + i

        self.set_special_tokens(special_tokens)

        number_pattern = r"\d{1,3}"  # Match 0–999 as separate tokens
        fallback_pattern = base_enc._pat_str
        custom_pat_str = f"{number_pattern}|{fallback_pattern}"

        # Total vocab size of 50267
        self.tokenizer = tiktoken.Encoding(
            name="icl-tokenizer",
            pat_str=custom_pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens=special_tokens,
        )
        
    def set_special_tokens(self, token_map):
        self.special_tokens = token_map
        for token, idx in self.special_tokens.items():
            name_match = re.match(r"<\|([a-zA-Z0-9_]+)\|>", token)
            if name_match:
                name = name_match.group(1)
                setattr(self, f"{name}_token_id", idx)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)