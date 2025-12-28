import time
import regex as re
import pickle
from collections.abc import Iterable, Iterator

from cs336_basics import train_bpe


class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
    ):
        """
        Constructs a tokenizer from a vocab, list of merges,
        and (optionally) list of special tokens.
        """
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}
        self.encode_cache = {}
        self.cache_hits = 0

        self.pretokenize_pattern = re.compile(train_bpe.PAT)

        if special_tokens:
            # Sort the special tokens in descent length.
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            # Join the escaped tokens with the regex "OR" operator (|).
            # Wrapping the entire expression in () creates a Capturing Group.
            # The matched delimiters are included in the resulting list.
            self.special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"

            next_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                # Search the special token bytes in the inverse vocabulary dictionary.
                # Append the special token bytes into both vocab and vocab_inv.
                if token_bytes not in self.vocab_inv:
                    self.vocab[next_id] = token_bytes
                    self.vocab_inv[token_bytes] = next_id
                    next_id += 1
        else:
            self.special_tokens = None
            self.special_pattern = None


    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] = None
    ):
        """
        Constructs a Tokenizer from a serialized vocab dictionary,
        a serialized list of merges, and (optionally) list of special tokens.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        """
        Encodes an input text into a sequence of token IDs.
        """
        if not self.special_tokens:
            return self._encode_chunk(text)

        # Using regex, split the input text at the special tokens.
        chunks = re.split(self.special_pattern, text)

        ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.vocab_inv[chunk.encode("UTF-8")])
            else:
                ids.extend(self._encode_chunk(chunk))

        return ids


    def _encode_chunk(self, chunk: str) -> list[int]:
        """
        Encodes a chunk of text into a sequence of token IDs.
        """
        pretokens = self._pretokenize(chunk)
        pretoken_reprs: dict[str, list[bytes]] = {}

        ids = []
        for p in pretokens:
            # Search pretokens in the encode cache.
            if p in self.encode_cache:
                ids.extend(self.encode_cache[p])
                self.cache_hits += 1
            else:
                if p not in pretoken_reprs:
                    match_bytes = list(bytes([b]) for b in p.encode("UTF-8"))
                    pretoken_reprs[p] = match_bytes

                # This is matching the process of training BPE tokenizer.
                # Input text is split into chunks. Chunks are split into pre-tokens.
                # Each pre-token is encoded into bytes and then split into a list of
                # bytes. Then the bytes are merged according to the same merges from
                # the training record.
                merged = self._merge_subword(pretoken_reprs[p])
                token_ids = [self.vocab_inv[subword] for subword in merged]
                self.encode_cache[p] = token_ids
                ids.extend(token_ids)

        return ids


    def _pretokenize(self, text: str) -> list[str]:
        """
        Pre-tokenizes a chunk of text into a list of pre-tokens (strings).
        """
        pretokens: list[str] = []
        for match in self.pretokenize_pattern.finditer(text):
            match_str = match.group()
            pretokens.append(match_str)

        return pretokens


    def _merge_subword(self, rep: list[bytes]) -> list[bytes]:
        """
        Given a list of subword units (bytes), repeatedly merges adjacent pairs
        in ascending rank order until no more merges are found.
        """
        while True:
            best_rank = float("inf")
            best_idx = None

            # Scan adjacent pairs
            for i in range(len(rep) - 1):
                pair = (rep[i], rep[i + 1])
                rank = self.merges_dict.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            # If no merges found, we're done
            if best_idx is None:
                return rep

            # Merge the best pair
            merged = rep[best_idx] + rep[best_idx + 1]  # Concatenate bytes
            rep = rep[:best_idx] + [merged] + rep[best_idx + 2 :]


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Yields token IDs lazily from an iterable of strings (e.g., a file handle).
        """
        for text in iterable:
            yield from self.encode(text)


    def decode(self, ids: list[int]) -> str:
        """
        Decodes a sequence of token IDs into text.
        """
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("UTF-8", errors="replace")



if __name__ == "__main__":
    # input_path = "./data/owt_train.txt"
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_path = "./out_backup/TinyStories_vocab.pkl"
    merge_path = "./out_backup/TinyStories_merges.pkl"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens)

    with open(input_path, "r", encoding="utf-8") as f:
        texts = f.read()

    start_time = time.time()
    ids = tokenizer.encode(texts)
    end_time = time.time()
    print(f"Encoding took {end_time - start_time} seconds")
    print(f"Throughput of the tokenizer: {len(texts)/(end_time - start_time)} bytes/second")
    #print(ids)