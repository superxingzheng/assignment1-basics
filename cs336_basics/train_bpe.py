"""
stanford-cs336/assignment1-basics/cs336_basics/train_bpe.py
Last updated by: Zheng Xing <superxingzheng@gmail.com>

1. Unicode Encodings
Unicode standard maps about 150K characters to their integer representations.
About 98% of the web pages use UTF-8. Some characters are 1 byte in UTF-8 and
some characters are 2 bytes in UTF-8.
The roughly 150K vocabulary is converted to a 256 vocabulary using UTF-8 encoding.
2. Subword Tokenization
A sentence with 10 words might be 10 tokens long in a word-level language model.
But it could be 50 tokens long in a character-level language model, depending on
the length of the words. Subword Tokenization is a midpoint between world-level
and character-level tokenization. It is a trade-off between small vocabulary and
compression.
3. Bye-pair Encoding (BPE) Tokenizer
BPE Tokenizer is "trained" to merge most frequent pairs of bytes.
4. Vocabulary Initialization
The tokenizer vocabulary is a one-to-one mapping from integer IDs to tokens.
The initial BPE tokenizer vocabulary size is 256 plus 1 for <|endoftext|>.
5. Pre-tokenization
Merging the most frequently next-to-each-other bytes is computationally
expensive. The original BPE implementation of Sennrich et al. [2016] pre-tokenizes
by simply splitting on whitespace (i.e., s.split(" ")). In contract, we'll use
a regex-based pre-tokenizer (used by GPT-2; Radford et al, 2019). However, when the
input file is large, finding matches using the regex pattern can be very slow. Plus,
we need to count the frequency of each pair of bytes for the later merges. So,
we are splitting the input file into chunks and process the chunks in all CPU cores
in parallel.
6. Computing the BPE Merges is the main part of the "training" of the BPE tokenizer.
The pair of bytes with the highest frequency needs to be merged and added to the
vocabulary. A heap queue (priority queue) is used to sort the pairs according to
their frequency.
"""
import time
import multiprocessing as mp
import regex as re
from typing import BinaryIO
import os
from functools import reduce
import collections
import heapq
import pickle
import json


# Regex for coarse tokenization
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        merges_outpath: str = None,
        vocab_outpath: str = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, train a BPE tokenizer.
    Args:
        input_path: Path to the input text file.
        vocab_size: A positive integer that defines the maximum final vocabulary size
        (including the initial byte vocabulary, vocabulary items produced from merging,
        and any special tokens).
        special_tokens: A list of strings to add to the vocabulary.
        These special tokens do not otherwise affect BPE training.
    Returns:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the
        vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training.
        ach list item is a tuple of bytes (<token1>, <token2>), representing that <token1>
        was merged with <token2>. The merges should be ordered by order of creation.
    """
    train_start_time = time.time()

    initial_tokens = [tok.encode("UTF-8") for tok in special_tokens] + [bytes([i]) for i in range(256)]
    # print(initial_tokens)
    # The enumerate() returns a an iterator of tuples. Each tuple the index and the byte.
    vocab = {i: token for i, token in enumerate(initial_tokens)}
    # print(vocab)
    merges = []

    print("\nPre-tokenization: start")
    start_time = time.time()
    freqs = pre_tokenize(input_path, special_tokens)
    print(f"Pre-tokenization: finished in {time.time() - start_time:.2f} seconds.")

    #print("Initial sorting pair frequencies: start")
    start_time = time.time()
    pair_freqs, pairs_to_keys = get_pair_freqs(freqs)

    # Build a max-heap by pushing negative frequencies
    # Max Heap keeps the maximum element at the root.
    # Min Heap keeps the minimum element at the root.
    pair_heap = []
    for p, f in pair_freqs.items():
        if f > 0:
            heapq.heappush(pair_heap, (-f, ReverseLexOrderPair(p), p))
    #print(f"Initial sorting pair frequencies: finished in {time.time() - start_time:.2f} seconds.")
    # print(f"pair_heap: {pair_heap}")

    n_initial_tokens = len(initial_tokens)
    # The number of merges is the max vocabulary size minus the number of initial tokens.
    n_merges = vocab_size - n_initial_tokens

    #print("Merge: start")
    start_time = time.time()
    for i in range(n_initial_tokens, n_initial_tokens + n_merges):
        if not pair_heap:
            break
        while pair_heap:
            # Pop the smallest which is the highest frequency pair.
            neg_freq, _, top_pair = heapq.heappop(pair_heap)
            freq = - neg_freq
            # pair_freqs is a dict[tuple[bytes], int] mapping pairs to their frequencies.
            if pair_freqs.get(top_pair, 0) == freq:
                pair = top_pair
                break
            if top_pair in pair_freqs and pair_freqs[top_pair] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[top_pair], ReverseLexOrderPair(top_pair), top_pair))
        else:
            # If pair_heap is empty after the loop, we are done
            break

        if pair_freqs.get(pair, 0) <= 0:
            break

        # Add this new merge token to vocab and record the merge
        vocab[i] = pair[0] + pair[1]
        merges.append(pair)

        # Merge in freqs, then update the heap for pairs changed by this merge
        changed_pairs = merge(freqs, pair_freqs, pairs_to_keys, pair)

        for cp in changed_pairs:
            if cp in pair_freqs and pair_freqs[cp] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[cp], ReverseLexOrderPair(cp), cp))

        # Print progress every 100 merges or at the last iteration
        if ((i > n_initial_tokens) and ((i - n_initial_tokens + 1) % 100 == 0)) or (
                i == n_initial_tokens + n_merges - 1
        ):
            print(f"{i - n_initial_tokens + 1}/{n_merges} merges completed (merge runtime: {time.time() - start_time:.2f} seconds)")

    #print(f"Merges completed in {time.time() - start_time:.2f}s")
    print(f"Training completed in {time.time() - train_start_time:.2f}s")

    #print(f"vocab:\n{vocab}")
    #print(f"merges:\n{merges}")

    # Optionally save merges and vocab
    if merges_outpath:
        write_merges(merges, merges_outpath)
    if vocab_outpath:
        write_vocab(vocab, vocab_outpath)

    return vocab, merges


def pre_tokenize(input_path: str,
                 special_tokens: list[str]) -> dict[tuple[bytes], int]:
    """
    Break the input file opened in binary mode into chunks, using special tokens as
    boundaries.
    Using multiprocessing, apply regex-based pre-tokenization onto one chunk
    in each CPU process.

    Args:
        input_path: Path to the input text file.
        special_tokens: A list of strings as boundaries to split input file into chunks.

    Returns:
        A dictionary from pre-tokens to their frequency.
    """
    num_processes = mp.cpu_count()
    #print(f"num_processes: {num_processes}")
    pool = mp.Pool(processes=num_processes)
    chunk_freqs = []
    if special_tokens:
        # Join escaped tokens with "|" and compile the regex
        pattern_string = "|".join(re.escape(tok) for tok in special_tokens)
        special_pattern = re.compile(pattern_string)
    else:
        # Set to None if special_tokens is empty or falsy
        special_pattern = None

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Using zip and slicing to iterate through pairs of boundaries.
        # So, the start, end are boundaries[0], boundaries[1] and so on.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_str = chunk_bytes.decode("UTF-8", errors="ignore")
            # Launch one asynchronized background process for each chunk.
            # In each process, the chunk is pre-tokenized. A dictionary from
            # pre-tokens to their frequencies is returned.
            # A list of frequency dictionaries is collected from all processes.
            chunk_freqs.append(pool.apply_async(pre_tokenize_chunk, (chunk_str, special_pattern)))

    pool.close()
    pool.join()

    # Collect and merge partial results.
    # chunk_freqs is a list of dictionaries mapping pre-tokens to their frequencies.
    # These dictionaries are from all CPU processes and they are overlapping.
    # Here the res.get() is not a call to the dictionary method get(key).
    # It is a method of a multiprocessing.pool.AsyncResult object.
    # It is blocking, waiting for the corresponding process to return the element
    # to be appended into the chunk_freqs.
    # So, eventually here, freq_dicts will be a completed list of dictionaries.
    freq_dicts = [res.get() for res in chunk_freqs]
    # merge_freq_dicts() merges {} into freq_dicts.
    # Then it merges the next item in the freq_dicts into the last result.
    combined_freqs = reduce(merge_freq_dicts, freq_dicts, {})
    # print(f"Combined frequencies: {combined_freqs}")

    return combined_freqs


def find_chunk_boundaries(file: BinaryIO,
                          desired_num_chunks: int,
                          split_special_token: bytes) -> list[int]:
    """
    Find the precise positions of the split_special_token.
    Returns:
        A list of integer boundary positions.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END) # Move the read pointer to the end of the file.
    file_size = file.tell() # Return the integer of the current byte offset from the beginning of the file.
    file.seek(0) # Move the read pointer back to the beginning.
    chunk_size = file_size // desired_num_chunks # Using floor division //

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size # Forcing the last boundary to be EOF.

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        pos = chunk_boundaries[bi]
        file.seek(pos)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                # If EOF is reached before finding split token
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # Found the split token; adjust boundary precisely
                chunk_boundaries[bi] = pos + found_at
                break
            pos += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(
        chunk: str,
        special_pattern: re.Pattern | None) -> dict[tuple[bytes], int]:
    """
    Regex pre-tokenizes the chunk.
    Splits first on special tokens, then uses PAT which is a global variable.

    Args:
        chunk: The chunk to pre-tokenize.
        special_pattern: A regular expression used to split the chunk at the special token.

    Returns:
        A dictionary from pre-tokens to their frequency.
    """
    # Initialize a dictionary indexed by tuples of Bytes (pre-tokens). The values are integer
    # frequency of the pre-tokens which are represented as tuples of Bytes.
    freqs: dict[tuple[bytes], int] = {}

    sub_chunks = special_pattern.split(chunk) if special_pattern else [chunk]

    for sub_chunk in sub_chunks:
        # Pre-tokenization using regex from GTP-2.
        # Using PAT.finditer() instead of PAT.findall() to avoid storing too many pre-tokens in RAM.
        for match in PAT.finditer(sub_chunk):
            # Convert the pre-token from a List of Bytes into a tuple of Bytes.
            match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
            # Because the freqs dictionary is indexed by the tuples of Bytes,
            # freqs.get(match_bytes, 0) will return the value for the match_bytes.
            # When the math_bytes is not found in the current dictionary, default 0
            # will be returned.
            # For every iterator returned by the PAT.finditer(), this will increase
            # the value for the matching pre-token by 1.
            freqs[match_bytes] = freqs.get(match_bytes, 0) + 1

    return freqs


def merge_freq_dicts(
        dict1: dict[tuple[bytes], int],
        dict2: dict[tuple[bytes], int]
) -> dict[tuple[bytes], int]:
    """
    Adds frequencies from dict2 into dict1.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        result[key] = result.get(key, 0) + value

    return result


def get_pair_freqs(
        freqs: dict[tuple[bytes], int]
) -> tuple[dict[tuple[bytes, bytes], int],
dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    """
    Builds a pair-frequency table and reverse mapping (pair -> set of keys/pre-tokens).
    """
    pair_freqs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]] = collections.defaultdict(set)

    # Iterate through all the pre-tokens and their frequencies.
    # Count all the pair frequencies and fill the pair_freqs dictionary.
    for pre_token, freq in freqs.items():
        # Iterate through all the pairs in each pre-token.
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i + 1])
            pair_freqs[pair] += freq

            # Build the mapping from Byte-pairs to sets of pre-tokens.
            pairs_to_keys[pair].add(pre_token)

    return pair_freqs, pairs_to_keys


class ReverseLexOrderPair:
    """
    Encapsulates (bytes, bytes) so that in a min-heap, the "largest in normal lex order"
    is treated as the smallest. Ensures that tie frequencies pop in reverse lex order.
    """

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "ReverseLexOrderPair") -> bool:
        # Invert normal order: self < other if self is > other (so larger lex sorts first).
        return self.pair > other.pair

    def __eq__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair == other.pair


def build_new_repr(old_repr: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    """
    Replaces every occurrence of pair=(x,y) in old_repr with the merged symbol x+y.
    """
    new_symbols = []
    i = 0
    while i < len(old_repr):
        if i < len(old_repr) - 1 and old_repr[i] == pair[0] and old_repr[i + 1] == pair[1]:
            new_symbols.append(old_repr[i] + old_repr[i + 1])  # merges, e.g. b'A' + b'B' => b'AB'
            i += 2
        else:
            new_symbols.append(old_repr[i])
            i += 1
    return tuple(new_symbols)


def merge(
    freqs: dict[tuple[bytes], int],
    pair_freqs: dict[tuple[bytes, bytes], int],
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]],
    pair: tuple[bytes, bytes],
) -> set[tuple[bytes, bytes]]:
    """
    Merges 'pair' into freqs and updates pair_freqs & pairs_to_keys for all affected old/new keys.
    """
    changed_pairs = set()
    keys_to_modify = pairs_to_keys[pair].copy()

    for old_key in keys_to_modify:
        old_freq = freqs.pop(old_key)
        new_key = build_new_repr(old_key, pair)

        # Decrement frequencies in pair_freqs for old_key's adjacencies
        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i + 1]
            pair_freqs[left, right] -= old_freq
            changed_pairs.add((left, right))
            if pair_freqs[left, right] <= 0:
                del pair_freqs[left, right]
            pairs_to_keys[left, right].discard(old_key)

        # Increment frequencies for new_key's adjacencies
        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i + 1]
            pair_freqs[left, right] += old_freq
            changed_pairs.add((left, right))
            pairs_to_keys[left, right].add(new_key)

        # Put new_key back with updated freq
        freqs[new_key] = freqs.get(new_key, 0) + old_freq

    pairs_to_keys[pair] = set()

    return changed_pairs


def write_merges(merges, outpath):
    """
    Pickle the merges list to a binary file.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for pair in merges:
            # pair is typically (token_a, token_b)
            # This writes them separated by a space
            f.write(f"{pair[0]} {pair[1]}\n")
    print(f"Saved {len(merges)} merges to {outpath}")


def write_vocab(vocab, outpath):
    """
    Pickle the vocab dict to a binary file.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for idx, token in vocab.items():
            f.write(f"{idx}: {token}\n")
    print(f"Saved vocabulary with {len(vocab)} tokens to {outpath}")


if __name__ == "__main__":
    (vocab, merge) = train_bpe(
        #input_path = "./data/TinyStoriesV2-GPT4-valid.txt",
        #input_path = "./data/TinyStoriesV2-GPT4-train.txt",
        input_path  = "./data/owt_train.txt",
        #vocab_size=10000,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        merges_outpath="./out/merges.txt",
        vocab_outpath="./out/vocab.txt"
    )