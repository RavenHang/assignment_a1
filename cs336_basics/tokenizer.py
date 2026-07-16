import os
import regex as re
from typing import Iterable, Iterator
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(self,
                vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary,
        list of merges, and (optionally) a list of special tokens.
        This function should accept the following parameters:
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        max_id = max(self.vocab.keys()) if self.vocab else -1
        
        self.special_token_to_id = {}
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes in self.byte_to_id:
                self.special_token_to_id[st] = self.byte_to_id[st_bytes]
            else:
                max_id += 1
                self.vocab[max_id] = st_bytes
                self.byte_to_id[st_bytes] = max_id
                self.special_token_to_id[st] = max_id
            
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(st) for st in sorted_special_tokens]
            self.special_regex = re.compile(f"({'|'.join(escaped)})")
        else:
            self.special_regex = None
    
    def encode(self, text: str) ->list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        if self.special_regex:
            chunks = self.special_regex.split(text)
        else:
            chunks = [text]
        ids = []
        for chunk in chunks:
            if not chunk:
                continue
                
            if chunk in self.special_token_to_id:
                ids.append(self.special_token_to_id[chunk])
            else:
                ids.extend(self._encode_chunk(chunk))
                
        return ids
    
    def _encode_chunk(self, text_chunk: str) -> list[int]:
        ids = []
        for match in self.pat.finditer(text_chunk):
            pre_token = match.group()
            b_seq = [bytes([b]) for b in pre_token.encode('utf-8')]
            
            while len(b_seq) >= 2:
                best_pair = None
                best_rank = float('inf')
                
                for i in range(len(b_seq) - 1):
                    pair = (b_seq[i], b_seq[i+1])
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            
                if best_pair is None:
                    break
                
                new_b_seq = []
                i = 0
                while i < len(b_seq):
                    if i < len(b_seq) - 1 and (b_seq[i], b_seq[i+1]) == best_pair:
                        new_b_seq.append(b_seq[i] + b_seq[i+1])
                        i += 2
                    else:
                        new_b_seq.append(b_seq[i])
                        i += 1
                b_seq = new_b_seq
                
            for b in b_seq:
                ids.append(self.byte_to_id[b])
        return ids    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        b_list = []
        for token_id in ids:
            if token_id in self.vocab:
                b_list.append(self.vocab[token_id])
            else:
                pass 
                
        b_text = b"".join(b_list)
        return b_text.decode("utf-8", errors="replace")

def _process_chunk(chunk_text: str) -> Counter:
    local_counter = Counter()
    for match in PAT.finditer(chunk_text):
        local_counter[match.group()] += 1
    return local_counter
 
 
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    byte_to_id = {b: i for i, b in vocab.items()}
 
    max_id = 255
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in byte_to_id:
            max_id += 1
            vocab[max_id] = st_bytes
            byte_to_id[st_bytes] = max_id
 
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []
 
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()
 
    if special_tokens:
        sorted_special = sorted(special_tokens, key=len, reverse=True)
        split_pat = re.compile(f"({'|'.join(re.escape(st) for st in sorted_special)})")
        special_set = set(special_tokens)
        text_chunks = [p for p in split_pat.split(corpus) if p and p not in special_set]
    else:
        text_chunks = [corpus] if corpus else []
 
    pre_token_counts = Counter()
    if text_chunks:
        num_workers = min(os.cpu_count() or 1, len(text_chunks))
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = pool.map(_process_chunk, text_chunks)
                for local_counts in results:
                    pre_token_counts.update(local_counts)
        else:
            pre_token_counts = _process_chunk(text_chunks[0])
 
    words_map: dict[tuple[bytes, ...], int] = {}
    for word, count in pre_token_counts.items():
        b_seq = tuple(bytes([b]) for b in word.encode("utf-8"))
        words_map[b_seq] = words_map.get(b_seq, 0) + count
 
    pair_counts = defaultdict(int)
    for word_seq, count in words_map.items():
        for i in range(len(word_seq) - 1):
            pair_counts[(word_seq[i], word_seq[i + 1])] += count
 
    merges: list[tuple[bytes, bytes]] = []
 
    for _ in range(num_merges):
        if not pair_counts:
            break
 
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        max_freq = pair_counts[best_pair]
 
        if max_freq <= 0:
            break
 
        merged_token = best_pair[0] + best_pair[1]
        max_id += 1
        vocab[max_id] = merged_token
        merges.append(best_pair)
 
        new_words_map = {}
        for word_seq, count in words_map.items():
            contains_pair = False
            for i in range(len(word_seq) - 1):
                if (word_seq[i], word_seq[i + 1]) == best_pair:
                    contains_pair = True
                    break
 
            if not contains_pair:
                new_words_map[word_seq] = new_words_map.get(word_seq, 0) + count
                continue
 
            for i in range(len(word_seq) - 1):
                pair_counts[(word_seq[i], word_seq[i + 1])] -= count
 
            new_seq = []
            i = 0
            while i < len(word_seq):
                if i < len(word_seq) - 1 and (word_seq[i], word_seq[i + 1]) == best_pair:
                    new_seq.append(merged_token)
                    i += 2
                else:
                    new_seq.append(word_seq[i])
                    i += 1
            new_seq_tuple = tuple(new_seq)
 
            new_words_map[new_seq_tuple] = new_words_map.get(new_seq_tuple, 0) + count
 
            for i in range(len(new_seq_tuple) - 1):
                pair_counts[(new_seq_tuple[i], new_seq_tuple[i + 1])] += count
 
        words_map = new_words_map
 
        if best_pair in pair_counts:
            del pair_counts[best_pair]
        pair_counts = defaultdict(int, {p: c for p, c in pair_counts.items() if c > 0})
 
    return vocab, merges