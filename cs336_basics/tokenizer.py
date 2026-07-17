from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Iterable, Iterator

import regex as re


# GPT-2 风格预分词正则
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        根据 vocabulary、BPE merge 规则和特殊 token 构造 tokenizer。
        """
        self.vocab = vocab.copy()
        self.merges = list(merges)
        self.special_tokens = (
            list(special_tokens) if special_tokens is not None else []
        )

        self.byte_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        max_id = max(self.vocab, default=-1)

        self.special_token_to_id: dict[str, int] = {}

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")

            if special_bytes in self.byte_to_id:
                token_id = self.byte_to_id[special_bytes]
            else:
                max_id += 1
                token_id = max_id

                self.vocab[token_id] = special_bytes
                self.byte_to_id[special_bytes] = token_id

            self.special_token_to_id[special_token] = token_id

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        if self.special_tokens:
            sorted_special_tokens = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )

            escaped_tokens = [
                re.escape(token)
                for token in sorted_special_tokens
            ]

            self.special_regex = re.compile(
                f"({'|'.join(escaped_tokens)})"
            )
        else:
            self.special_regex = None

    def encode(self, text: str) -> list[int]:
        """
        将字符串编码成 token ID。
        """
        if self.special_regex is not None:
            chunks = self.special_regex.split(text)
        else:
            chunks = [text]

        token_ids: list[int] = []

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[chunk])
            else:
                token_ids.extend(self._encode_chunk(chunk))

        return token_ids

    def _encode_chunk(self, text_chunk: str) -> list[int]:
        """
        对不包含特殊 token 的普通文本执行预分词和 BPE。
        """
        token_ids: list[int] = []

        for match in PAT.finditer(text_chunk):
            pre_token = match.group()

            symbols = [
                bytes([byte_value])
                for byte_value in pre_token.encode("utf-8")
            ]

            while len(symbols) >= 2:
                best_pair: tuple[bytes, bytes] | None = None
                best_rank = float("inf")

                for index in range(len(symbols) - 1):
                    pair = (
                        symbols[index],
                        symbols[index + 1],
                    )

                    rank = self.merge_ranks.get(pair)

                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pair = pair

                if best_pair is None:
                    break

                merged_symbols: list[bytes] = []
                index = 0

                while index < len(symbols):
                    if (
                        index + 1 < len(symbols)
                        and symbols[index] == best_pair[0]
                        and symbols[index + 1] == best_pair[1]
                    ):
                        merged_symbols.append(
                            symbols[index] + symbols[index + 1]
                        )
                        index += 2
                    else:
                        merged_symbols.append(symbols[index])
                        index += 1

                symbols = merged_symbols

            for symbol in symbols:
                token_id = self.byte_to_id.get(symbol)

                if token_id is None:
                    raise ValueError(
                        f"Vocabulary 中找不到 token：{symbol!r}"
                    )

                token_ids.append(token_id)

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        逐段编码文本，避免一次性加载全部待编码文本。
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 解码为 UTF-8 字符串。
        """
        byte_parts: list[bytes] = []

        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(f"未知 token ID：{token_id}")

            byte_parts.append(self.vocab[token_id])

        return b"".join(byte_parts).decode(
            "utf-8",
            errors="replace",
        )


# ============================================================
# 并行预分词
# ============================================================

def _find_next_delimiter_end(
    file_obj,
    start_offset: int,
    delimiter: bytes,
    file_size: int,
    block_size: int = 1024 * 1024,
) -> int:
    """
    从 start_offset 开始向后寻找 delimiter。

    返回 delimiter 结束位置，从而保证文件块边界位于完整文档之间。
    """
    if start_offset >= file_size:
        return file_size

    file_obj.seek(start_offset)

    overlap_size = max(0, len(delimiter) - 1)
    overlap = b""

    while True:
        block_start = file_obj.tell()
        block = file_obj.read(block_size)

        if not block:
            return file_size

        search_data = overlap + block
        search_data_start = block_start - len(overlap)

        delimiter_index = search_data.find(delimiter)

        if delimiter_index >= 0:
            return (
                search_data_start
                + delimiter_index
                + len(delimiter)
            )

        if overlap_size > 0:
            overlap = search_data[-overlap_size:]
        else:
            overlap = b""


def _build_document_aligned_ranges(
    input_path: str,
    num_ranges: int,
    delimiter: bytes | None,
) -> list[tuple[int, int]]:
    """
    将文件划分成若干区间。

    当存在 <|endoftext|> 时，每个区间边界都会对齐到文档边界，
    不会切断 UTF-8 字符，也不会切断一个故事。
    """
    file_size = os.path.getsize(input_path)

    if file_size == 0:
        return []

    if num_ranges <= 1 or not delimiter:
        return [(0, file_size)]

    boundaries = [0]

    with open(input_path, "rb") as file_obj:
        for range_index in range(1, num_ranges):
            approximate_offset = (
                file_size * range_index // num_ranges
            )

            boundary = _find_next_delimiter_end(
                file_obj=file_obj,
                start_offset=approximate_offset,
                delimiter=delimiter,
                file_size=file_size,
            )

            if boundary > boundaries[-1] and boundary < file_size:
                boundaries.append(boundary)

    boundaries.append(file_size)

    return [
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
        if boundaries[index] < boundaries[index + 1]
    ]


def _count_pre_tokens_in_text(
    text: str,
    counter: Counter[bytes],
) -> None:
    """
    对一段不包含特殊 token 的文本进行预分词。
    Counter 的 key 直接保存 UTF-8 bytes，避免主进程再次编码。
    """
    for match in PAT.finditer(text):
        pre_token_bytes = match.group().encode("utf-8")
        counter[pre_token_bytes] += 1


def _pretokenize_file_range(
    task: tuple[str, int, int, tuple[str, ...]],
) -> Counter[bytes]:
    """
    multiprocessing worker。

    每个 worker 自己读取文件对应区间，避免主进程通过进程管道
    传输大段 corpus。
    """
    input_path, start_offset, end_offset, special_tokens = task

    with open(input_path, "rb") as file_obj:
        file_obj.seek(start_offset)
        raw_data = file_obj.read(end_offset - start_offset)

    text = raw_data.decode("utf-8")
    local_counter: Counter[bytes] = Counter()

    if not special_tokens:
        _count_pre_tokens_in_text(text, local_counter)
        return local_counter

    sorted_special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    special_pattern = re.compile(
        "|".join(
            re.escape(token)
            for token in sorted_special_tokens
        )
    )

    previous_end = 0

    # 特殊 token 本身不进入普通 BPE 预分词
    for match in special_pattern.finditer(text):
        normal_text = text[previous_end:match.start()]

        if normal_text:
            _count_pre_tokens_in_text(
                normal_text,
                local_counter,
            )

        previous_end = match.end()

    remaining_text = text[previous_end:]

    if remaining_text:
        _count_pre_tokens_in_text(
            remaining_text,
            local_counter,
        )

    return local_counter


def _pretokenize_file(
    input_path: str,
    special_tokens: list[str],
    num_workers: int,
) -> Counter[bytes]:
    """
    并行完成整个数据集的预分词。
    """
    if num_workers <= 0:
        num_workers = min(os.cpu_count() or 1, 8)

    # TinyStories 使用第一个特殊 token 分隔文档
    delimiter = (
        special_tokens[0].encode("utf-8")
        if special_tokens
        else None
    )

    file_size = os.path.getsize(input_path)

    # 每个 worker 分到多个任务，以改善负载均衡。
    # 同时限制最小块大小，避免生成大量小任务。
    desired_ranges = max(1, num_workers * 4)
    minimum_range_size = 8 * 1024 * 1024

    maximum_useful_ranges = max(
        1,
        file_size // minimum_range_size,
    )

    desired_ranges = min(
        desired_ranges,
        maximum_useful_ranges,
    )

    ranges = _build_document_aligned_ranges(
        input_path=input_path,
        num_ranges=desired_ranges,
        delimiter=delimiter,
    )

    if not ranges:
        return Counter()

    tasks = [
        (
            input_path,
            start_offset,
            end_offset,
            tuple(special_tokens),
        )
        for start_offset, end_offset in ranges
    ]

    worker_count = min(
        num_workers,
        len(tasks),
    )

    global_counter: Counter[bytes] = Counter()

    if worker_count <= 1:
        for task in tasks:
            global_counter.update(
                _pretokenize_file_range(task)
            )

        return global_counter

    with Pool(processes=worker_count) as pool:
        # imap_unordered 可以逐个回收结果，降低峰值内存
        for local_counter in pool.imap_unordered(
            _pretokenize_file_range,
            tasks,
            chunksize=1,
        ):
            global_counter.update(local_counter)

    return global_counter


# ============================================================
# 增量 BPE 训练
# ============================================================

class _MaxHeapPair:
    """
    heapq 默认是最小堆，本类反转比较规则，使其表现为最大堆。

    排序规则与下面的朴素写法一致：

        max(pair_counts, key=lambda p: (pair_counts[p], p))

    先选频率最大者；频率相同时，选择 bytes pair 字典序最大者。
    """

    __slots__ = (
        "count",
        "left_bytes",
        "right_bytes",
        "pair",
    )

    def __init__(
        self,
        count: int,
        left_bytes: bytes,
        right_bytes: bytes,
        pair: tuple[int, int],
    ):
        self.count = count
        self.left_bytes = left_bytes
        self.right_bytes = right_bytes
        self.pair = pair

    def __lt__(self, other: "_MaxHeapPair") -> bool:
        if self.count != other.count:
            return self.count > other.count

        if self.left_bytes != other.left_bytes:
            return self.left_bytes > other.left_bytes

        if self.right_bytes != other.right_bytes:
            return self.right_bytes > other.right_bytes

        return self.pair > other.pair


def _count_adjacent_pairs(
    sequence: list[int],
) -> dict[tuple[int, int], int]:
    """
    统计一个 pre-token 内部各 pair 出现次数。

    注意：
        对于 [a, a, a]，pair (a, a) 出现两次。
    """
    local_counts: dict[tuple[int, int], int] = {}

    for index in range(len(sequence) - 1):
        pair = (
            sequence[index],
            sequence[index + 1],
        )

        local_counts[pair] = local_counts.get(pair, 0) + 1

    return local_counts


def _merge_pair_in_sequence(
    sequence: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    """
    从左到右合并 sequence 中所有不重叠的目标 pair。
    """
    left_token_id, right_token_id = pair

    merged_sequence: list[int] = []
    index = 0

    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == left_token_id
            and sequence[index + 1] == right_token_id
        ):
            merged_sequence.append(new_token_id)
            index += 2
        else:
            merged_sequence.append(sequence[index])
            index += 1

    return merged_sequence


def _initialize_bpe_state(
    pre_token_counts: Counter[bytes],
) -> tuple[
    list[list[int]],
    list[int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], set[int]],
]:
    """
    初始化增量 BPE 训练所需的数据结构。

    word_sequences[word_id]
        当前 pre-token 的符号序列。

    word_frequencies[word_id]
        该 pre-token 在 corpus 中出现的次数。

    pair_counts[pair]
        pair 在整个 corpus 中的加权出现次数。

    pair_to_word_ids[pair]
        当前包含该 pair 的所有 pre-token ID。
    """
    word_sequences: list[list[int]] = []
    word_frequencies: list[int] = []

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_word_ids: dict[
        tuple[int, int],
        set[int],
    ] = defaultdict(set)

    for pre_token_bytes, frequency in pre_token_counts.items():
        # bytes 可直接迭代为 0~255 的整数
        sequence = list(pre_token_bytes)

        word_id = len(word_sequences)

        word_sequences.append(sequence)
        word_frequencies.append(frequency)

        local_pair_counts = _count_adjacent_pairs(sequence)

        for pair, local_count in local_pair_counts.items():
            pair_counts[pair] += local_count * frequency
            pair_to_word_ids[pair].add(word_id)

    return (
        word_sequences,
        word_frequencies,
        dict(pair_counts),
        dict(pair_to_word_ids),
    )


def _run_bpe_merges(
    vocab: dict[int, bytes],
    pre_token_counts: Counter[bytes],
    num_merges: int,
) -> list[tuple[bytes, bytes]]:
    """
    增量执行 BPE merges。

    与原始代码不同，这里不会在每次 merge 时扫描所有 pre-token。
    每轮只访问包含当前 best_pair 的词。
    """
    (
        word_sequences,
        word_frequencies,
        pair_counts,
        pair_to_word_ids,
    ) = _initialize_bpe_state(pre_token_counts)

    pair_heap: list[_MaxHeapPair] = []

    for pair, count in pair_counts.items():
        left_token_id, right_token_id = pair

        pair_heap.append(
            _MaxHeapPair(
                count=count,
                left_bytes=vocab[left_token_id],
                right_bytes=vocab[right_token_id],
                pair=pair,
            )
        )

    heapq.heapify(pair_heap)

    merges: list[tuple[bytes, bytes]] = []
    max_token_id = max(vocab, default=-1)

    for _ in range(num_merges):
        best_pair: tuple[int, int] | None = None

        # Lazy deletion：
        # pair 频率变化后旧 heap item 不立刻删除，
        # 弹出时检查其频率是否仍然有效。
        while pair_heap:
            heap_item = heapq.heappop(pair_heap)
            current_count = pair_counts.get(
                heap_item.pair,
                0,
            )

            if (
                current_count > 0
                and current_count == heap_item.count
            ):
                best_pair = heap_item.pair
                break

        if best_pair is None:
            break

        left_token_id, right_token_id = best_pair

        left_bytes = vocab[left_token_id]
        right_bytes = vocab[right_token_id]
        merged_bytes = left_bytes + right_bytes

        max_token_id += 1
        new_token_id = max_token_id

        vocab[new_token_id] = merged_bytes
        merges.append((left_bytes, right_bytes))

        affected_word_ids = tuple(
            pair_to_word_ids.get(best_pair, ())
        )

        # 汇总所有受影响词对全局 pair frequency 的改变量。
        # 同一 pair 在多个词中变化时，只向 heap 写入一次新值。
        pair_deltas: dict[
            tuple[int, int],
            int,
        ] = defaultdict(int)

        for word_id in affected_word_ids:
            old_sequence = word_sequences[word_id]
            frequency = word_frequencies[word_id]

            old_local_counts = _count_adjacent_pairs(
                old_sequence
            )

            new_sequence = _merge_pair_in_sequence(
                sequence=old_sequence,
                pair=best_pair,
                new_token_id=new_token_id,
            )

            new_local_counts = _count_adjacent_pairs(
                new_sequence
            )

            all_local_pairs = (
                old_local_counts.keys()
                | new_local_counts.keys()
            )

            for pair in all_local_pairs:
                old_local_count = old_local_counts.get(pair, 0)
                new_local_count = new_local_counts.get(pair, 0)

                local_delta = (
                    new_local_count - old_local_count
                )

                if local_delta != 0:
                    pair_deltas[pair] += (
                        local_delta * frequency
                    )

                old_contains_pair = old_local_count > 0
                new_contains_pair = new_local_count > 0

                if old_contains_pair and not new_contains_pair:
                    word_ids = pair_to_word_ids.get(pair)

                    if word_ids is not None:
                        word_ids.discard(word_id)

                        if not word_ids:
                            pair_to_word_ids.pop(
                                pair,
                                None,
                            )

                elif (
                    not old_contains_pair
                    and new_contains_pair
                ):
                    pair_to_word_ids.setdefault(
                        pair,
                        set(),
                    ).add(word_id)

            word_sequences[word_id] = new_sequence

        changed_pairs: list[tuple[int, int]] = []

        for pair, delta in pair_deltas.items():
            if delta == 0:
                continue

            new_count = pair_counts.get(pair, 0) + delta

            if new_count > 0:
                pair_counts[pair] = new_count
                changed_pairs.append(pair)
            else:
                pair_counts.pop(pair, None)

        # best_pair 已经在所有受影响词中被合并，因此应当消失。
        pair_counts.pop(best_pair, None)
        pair_to_word_ids.pop(best_pair, None)

        # 只为频率发生变化的 pair 写入新的 heap item。
        # 旧 item 留在 heap 中，由 lazy deletion 过滤。
        for pair in changed_pairs:
            if pair == best_pair:
                continue

            current_count = pair_counts.get(pair, 0)

            if current_count <= 0:
                continue

            current_left_id, current_right_id = pair

            heapq.heappush(
                pair_heap,
                _MaxHeapPair(
                    count=current_count,
                    left_bytes=vocab[current_left_id],
                    right_bytes=vocab[current_right_id],
                    pair=pair,
                ),
            )

    return merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """
    训练 byte-level BPE tokenizer。

    Parameters
    ----------
    input_path:
        TinyStories 文本文件路径。

    vocab_size:
        最终最大词表大小。包括：
        - 256 个原始 byte token
        - special tokens
        - BPE merge 生成的 token

    special_tokens:
        特殊 token，例如 ["<|endoftext|>"]。

    num_workers:
        预分词进程数。None 时最多使用 8 个进程。
    """
    input_path = os.fspath(input_path)

    if vocab_size <= 0:
        raise ValueError("vocab_size 必须大于 0")

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    # 0~255 对应所有可能的单字节
    vocab: dict[int, bytes] = {
        byte_value: bytes([byte_value])
        for byte_value in range(256)
    }

    byte_to_id = {
        token_bytes: token_id
        for token_id, token_bytes in vocab.items()
    }

    max_token_id = 255

    # 特殊 token 直接进入词表，不参与普通 BPE merge
    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")

        if special_bytes not in byte_to_id:
            max_token_id += 1

            vocab[max_token_id] = special_bytes
            byte_to_id[special_bytes] = max_token_id

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} 小于初始词表大小 "
            f"{len(vocab)}"
        )

    num_merges = vocab_size - len(vocab)

    if num_merges == 0:
        return vocab, []

    pre_token_counts = _pretokenize_file(
        input_path=input_path,
        special_tokens=special_tokens,
        num_workers=num_workers,
    )

    merges = _run_bpe_merges(
        vocab=vocab,
        pre_token_counts=pre_token_counts,
        num_merges=num_merges,
    )

    return vocab, merges