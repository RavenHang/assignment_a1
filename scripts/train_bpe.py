from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import threading
import time
from pathlib import Path

from cs336_basics.tokenizer import train_bpe


try:
    import psutil
except ImportError:
    psutil = None


class PeakMemoryMonitor:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if psutil is None:
            return

        self._thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> float:
        if psutil is not None and self._thread is not None:
            self._stop_event.set()
            self._thread.join()

            return self.peak_bytes / (1024**2)

        return get_fallback_peak_rss_mb()

    def _monitor(self) -> None:
        process = psutil.Process(os.getpid())

        while not self._stop_event.is_set():
            total_rss = 0

            try:
                total_rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            try:
                children = process.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                children = []

            for child in children:
                try:
                    total_rss += child.memory_info().rss
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                ):
                    continue

            self.peak_bytes = max(
                self.peak_bytes,
                total_rss,
            )

            self._stop_event.wait(self.interval_seconds)


def get_fallback_peak_rss_mb() -> float:
    try:
        import resource
    except ImportError:
        return float("nan")

    max_rss = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    if sys.platform == "darwin":
        return max_rss / (1024**2)

    return max_rss / 1024


def bytes_to_display(token_bytes: bytes) -> str:
    return token_bytes.decode(
        "utf-8",
        errors="replace",
    )

def find_longest_token(
    vocab: dict[int, bytes],
) -> tuple[int, bytes]:
    return max(
        vocab.items(),
        key=lambda item: (
            len(item[1]),
            item[1],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train byte-level BPE on TinyStories"
    )

    parser.add_argument(
        "--name",
        required=True,
        help="输出文件名前缀",
    )

    parser.add_argument(
        "--input-path",
        required=True,
        help="TinyStories 数据文件",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="最大 vocabulary 大小",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="预分词进程数量",
    )

    parser.add_argument(
        "--output-dir",
        default="./bpe_output",
        help="vocab、merges 和 profile 输出目录",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="开启 cProfile；",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = Path(args.input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"找不到输入文件：{input_path}"
        )

    output_prefix = output_dir / args.name

    memory_monitor = PeakMemoryMonitor()
    memory_monitor.start()

    profiler = cProfile.Profile()
    start_time = time.perf_counter()

    if args.profile:
        profiler.enable()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        num_workers=args.workers,
    )

    if args.profile:
        profiler.disable()

    elapsed_seconds = time.perf_counter() - start_time
    peak_memory_mb = memory_monitor.stop()

    if args.profile:
        profile_path = Path(
            f"{output_prefix}.prof"
        )

        profiler.dump_stats(profile_path)

        print("\nProfile 中累计耗时最高的 30 个函数：")
        pstats.Stats(profiler).strip_dirs().sort_stats(
            "cumtime"
        ).print_stats(30)

        print(f"Profile 文件：{profile_path}")

    longest_token_id, longest_token = find_longest_token(
        vocab
    )

    print("\n" + "=" * 70)
    print("训练完成")
    print(f"实际词表大小：{len(vocab)}")
    print(f"Merge 数量：{len(merges)}")
    print(f"训练时间：{elapsed_seconds:.3f} 秒")
    print(f"峰值内存：{peak_memory_mb:.2f} MiB")
    print()
    print(f"最长 Token ID：{longest_token_id}")
    print(f"最长 Token 字节数：{len(longest_token)}")
    print(f"最长 Token repr：{longest_token!r}")
    print(f"最长 Token hex：{longest_token.hex()}")
    print(
        "最长 Token 文本："
        f"{bytes_to_display(longest_token)!r}"
    )
    print()
    print("=" * 70)

    print("\n 第一问结果：")
    print(
        f"Training took {elapsed_seconds:.2f} seconds with "
        f"approximately {peak_memory_mb:.2f} MiB peak RSS. "
        f"The longest token was "
        f"{bytes_to_display(longest_token)!r} "
        f"({len(longest_token)} bytes)."
    )

if __name__ == "__main__":
    main()