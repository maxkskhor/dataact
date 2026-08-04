from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator


@dataclass
class TurnMetrics:
    turn: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: float


class _TimeResult:
    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0


@contextmanager
def time_block() -> Generator[_TimeResult, None, None]:
    result = _TimeResult()
    start = time.monotonic()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.monotonic() - start) * 1000
