from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from data_harness.core.serialize import to_jsonable
from data_harness.llm.providers.base import NormalizedResponse
from data_harness.llm.types import Message, ToolAnnotations, ToolResultBlock, ToolSpec


def log_error_turn(
    turn: int,
    system: str,
    messages: list[Message],
    error: str,
    run_file: str,
) -> None:
    """Append a minimal error record to the run JSONL file."""
    system_hash = hashlib.sha256(system.encode()).hexdigest()
    record: dict[str, Any] = {
        "turn": turn,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "system_hash": system_hash,
        "messages": to_jsonable(messages),
        "status": "error",
        "error": error,
    }
    with open(run_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def setup_logger(run_dir: str = "./runs") -> str:
    """Create a timestamped JSONL file and configure loguru. Returns the path."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_file = str(Path(run_dir) / f"{ts}.jsonl")
    # Ensure the file exists
    Path(run_file).touch()
    logger.remove()
    logger.add(
        lambda msg: None, level="INFO"
    )  # suppress default stderr; callers can add sinks
    return run_file


def _annotations_to_dict(ann: ToolAnnotations) -> dict:
    return {
        k: v
        for k, v in {
            "title": ann.title,
            "read_only": ann.read_only,
            "cache_mutating": ann.cache_mutating,
            "destructive": ann.destructive,
            "open_world": ann.open_world,
        }.items()
        if v is not None
    }


def log_turn(
    turn: int,
    system: str,
    messages: list[Message],
    response: NormalizedResponse,
    tool_results: list[ToolResultBlock],
    latency_ms: float,
    run_file: str,
    cache_storage: dict[str, dict[str, str]] | None = None,
    visible_tools: list[str] | None = None,
    tool_error_count: int = 0,
    all_tools: list[ToolSpec] | None = None,
) -> None:
    """Append one JSON line to the run JSONL file."""
    system_hash = hashlib.sha256(system.encode()).hexdigest()

    record: dict[str, Any] = {
        "turn": turn,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "system_hash": system_hash,
        "messages": to_jsonable(messages),
        "response_content": to_jsonable(response.content),
        "stop_reason": response.stop_reason.value,
        "tool_results": to_jsonable(tool_results),
        "tool_error_count": tool_error_count,
        "metrics": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cache_read_tokens": response.cache_read_tokens,
            "cache_write_tokens": response.cache_write_tokens,
            "latency_ms": latency_ms,
        },
    }

    if visible_tools is not None:
        record["visible_tools"] = visible_tools
    if all_tools is not None:
        ann_map = {
            t.name: _annotations_to_dict(t.annotations)
            for t in all_tools
            if t.annotations is not None and _annotations_to_dict(t.annotations)
        }
        if ann_map:
            record["tool_annotations"] = ann_map
    if turn == 1:
        record["system"] = system
    if cache_storage is not None:
        record["cache_storage"] = cache_storage

    with open(run_file, "a") as f:
        f.write(json.dumps(record) + "\n")
