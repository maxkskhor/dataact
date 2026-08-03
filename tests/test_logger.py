import json
from pathlib import Path

import pytest

from data_harness.core.logger import log_turn, setup_logger
from data_harness.llm.providers.base import NormalizedResponse, StopReason
from data_harness.llm.types import Message, TextBlock, ToolResultBlock


def make_response(text="OK"):
    return NormalizedResponse(
        stop_reason=StopReason.END_TURN,
        content=[TextBlock(text=text)],
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )


class TestSetupLogger:
    def test_creates_jsonl_file(self, tmp_path):
        run_file = setup_logger(run_dir=str(tmp_path))
        assert run_file.endswith(".jsonl")
        assert Path(run_file).exists()

    def test_file_in_run_dir(self, tmp_path):
        run_file = setup_logger(run_dir=str(tmp_path))
        assert str(tmp_path) in run_file


class TestLogTurn:
    def _read_lines(self, run_file):
        with open(run_file) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_log_turn_appends_json_line(self, tmp_path):
        run_file = setup_logger(run_dir=str(tmp_path))
        messages = [Message(role="user", content=[TextBlock(text="hi")])]
        resp = make_response()
        log_turn(
            turn=1,
            system="The system prompt.",
            messages=messages,
            response=resp,
            tool_results=[],
            latency_ms=42.0,
            run_file=run_file,
        )
        lines = self._read_lines(run_file)
        assert len(lines) == 1
        line = lines[0]
        assert line["turn"] == 1
        assert "timestamp" in line
        assert "messages" in line
        assert "response_content" in line
        assert "stop_reason" in line
        assert "metrics" in line
        assert line["metrics"]["input_tokens"] == 10
        assert line["metrics"]["output_tokens"] == 5
        assert line["metrics"]["latency_ms"] == 42.0

    def test_turn_1_has_system_and_hash(self, tmp_path):
        run_file = setup_logger(run_dir=str(tmp_path))
        messages = [Message(role="user", content=[TextBlock(text="hi")])]
        resp = make_response()
        log_turn(
            turn=1,
            system="The system prompt.",
            messages=messages,
            response=resp,
            tool_results=[],
            latency_ms=10.0,
            run_file=run_file,
        )
        lines = self._read_lines(run_file)
        assert lines[0]["system"] == "The system prompt."
        assert "system_hash" in lines[0]

    def test_turn_2_has_hash_only(self, tmp_path):
        run_file = setup_logger(run_dir=str(tmp_path))
        messages = [Message(role="user", content=[TextBlock(text="hi")])]
        resp = make_response()
        for turn in [1, 2]:
            log_turn(
                turn=turn,
                system="The system prompt.",
                messages=messages,
                response=resp,
                tool_results=[],
                latency_ms=10.0,
                run_file=run_file,
            )
        lines = self._read_lines(run_file)
        assert "system" not in lines[1]
        assert "system_hash" in lines[1]

    def test_system_hash_identical_across_turns(self, tmp_path):
        run_file = setup_logger(run_dir=str(tmp_path))
        messages = [Message(role="user", content=[TextBlock(text="hi")])]
        resp = make_response()
        for turn in [1, 2, 3]:
            log_turn(
                turn=turn,
                system="The system prompt.",
                messages=messages,
                response=resp,
                tool_results=[],
                latency_ms=10.0,
                run_file=run_file,
            )
        lines = self._read_lines(run_file)
        hashes = [line["system_hash"] for line in lines]
        assert len(set(hashes)) == 1

    def test_dataframe_in_messages_no_crash(self, tmp_path):
        pd = pytest.importorskip("pandas")
        run_file = setup_logger(run_dir=str(tmp_path))
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        # Simulate a message that somehow has a dataframe-like result
        messages = [Message(role="user", content=[TextBlock(text="query")])]
        resp = make_response()
        log_turn(
            turn=1,
            system="sys",
            messages=messages,
            response=resp,
            tool_results=[ToolResultBlock(tool_use_id="x", content=str(df.head(2)))],
            latency_ms=5.0,
            run_file=run_file,
        )
        lines = self._read_lines(run_file)
        assert len(lines) == 1
