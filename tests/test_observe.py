import time

from data_harness.core.observe import TurnMetrics, time_block


def test_turn_metrics_fields():
    m = TurnMetrics(
        turn=1,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=200,
        cache_write_tokens=300,
        latency_ms=42.5,
    )
    assert m.turn == 1
    assert m.input_tokens == 100
    assert m.output_tokens == 50
    assert m.cache_read_tokens == 200
    assert m.cache_write_tokens == 300
    assert m.latency_ms == 42.5


def test_time_block_measures_elapsed():
    with time_block() as tb:
        time.sleep(0.05)
    assert tb.elapsed_ms >= 40
    assert tb.elapsed_ms < 5000  # sanity upper bound


def test_time_block_before_exit_is_none():
    with time_block() as tb:
        pass
    assert tb.elapsed_ms >= 0
