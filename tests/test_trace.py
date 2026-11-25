"""Unit tests for atrace CPU-frequency parsing / summary."""
from mobench import trace

SAMPLE = """# tracer: nop
          atrace-1 (-----) [000] .... 100.000000: tracing_mark_write: x
   llm_bench-5 (-----) [003] .... 100.100000: cpu_frequency: state=2112000 cpu_id=3
   llm_bench-5 (-----) [003] .... 100.250000: cpu_frequency: cpu_id=3 state=1800000
   llm_bench-5 (-----) [001] .... 100.300000: cpu_frequency: state=960000 cpu_id=1
"""


def test_parse_freq_events_both_field_orders():
    evs = list(trace.parse_freq_events(SAMPLE))
    assert len(evs) == 3
    (t0, cpu0, hz0), (t1, cpu1, hz1), (t2, cpu2, hz2) = evs
    assert (round(t0, 3), cpu0, hz0) == (0.0, 3, 2112000)
    assert (cpu1, hz1) == (3, 1800000)      # cpu_id-first order
    assert (cpu2, hz2) == (1, 960000)
    assert t2 > t1 > t0


def test_summarize_low_share_and_mean():
    per = {0: [(0, 2000000), (1, 1800000), (2, 500000), (3, 600000)]}
    s = trace.summarize_freq(per)
    assert s["cpu"][0]["mean_mhz"] == (2_000_000 + 1_800_000 + 500_000 + 600_000) // 4 // 1000
    assert s["cpu"][0]["low_freq_share"] == 0.5   # 2 of 4 below 60% of max
    assert s["overall_max_mhz"] == 2000
