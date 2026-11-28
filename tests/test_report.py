"""Report generation tests (benchmark-core 执行流水线与报告: 高温打标)."""
import json

import pytest

from mobench.records import RECORD_SCHEMA
from mobench.report import render_html, render_markdown, summarize_rows, throttle_risk


def rec(tmp_path, value=42.7, temp_c=None):
    src = tmp_path / "raw.log"
    src.write_text("raw", encoding="utf-8")
    r = {
        "schema": RECORD_SCHEMA,
        "engine": "llamacpp",
        "workload_id": "tinyllama",
        "metric": "gen_tok_s",
        "value": value,
        "unit": "tok/s",
        "threads": 4,
        "precision": "q8_0",
        "round": 1,
        "platform": "host",
        "source_log": str(src),
    }
    if temp_c is not None:
        r["device"] = {"temp_c": temp_c, "platform": "host"}
    return r


def test_markdown_highlights_throttle_risk(tmp_path):
    # Arrange
    records = [rec(tmp_path, temp_c=47.5), rec(tmp_path, value=43.0, temp_c=47.5)]

    # Act
    md = render_markdown(summarize_rows(records), records)

    # Assert
    assert "降频风险" in md
    assert "45" in md


# -- 测量条件标注（add-cpu-pin-lockfreq） ---------------------------------


def conditioned(tmp_path, cpus, locked, value=42.7):
    r = rec(tmp_path, value=value)
    r["device"] = {"temp_c": 34.9, "platform": "apollo",
                   "pin": {"cpus": cpus, "mask": "0xf0", "origin": "auto"},
                   "freq": {"locked": locked, "governor": "performance"}}
    return r


def test_condition_label_formats_cpu_set():
    # Act / Assert
    from mobench.report import condition_label

    assert condition_label({"pin": {"cpus": [4, 5, 6, 7]},
                            "freq": {"locked": True}}) == "cpu4-7·freq锁"
    assert condition_label({"pin": {"cpus": [4, 5, 7]},
                            "freq": {"locked": True}}) == "cpu4,5,7·freq锁"
    assert condition_label({"pin": {"cpus": [7]},
                            "freq": {"locked": False}}) == "cpu7·freq未锁"


def test_condition_label_marks_unpinned_and_legacy():
    # Act / Assert
    from mobench.report import condition_label

    assert "未绑核" in condition_label({"pin": {"cpus": [], "origin": "unavailable"},
                                       "freq": {"locked": False}})
    assert "未绑核" in condition_label({"pin": {"cpus": [], "origin": "disabled"},
                                       "freq": {}})
    # 旧记录：device 无控制块
    assert condition_label({"temp_c": 30.0}) == "-"
    assert condition_label(None) == "-"


def test_markdown_states_measurement_condition(tmp_path):
    # Arrange
    records = [conditioned(tmp_path, [4, 5, 6, 7], True)]

    # Act
    md = render_markdown(summarize_rows(records), records)

    # Assert
    assert "cpu4-7·freq锁" in md


def test_markdown_marks_unpinned_group(tmp_path):
    # Arrange
    r = rec(tmp_path)
    r["device"] = {"pin": {"cpus": [], "origin": "unavailable"},
                   "freq": {"locked": False}}

    # Act
    md = render_markdown(summarize_rows([r]), [r])

    # Assert
    assert "未绑核" in md


def test_html_states_measurement_condition(tmp_path):
    # Arrange
    records = [conditioned(tmp_path, [4, 5, 6, 7], False)]

    # Act
    html = render_html(summarize_rows(records), records)

    # Assert
    assert "cpu4-7" in html
    assert "freq未锁" in html


def test_pinned_and_unpinned_render_as_separate_rows(tmp_path):
    """条件不同 -> 两行，不能合并成一行。"""
    # Arrange
    records = [conditioned(tmp_path, [4, 5, 6, 7], True, value=40.0),
               conditioned(tmp_path, [], False, value=20.0)]

    # Act
    rows = summarize_rows(records)

    # Assert
    assert len(rows) == 2
    labels = sorted(r["condition"] for r in rows)
    assert any("cpu4-7" in x for x in labels)
    assert any("未绑核" in x for x in labels)


def test_normal_temp_no_throttle_banner(tmp_path):
    # Arrange
    records = [rec(tmp_path, temp_c=38.0)]

    # Act
    md = render_markdown(summarize_rows(records), records)

    # Assert
    assert "降频风险" not in md


def test_html_reports_rows_and_escape(tmp_path):
    # Arrange
    normal = rec(tmp_path, temp_c=40.0)
    xss = rec(tmp_path, temp_c=40.0)
    xss["workload_id"] = "<script>alert(1)</script>"
    records = [normal, xss]

    # Act
    html = render_html(summarize_rows(records), records)

    # Assert
    assert "<table>" in html
    assert "tinyllama" in html
    assert "gen_tok_s" in html
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_summarize_aggregates_rounds(tmp_path):
    # Arrange
    records = [rec(tmp_path, value=10.0, temp_c=36.0),
               rec(tmp_path, value=12.0, temp_c=36.0)]

    # Act
    rows = summarize_rows(records)

    # Assert
    assert len(rows) == 1
    assert rows[0]["n"] == 2
    assert rows[0]["mean"] == pytest.approx(11.0)


def test_summarize_rows_zero_mean_and_sample_stdev(tmp_path):
    import statistics

    zero_recs = [rec(tmp_path, value=0.0), rec(tmp_path, value=0.0)]
    zrows = summarize_rows(zero_recs)
    assert zrows[0]["cv_pct"] == 0.0

    vals = [10.0, 12.0, 14.0]
    recs = [rec(tmp_path, value=v) for v in vals]
    srows = summarize_rows(recs)
    expected_cv = round(statistics.stdev(vals) / statistics.fmean(vals) * 100, 2)
    assert srows[0]["cv_pct"] == pytest.approx(expected_cv)
