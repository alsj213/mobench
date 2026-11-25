"""Analysis module tests over canonical records (spec: methodology/power/regression)."""
import json
import textwrap

import pytest

from mobench.analysis import power as power_mod
from mobench.analysis import regression as reg_mod
from mobench.analysis import stats as stats_mod
from mobench.analysis.compare_precision import cosine_similarity, precision_status
from mobench.records import RECORD_SCHEMA, append_record, iter_records


def write_records(tmp_path, engine="llamacpp", workload="tinyllama", metric="gen_tok_s",
                  values=(42.0, 43.0, 41.0, 44.0, 42.5), precision="q8_0", threads=4,
                  fname="records.jsonl"):
    src = tmp_path / "raw.log"
    src.write_text("raw", encoding="utf-8")
    rec_dir = tmp_path / fname
    for i, v in enumerate(values, start=1):
        rec = {"schema": RECORD_SCHEMA, "engine": engine, "workload_id": workload,
               "metric": metric, "value": v, "unit": "tok/s", "threads": threads,
               "precision": precision, "round": i, "platform": "host",
               "source_log": str(src)}
        append_record(str(rec_dir), rec)
    return [r for r in iter_records(str(rec_dir))]


# ---- stats --------------------------------------------------------------
def test_stats_build_rows_aggregates(tmp_path):
    # Arrange
    records = write_records(tmp_path, values=(10.0, 10.1, 10.2, 10.3, 10.4))

    # Act
    rows = stats_mod.build_rows(records)

    # Assert
    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 5
    assert row["samples_min"] is True
    assert row["stats"]["mean"] == pytest.approx(10.2)
    assert row["high_variance"] is False
    assert row["workload_id"] == "tinyllama"
    assert row["metric"] == "gen_tok_s"


def test_stats_flags_high_variance(tmp_path):
    # Arrange: CV well over 5%
    records = write_records(tmp_path, values=(1.0, 10.0, 1.5, 9.0, 12.0))

    # Act
    rows = stats_mod.build_rows(records)

    # Assert
    assert rows[0]["high_variance"] is True


def test_stats_insufficient_when_under_5(tmp_path):
    # Arrange
    records = write_records(tmp_path, values=(1.0, 2.0))

    # Act
    rows = stats_mod.build_rows(records)

    # Assert
    assert rows[0]["samples_min"] is False


# ---- regression ---------------------------------------------------------
def test_regression_detects_throughput_regression(tmp_path):
    # Arrange: baseline high, current lower tok/s on >=5 samples each
    base_recs = write_records(tmp_path, values=(50, 51, 49, 50, 50.5), fname="base.jsonl")
    baseline = tmp_path / "baseline.json"
    reg_mod.save_baseline(str(baseline), base_recs)
    cur_recs = write_records(tmp_path, values=(40, 41, 39, 40, 40.5), fname="cur.jsonl")

    # Act
    result = reg_mod.compare(str(baseline), cur_recs)

    # Assert: higher-better metric worsened beyond 5%
    assert result["rows"]
    row = result["rows"][0]
    assert row["worse"] is True
    assert row["flagged"] is True
    assert row["delta_pct"] < -15


def test_regression_no_flag_on_improvement(tmp_path):
    # Arrange: current better (higher) than baseline
    base_recs = write_records(tmp_path, values=(40, 41, 39, 40, 40.5), fname="b2.jsonl")
    baseline = tmp_path / "baseline2.json"
    reg_mod.save_baseline(str(baseline), base_recs)
    cur_recs = write_records(tmp_path, values=(50, 51, 49, 50, 50.5), fname="c2.jsonl")

    # Act
    result = reg_mod.compare(str(baseline), cur_recs)

    # Assert
    assert result["rows"][0]["flagged"] is False
    assert result["rows"][0]["worse"] is False


def _real_shape_rec(prompt_tokens, value, source_log, metric="prompt_eval_ms_tok"):
    """A record shaped like real mobench output: params make group_key a
    nested tuple, which the baseline JSON round-trip must preserve."""
    return {"schema": "mobench/record@1", "engine": "llamacpp",
            "workload_id": "qwen3-0.6b-q4_k_m",
            "metric": metric, "value": value, "unit": "ms/token",
            "threads": 4, "precision": "q4_k_m", "source_log": source_log,
            "params": {"prompt_tokens": prompt_tokens, "gen_tokens": 512}}


def test_regression_roundtrip_with_params_key(tmp_path):
    # Arrange: real-shaped records with params (nested-tuple group key)
    log = tmp_path / "raw.log"
    log.write_text("x", encoding="utf-8")
    base = [_real_shape_rec(512, v, str(log)) for v in (5.4, 5.41, 5.39, 5.40, 5.42)]
    baseline = tmp_path / "real.json"
    reg_mod.save_baseline(str(baseline), base)
    cur = [_real_shape_rec(512, v, str(log)) for v in (4.5, 4.51, 4.49, 4.50, 4.52)]

    # Act: must not raise (nested list keys were previously unhashable)
    result = reg_mod.compare(str(baseline), cur)

    # Assert
    assert len(result["rows"]) == 1
    assert result["rows"][0]["delta_pct"] < -15


def test_regression_params_distinguish_groups(tmp_path):
    # Different params -> different folded keys -> no overlapping groups
    log = tmp_path / "raw.log"
    log.write_text("x", encoding="utf-8")
    base = [_real_shape_rec(512, v, str(log)) for v in (5.4, 5.41, 5.39, 5.40, 5.42)]
    baseline = tmp_path / "params.json"
    reg_mod.save_baseline(str(baseline), base)
    cur = [_real_shape_rec(1024, v, str(log)) for v in (4.5, 4.51, 4.49, 4.50, 4.52)]
    result = reg_mod.compare(str(baseline), cur)
    assert result["rows"] == []
    assert "note" in result


def test_regression_cli_compare_real_shape(tmp_path, capsys):
    # Full CLI path (render_report) must handle params-folded variable-length keys.
    from mobench.cli import main as cli_main
    log = tmp_path / "raw.log"
    log.write_text("x", encoding="utf-8")
    base = tmp_path / "base.jsonl"
    cur = tmp_path / "cur.jsonl"
    for path, vals in ((base, (5.4, 5.41, 5.39, 5.40, 5.42)),
                       (cur, (4.5, 4.51, 4.49, 4.50, 4.52))):
        path.write_text(
            "\n".join(json.dumps(_real_shape_rec(512, v, str(log))) for v in vals) + "\n",
            encoding="utf-8")
    baseline = tmp_path / "cli-base.json"
    assert cli_main(["regression", "save", "--results", str(base),
                     "--out", str(baseline)]) == 0
    code = cli_main(["regression", "compare", "--results", str(cur),
                     "--baseline", str(baseline)])
    out = capsys.readouterr().out
    assert code == 0
    assert "回归对比结果" in out
    assert "params" in out


# ---- power ------------------------------------------------------------
def test_power_ignores_capacity_as_power():
    # Arrange: batterystats prints Capacity but Computed drain 0
    text = "Capacity: 5000, Computed drain: 0"

    # Act
    result = power_mod.report(text, source="batterystats", inference_count=100)

    # Assert: drain 0 is NOT misread as 5000 mAh -> unavailable, not huge joules
    assert result["status"] == "unavailable"
    assert "unavailable" in result["label"]


def test_power_computes_energy_per_inference():
    # Arrange
    text = "Capacity: 5000\nComputed drain: 100 mAh"

    # Act
    result = power_mod.report(text, source="batterystats", inference_count=100,
                              throughput=54.0)

    # Assert
    assert result["status"] == "ok"
    assert result["energy_joules"] == pytest.approx(360.0)  # 100 mAh * 3.6
    assert result["energy_per_inference_j"] == pytest.approx(3.6)
    assert result["perf_per_watt"] == pytest.approx(54.0 / 360.0)


def test_power_rapl():
    # Arrange
    text = "RAPL package energy 123.4 J"

    # Act
    result = power_mod.report(text, source="rapl", inference_count=10)

    # Assert
    assert result["energy_joules"] == pytest.approx(123.4)


def test_power_wrapped_numbers(tmp_path):
    # Arrange: number wrapped across lines (old parsing bug regression)
    text = "Computed drain:\n  42 mAh"

    # Act
    rep = power_mod.report(text, source="batterystats", inference_count=42)

    # Assert
    assert rep["status"] == "ok"
    assert rep["energy_joules"] == pytest.approx(42 * 3.6)


# ---- precision ---------------------------------------------------------
def test_precision_status_unavailable_when_no_ref():
    # Act
    result = precision_status(ref_available=False)

    # Assert
    assert result["status"] == "unavailable"
    assert "unavailable" in result["label"]
    assert result["cosine"] is None


def test_precision_cosine_matches():
    # Arrange
    ref = [1.0, 0.0, 0.0]
    cand = [1.0, 0.0, 0.0]

    # Act
    result = precision_status(ref_available=True, ref=ref, cand=cand)

    # Assert
    assert result["cosine"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(0.0)
