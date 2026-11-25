"""Unit tests for mobench/records.py (benchmark-core 规范化测量记录)."""
import json

import pytest

from mobench.records import (RECORD_SCHEMA, RecordError, append_record,
                             group_key, iter_records, normalize_record,
                             read_paths, validate_record)


def make_rec(tmp_path, source_log=None, **over):
    src = source_log or tmp_path / "raw.log"
    if isinstance(src, str):
        pass
    else:
        src = str(src)
    rec = {
        "schema": RECORD_SCHEMA,
        "engine": "llamacpp",
        "workload_id": "tinyllama",
        "metric": "gen_tok_s",
        "value": 42.7,
        "unit": "tok/s",
        "source_log": src,
    }
    rec.update(over)
    return rec


def test_append_and_iter_roundtrip(tmp_path):
    # Arrange
    src = tmp_path / "raw.log"
    src.write_text("llama-bench output\n", encoding="utf-8")
    rec_file = tmp_path / "records.jsonl"
    rec = make_rec(tmp_path, src=str(src), threads=4, round=1)

    # Act
    append_record(str(rec_file), rec)

    # Assert
    out = list(iter_records(str(rec_file)))
    assert len(out) == 1
    assert out[0]["engine"] == "llamacpp"
    assert out[0]["metric"] == "gen_tok_s"
    assert out[0]["value"] == 42.7


def test_optional_fields_normalized(tmp_path):
    # Arrange
    src = tmp_path / "raw.log"
    src.write_text("x", encoding="utf-8")
    rec_file = tmp_path / "records.jsonl"
    rec = make_rec(tmp_path, src=str(src))

    # Act
    append_record(str(rec_file), rec)
    out = list(iter_records(str(rec_file)))

    # Assert
    assert out[0]["threads"] is None
    assert out[0]["gpu"] is False
    assert out[0]["round"] is None


def test_reject_missing_source_log(tmp_path):
    # Arrange
    rec = make_rec(tmp_path, source_log="", value=1.0)
    # value present; source_log missing -> required empty
    del rec["source_log"]

    # Act / Assert
    with pytest.raises(RecordError, match="source_log"):
        append_record(str(tmp_path / "r.jsonl"), rec)


def test_reject_nonexistent_source_log(tmp_path):
    # Arrange
    rec = make_rec(tmp_path, source_log=str(tmp_path / "nope.log"), value=1.0)

    # Act / Assert
    with pytest.raises(RecordError, match="source_log"):
        append_record(str(tmp_path / "r.jsonl"), rec)


def test_reject_non_numeric_value(tmp_path):
    # Arrange
    src = tmp_path / "raw.log"
    src.write_text("x", encoding="utf-8")
    rec = make_rec(tmp_path, src=str(src), value="fast")

    # Act / Assert
    with pytest.raises(RecordError, match="value"):
        append_record(str(tmp_path / "r.jsonl"), rec)


def test_reject_wrong_schema(tmp_path):
    # Arrange
    src = tmp_path / "raw.log"
    src.write_text("x", encoding="utf-8")
    rec = make_rec(tmp_path, src=str(src), schema="mobench/record@9")

    # Act / Assert
    with pytest.raises(RecordError, match="schema"):
        append_record(str(tmp_path / "r.jsonl"), rec)


def test_validate_record_empty_for_good(tmp_path):
    # Arrange
    src = tmp_path / "raw.log"
    src.write_text("x", encoding="utf-8")
    rec = make_rec(tmp_path, src=str(src))

    # Act / Assert
    assert validate_record(rec) == []


def test_group_key_and_normalize_immutability():
    # Arrange
    rec = {"engine": "mnn", "workload_id": "resnet50", "metric": "latency_ms",
           "threads": 4, "precision": "fp16"}

    # Act
    key = group_key(rec)

    # Assert
    assert key == ("mnn", "resnet50", "latency_ms", 4, "fp16")


def test_read_paths_globs_dir_records(tmp_path):
    # Arrange
    src = tmp_path / "raw.log"
    src.write_text("x", encoding="utf-8")
    rec_dir = tmp_path / "sub"
    rec_dir.mkdir()
    rec = make_rec(tmp_path, src=str(src))
    append_record(str(rec_dir / "records.jsonl"), rec)

    # Act
    got = read_paths([str(rec_dir)])

    # Assert
    assert len(got) == 1


def test_skip_hand_edited_bad_row(tmp_path):
    # Arrange
    src = tmp_path / "raw.log"
    src.write_text("x", encoding="utf-8")
    rec = make_rec(tmp_path, src=str(src))
    rec_file = tmp_path / "records.jsonl"
    append_record(str(rec_file), rec)
    # hand-append a bad row
    with open(rec_file, "a", encoding="utf-8") as fh:
        fh.write('{"schema":' + json.dumps(RECORD_SCHEMA) + ',"engine":"x"}\n')

    # Act
    out = list(iter_records(str(rec_file)))

    # Assert: bad row skipped, good one remains normalized
    assert len(out) == 1
    assert out[0]["engine"] == "llamacpp"


# -- 测量条件纳入分组键（add-cpu-pin-lockfreq） ---------------------------


def _with_condition(rec, cpus, locked):
    rec = dict(rec)
    rec["device"] = {"pin": {"cpus": cpus}, "freq": {"locked": locked}}
    return rec


def test_group_key_separates_pinned_from_unpinned(tmp_path):
    """绑核与未绑核是两个不同测量条件，混合统计即错误。"""
    # Arrange
    pinned = _with_condition(make_rec(tmp_path, threads=4), [4, 5, 6, 7], True)
    unpinned = _with_condition(make_rec(tmp_path, threads=4), [], False)

    # Act / Assert
    assert group_key(pinned) != group_key(unpinned)


def test_group_key_separates_different_pinned_sets(tmp_path):
    # Arrange
    big = _with_condition(make_rec(tmp_path, threads=4), [4, 5, 6, 7], True)
    little = _with_condition(make_rec(tmp_path, threads=4), [0, 1, 2, 3], True)

    # Act / Assert
    assert group_key(big) != group_key(little)


def test_group_key_separates_frequency_lock_state(tmp_path):
    # Arrange
    locked = _with_condition(make_rec(tmp_path, threads=4), [4, 5, 6, 7], True)
    unlocked = _with_condition(make_rec(tmp_path, threads=4), [4, 5, 6, 7], False)

    # Act / Assert
    assert group_key(locked) != group_key(unlocked)


def test_group_key_same_condition_still_groups_together(tmp_path):
    # Arrange
    a = _with_condition(make_rec(tmp_path, threads=4), [4, 5, 6, 7], True)
    b = _with_condition(make_rec(tmp_path, threads=4), [4, 5, 6, 7], True)

    # Act / Assert
    assert group_key(a) == group_key(b)


def test_group_key_unchanged_for_legacy_records(tmp_path):
    """旧记录（无 device，或 device 无 pin/freq）分组行为与变更前一致。"""
    # Arrange
    legacy = make_rec(tmp_path, threads=4)
    with_device = make_rec(tmp_path, threads=4)
    with_device["device"] = {"temp_c": 34.9, "platform": "apollo"}

    # Act / Assert
    assert group_key(legacy) == group_key(with_device)
    assert group_key(legacy) == ("llamacpp", "tinyllama", "gen_tok_s", 4, None)
