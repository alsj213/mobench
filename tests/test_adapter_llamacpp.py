"""Unit tests for the llama.cpp adapter (spec: benchmark-core 内置 llama.cpp 适配器)."""
import json

import pytest

from mobench.adapters.base import AdapterError, get_adapter
from mobench.adapters.llamacpp import LlamacppAdapter


def make_cfg(tmp_path):
    return {
        "engine": "llamacpp",
        "build": {"bin_dir": str(tmp_path), "bench_bin": "llama-bench"},
        "workload": {"prompt_tokens": 64, "gen_tokens": 128},
    }


def fixture_log(tmp_path, results=None):
    results = results or [
        {"n_prompt": 64, "n_gen": 128, "t_prompt_processing": 3.2,
         "t_generation": 21.0},
        {"n_prompt": 64, "n_gen": 128, "t_prompt_processing": 3.1,
         "t_generation": 20.4},
    ]
    log = tmp_path / "raw.log"
    log.write_text(json.dumps({"build": "abc", "results": results}), encoding="utf-8")
    return str(log)


def test_registry_lookup_and_resolve(tmp_path):
    # Arrange
    (tmp_path / "llama-bench").write_text("#!/bin/sh\n", encoding="utf-8")
    cfg = make_cfg(tmp_path)

    # Act
    adapter = get_adapter(cfg)

    # Assert
    assert isinstance(adapter, LlamacppAdapter)
    assert adapter.engine == "llamacpp"
    assert "llama-bench" in adapter.resolve_binary()


def test_missing_binary_raises_adapter_error(tmp_path):
    # Arrange
    cfg = make_cfg(tmp_path)  # llama-bench not created

    # Act / Assert
    adapter = LlamacppAdapter(cfg)
    with pytest.raises(AdapterError, match="llama-bench"):
        adapter.resolve_binary()


def test_build_cmds_default_to_o_json_and_legacy_override(tmp_path):
    # Arrange
    (tmp_path / "llama-bench").write_text("#!/bin/sh\n", encoding="utf-8")
    model = {"id": "tinyllama", "path": "/models/tinyllama.gguf"}

    # Act: current llama.cpp default `-o json`
    cmd = LlamacppAdapter(make_cfg(tmp_path)).build_cmds(model, threads=4, rounds=3)[0]

    # Assert
    assert "-m" in cmd and "/models/tinyllama.gguf" in cmd
    assert "-t" in cmd and "4" in cmd
    assert cmd[cmd.index("-m") - 1:].count("-o") >= 0  # flag form sanity below
    assert "-o" in cmd and "json" in cmd
    assert "-p" in cmd and "64" in cmd  # workload prompt_tokens applied

    # Act: legacy engine tree forces `--json` via config
    cfg = make_cfg(tmp_path)
    cfg["llamacpp"] = {"output": "--json"}
    legacy_cmd = LlamacppAdapter(cfg).build_cmds(model, threads=4, rounds=3)[0]

    # Assert
    assert "--json" in legacy_cmd and "-o" not in legacy_cmd


def test_parse_maps_timings_to_records(tmp_path):
    # Arrange
    cfg = make_cfg(tmp_path)
    adapter = LlamacppAdapter(cfg)
    model = {"id": "tinyllama", "path": "/m.gguf", "precision": "q8_0"}
    log = fixture_log(tmp_path)

    # Act
    records = adapter.parse(log, model, threads=4, round_no=1)

    # Assert (two results -> each metric appears twice, one value per result)
    assert len(records) == 2 * 3
    vals = {m: [r["value"] for r in records if r["metric"] == m] for m in
            ("prompt_eval_ms_tok", "gen_ms_tok", "gen_tok_s")}
    assert sorted(vals["prompt_eval_ms_tok"]) == pytest.approx([3.1, 3.2])
    assert sorted(vals["gen_ms_tok"]) == pytest.approx([20.4, 21.0])
    assert sorted(vals["gen_tok_s"]) == pytest.approx([1000.0 / 21.0, 1000.0 / 20.4])
    assert all(r["round"] == 1 and r["threads"] == 4
               and r["precision"] == "q8_0" and r["engine"] == "llamacpp"
               for r in records)


def test_parse_unparseable_raises(tmp_path):
    # Arrange
    cfg = make_cfg(tmp_path)
    adapter = LlamacppAdapter(cfg)
    log = tmp_path / "raw.log"
    log.write_text("hello not json at all\n", encoding="utf-8")
    model = {"id": "tinyllama", "path": "/m.gguf"}

    # Act / Assert
    with pytest.raises(AdapterError, match="cannot parse"):
        adapter.parse(str(log), model, threads=4, round_no=1)


def test_parse_json_embedded_in_text(tmp_path):
    # Arrange
    cfg = make_cfg(tmp_path)
    adapter = LlamacppAdapter(cfg)
    payload = json.dumps({"results": [{"t_prompt_processing": 1.0, "t_generation": 2.0}]})
    log = tmp_path / "raw.log"
    log.write_text("=== llama-bench ===\n" + payload + "\n=== end ===\n", encoding="utf-8")
    model = {"id": "m", "path": "/m.gguf"}

    # Act
    records = adapter.parse(str(log), model, threads=1, round_no=2)

    # Assert
    assert len(records) == 3
    assert all(r["round"] == 2 for r in records)


def test_parse_current_llama_bench_array(tmp_path):
    # Arrange: current llama.cpp `-o json` = top-level array, one test per item,
    # throughput in avg_ts (tok/s). pp item: n_gen 0; tg item: n_gen > 0.
    cfg = make_cfg(tmp_path)
    adapter = LlamacppAdapter(cfg)
    payload = [
        {"n_prompt": 64, "n_gen": 0, "avg_ts": 320.0, "avg_ns": 1e9},
        {"n_prompt": 0, "n_gen": 128, "avg_ts": 40.0, "avg_ns": 2e9},
    ]
    log = tmp_path / "array.log"
    log.write_text(json.dumps(payload), encoding="utf-8")
    model = {"id": "tinyllama", "path": "/m.gguf", "precision": "q8_0"}

    # Act
    records = adapter.parse(str(log), model, threads=8, round_no=1)

    # Assert: pp -> prompt_eval_ms_tok (1000/320), tg -> gen_tok_s (40) + gen_ms_tok (25)
    by = {r["metric"]: r for r in records}
    assert by["prompt_eval_ms_tok"]["value"] == pytest.approx(1000.0 / 320.0)
    assert by["gen_tok_s"]["value"] == pytest.approx(40.0)
    assert by["gen_ms_tok"]["value"] == pytest.approx(25.0)
    assert all(r["engine"] == "llamacpp" and r["round"] == 1
               and r["threads"] == 8 and r["precision"] == "q8_0"
               for r in records)


def test_llamacpp_backend_from_config(tmp_path):
    # Arrange
    (tmp_path / "llama-bench").write_text("x", encoding="utf-8")
    cfg = make_cfg(tmp_path)
    cfg["llamacpp"] = {"backend": "metal"}
    adapter = LlamacppAdapter(cfg)
    log = tmp_path / "raw.log"
    log.write_text(json.dumps({"results": [{"t_generation": 20.0}]}), encoding="utf-8")

    # parsed records carry backend=metal
    recs = adapter.parse(str(log), {"id": "m", "path": "/m.gguf"}, threads=4, round_no=1)
    assert recs[0]["backend"] == "metal"


# -- decode/KV controls (KV 量化波 Round 0, 2026-10-10) ---------------------

def test_build_cmds_kv_decode_controls(tmp_path):
    # Arrange: decode-at-depth + quantized KV + explicit FA
    (tmp_path / "llama-bench").write_text("#!/bin/sh\n", encoding="utf-8")
    cfg = make_cfg(tmp_path)
    cfg["workload"].update({
        "n_depth": 4096, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        "flash_attn": "on",
    })
    model = {"id": "m", "path": "/m.gguf"}

    # Act
    cmd = LlamacppAdapter(cfg).build_cmds(model, threads=4, rounds=1)[0]

    # Assert: each flag carries its value
    for flag, value in (("-d", "4096"), ("-ctk", "q8_0"),
                        ("-ctv", "q8_0"), ("-fa", "on")):
        assert flag in cmd and cmd[cmd.index(flag) + 1] == value


def test_build_cmds_without_kv_controls_is_legacy_shape(tmp_path):
    # Regression: configs without the new keys keep the exact legacy command.
    (tmp_path / "llama-bench").write_text("#!/bin/sh\n", encoding="utf-8")
    cmd = LlamacppAdapter(make_cfg(tmp_path)).build_cmds(
        {"id": "m", "path": "/m.gguf"}, threads=4, rounds=1)[0]

    assert not any(f in cmd for f in ("-d", "-ctk", "-ctv", "-fa"))


def test_parse_carries_matrix_cell_identity(tmp_path):
    # Arrange: one tg item at d4096 with q8_0 KV — record must self-describe.
    cfg = make_cfg(tmp_path)
    cfg["workload"].update({"n_depth": 4096, "cache_type_k": "q8_0",
                            "cache_type_v": "q8_0"})
    adapter = LlamacppAdapter(cfg)
    payload = [{"n_prompt": 0, "n_gen": 128, "n_depth": 4096,
                "type_k": "q8_0", "type_v": "q8_0", "avg_ts": 30.0}]
    log = tmp_path / "cell.log"
    log.write_text(json.dumps(payload), encoding="utf-8")

    # Act
    recs = adapter.parse(str(log), {"id": "m", "path": "/m.gguf"}, threads=4, round_no=1)

    # Assert
    by = {r["metric"]: r for r in recs}
    assert by["gen_tok_s"]["params"]["n_depth"] == 4096
    assert by["gen_tok_s"]["params"]["type_k"] == "q8_0"
    assert by["gen_tok_s"]["params"]["type_v"] == "q8_0"


def test_parse_config_echo_fills_fields_item_omits(tmp_path):
    # Arrange: item omits type_v (engine only prints non-default) — the
    # config-declared cell identity must still be recorded.
    cfg = make_cfg(tmp_path)
    cfg["workload"].update({"n_depth": 512, "cache_type_k": "f16",
                            "cache_type_v": "f16"})
    adapter = LlamacppAdapter(cfg)
    payload = [{"n_prompt": 0, "n_gen": 128, "avg_ts": 40.0}]
    log = tmp_path / "cell.log"
    log.write_text(json.dumps(payload), encoding="utf-8")

    # Act
    recs = adapter.parse(str(log), {"id": "m", "path": "/m.gguf"}, threads=4, round_no=1)

    # Assert
    by = {r["metric"]: r for r in recs}
    assert by["gen_tok_s"]["params"]["n_depth"] == 512
    assert by["gen_tok_s"]["params"]["type_k"] == "f16"
    assert by["gen_tok_s"]["params"]["type_v"] == "f16"


def test_parse_legacy_params_unchanged_without_controls(tmp_path):
    # Regression: no cell keys in config -> params stay prompt/gen only.
    cfg = make_cfg(tmp_path)
    adapter = LlamacppAdapter(cfg)
    recs = adapter.parse(fixture_log(tmp_path), {"id": "m", "path": "/m.gguf"},
                         threads=4, round_no=1)
    assert recs[0]["params"] == {"prompt_tokens": 64, "gen_tokens": 128}


def test_parse_single_element_array_with_leading_log_lines(tmp_path):
    # Regression: leading [INFO]/{debug} lines + single-element top-level array
    # must parse as list[dict], not get mis-sliced into inner dict.
    cfg = make_cfg(tmp_path)
    adapter = LlamacppAdapter(cfg)
    log = tmp_path / "noisy_array.log"
    log.write_text(
        "[INFO] load_backend: loaded CPU backend\n"
        "{debug} init ok\n"
        + json.dumps([{"n_prompt": 0, "n_gen": 128, "avg_ts": 42.5}])
        + "\n[INFO] done\n",
        encoding="utf-8",
    )
    recs = adapter.parse(str(log), {"id": "m", "path": "/m.gguf"}, threads=4, round_no=1)
    by = {r["metric"]: r for r in recs}
    assert by["gen_tok_s"]["value"] == pytest.approx(42.5)
    assert by["gen_ms_tok"]["value"] == pytest.approx(1000.0 / 42.5)


def test_parse_legacy_dict_with_leading_bracket_log_lines(tmp_path):
    cfg = make_cfg(tmp_path)
    adapter = LlamacppAdapter(cfg)
    log = tmp_path / "noisy_dict.log"
    log.write_text(
        "[INFO] starting llama-bench\n"
        + json.dumps({"results": [{"t_generation": 25.0}]})
        + "\n",
        encoding="utf-8",
    )
    recs = adapter.parse(str(log), {"id": "m", "path": "/m.gguf"}, threads=4, round_no=1)
    by = {r["metric"]: r for r in recs}
    assert by["gen_tok_s"]["value"] == pytest.approx(40.0)
