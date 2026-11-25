"""Build-recipe fingerprint checks (测量有效性门禁: required_cmake_args)."""
import json

import pytest

from mobench.build_recipe import (BuildRecipeError, find_cmake_cache,
                                  parse_required_args, read_cmake_cache,
                                  verify_build_recipe)
from mobench.config import load_config
from mobench.run import RunError, run_pipeline

SPEC = ("-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 "
        "-DGGML_OPENMP=ON -DGGML_CPU_REPACK=ON")


def test_parse_required_args_basic():
    assert parse_required_args(SPEC) == {
        "GGML_CPU_ARM_ARCH": "armv8.2-a+dotprod+fp16",
        "GGML_OPENMP": "ON",
        "GGML_CPU_REPACK": "ON",
    }


def test_parse_required_args_type_syntax():
    # -DKEY:TYPE=VALUE is accepted; TYPE is dropped.
    assert parse_required_args("-DGGML_OPENMP:BOOL=ON") == {"GGML_OPENMP": "ON"}


def test_parse_required_args_rejects_non_d_token():
    with pytest.raises(BuildRecipeError):
        parse_required_args("--werror -DGGML_OPENMP=ON")
    with pytest.raises(BuildRecipeError):
        parse_required_args("-DGGML_OPENMP")  # no '='


def test_parse_required_args_empty():
    assert parse_required_args("") == {}
    assert parse_required_args(None) == {}


def test_read_cmake_cache_skips_comments_and_blanks(tmp_path):
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        "// a comment\n"
        "# another\n"
        "\n"
        "GGML_CPU_ARM_ARCH:STRING=armv8.2-a+dotprod+fp16\n"
        "GGML_OPENMP:BOOL=ON\n"
        "PLAINLINE=no-colon-skipped\n",
        encoding="utf-8",
    )
    parsed = read_cmake_cache(str(cache))
    assert parsed["GGML_CPU_ARM_ARCH"] == "armv8.2-a+dotprod+fp16"
    assert parsed["GGML_OPENMP"] == "ON"
    assert "PLAINLINE" not in parsed


def make_build_tree(root, cache_lines):
    (root / "build" / "bin").mkdir(parents=True)
    (root / "build" / "CMakeCache.txt").write_text(
        "\n".join(cache_lines) + "\n", encoding="utf-8")
    return str(root / "build" / "bin")


def test_verify_no_declaration_is_noop():
    assert verify_build_recipe({}) == {"checked": False,
                                       "reason": "no required_cmake_args declared"}
    assert verify_build_recipe({"build": {}})["checked"] is False


def test_verify_pass(tmp_path):
    bin_dir = make_build_tree(tmp_path, [
        "GGML_CPU_ARM_ARCH:STRING=armv8.2-a+dotprod+fp16",
        "GGML_OPENMP:BOOL=ON",
        "GGML_CPU_REPACK:BOOL=ON",
    ])
    info = verify_build_recipe({"build": {"bin_dir": bin_dir,
                                          "required_cmake_args": SPEC}})
    assert info["checked"] is True
    assert info["verified"]["GGML_CPU_ARM_ARCH"] == "armv8.2-a+dotprod+fp16"
    assert info["n_checked"] == 3


def test_verify_bool_normalization(tmp_path):
    # Required TRUE, cache ON -> equivalent booleans pass.
    bin_dir = make_build_tree(tmp_path, ["GGML_OPENMP:BOOL=ON"])
    info = verify_build_recipe({"build": {
        "bin_dir": bin_dir, "required_cmake_args": "-DGGML_OPENMP=TRUE"}})
    assert info["checked"] is True


def test_verify_mismatch_raises_with_values(tmp_path):
    # The K8 incident shape: dotprod missing from the actual build.
    bin_dir = make_build_tree(tmp_path, [
        "GGML_CPU_ARM_ARCH:STRING=armv8.2-a+fp16",
        "GGML_OPENMP:BOOL=ON",
    ])
    with pytest.raises(BuildRecipeError) as ei:
        verify_build_recipe({"build": {"bin_dir": bin_dir,
                                       "required_cmake_args": SPEC}})
    msg = str(ei.value)
    assert "GGML_CPU_ARM_ARCH" in msg
    assert "armv8.2-a+dotprod+fp16" in msg  # expected
    assert "armv8.2-a+fp16" in msg          # actual
    assert "GGML_CPU_REPACK" in msg         # missing key also reported


def test_verify_missing_cache_raises(tmp_path):
    (tmp_path / "bin").mkdir()
    with pytest.raises(BuildRecipeError) as ei:
        verify_build_recipe({"build": {"bin_dir": str(tmp_path / "bin"),
                                       "required_cmake_args": SPEC}})
    assert "CMakeCache.txt" in str(ei.value)


def test_verify_no_bin_dir_raises():
    with pytest.raises(BuildRecipeError):
        verify_build_recipe({"build": {"required_cmake_args": SPEC}})


def test_find_cmake_cache_prefers_build_root(tmp_path):
    bin_dir = make_build_tree(tmp_path, ["A:STRING=1"])
    assert find_cmake_cache(bin_dir) == str(tmp_path / "build" / "CMakeCache.txt")


# -- pipeline integration ----------------------------------------------------

def fake_llama_bench(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        "print(json.dumps({'build':'fake','results':[\n"
        "  {'n_prompt':64,'n_gen':128,'t_prompt_processing':3.2,'t_generation':21.0}]}))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def setup_project(tmp_path, required_args=None, cache_lines=()):
    fake_llama_bench(tmp_path / "bin" / "llama-bench")
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "tinyllama.gguf").write_bytes(b"GGUF" * 64)
    req = f'  required_cmake_args: "{required_args}"\n' if required_args else ""
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(
        "engine: llamacpp\n"
        "platform:\n  type: host\n"
        "build:\n  bin_dir: bin\n  bench_bin: llama-bench\n" + req +
        "models:\n  - id: tinyllama\n    path: models/tinyllama.gguf\n"
        "    kind: llm\n    precision: q8_0\n"
        "workload:\n  threads: [4]\n  rounds: 1\n"
        "results_dir: results\n",
        encoding="utf-8",
    )
    if cache_lines:
        (tmp_path / "CMakeCache.txt").write_text(
            "\n".join(cache_lines) + "\n", encoding="utf-8")
    return str(cfg)


def test_pipeline_records_verified_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(
        tmp_path, required_args="-DGGML_OPENMP=ON",
        cache_lines=["GGML_OPENMP:BOOL=ON"])
    meta = run_pipeline(load_config(cfg_path))
    assert meta["build_recipe"]["checked"] is True


def test_pipeline_rejects_recipe_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(
        tmp_path, required_args="-DGGML_CPU_REPACK=ON",
        cache_lines=["GGML_CPU_REPACK:BOOL=OFF"])
    with pytest.raises(RunError) as ei:
        run_pipeline(load_config(cfg_path))
    assert "[step:build recipe]" in str(ei.value)
