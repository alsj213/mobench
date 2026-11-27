"""Unit tests for mobench/config.py (benchmark-core 通用配置与 Schema 校验)."""
import textwrap

import pytest

from mobench.config import (BUILTIN_ENGINES, ConfigError, apply_defaults,
                            load_config, load_yaml, validate)


def write_cfg(tmp_path, body):
    path = tmp_path / "mobench.yml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


VALID = """
engine: llamacpp
platform:
  type: host
models:
  - id: tinyllama
    path: models/tinyllama-q8_0.gguf
    kind: llm
    precision: q8_0
workload:
  threads: [1, 4]
  rounds: 5
"""


def test_load_valid_and_defaults(tmp_path):
    # Arrange
    cfg_path = write_cfg(tmp_path, VALID)

    # Act
    cfg = load_config(cfg_path)

    # Assert
    assert cfg["engine"] == "llamacpp"
    assert cfg["platform"]["type"] == "host"
    assert cfg["workload"]["threads"] == [1, 4]
    assert cfg["results_dir"] == "results"


def test_platform_defaults_to_host_and_workload_defaults(tmp_path):
    # Arrange
    cfg_path = write_cfg(tmp_path, "engine: mnn\nmodels:\n  - id: m\n    path: a.mnn\n")

    # Act
    cfg = load_config(cfg_path)

    # Assert
    assert cfg["platform"]["type"] == "host"
    assert cfg["workload"]["rounds"] == 5
    assert cfg["workload"]["threads"] == [4]


def test_missing_file_raises(tmp_path):
    # Act / Assert
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "nope.yml"))


def test_missing_required_sections_rejected(tmp_path):
    # Arrange
    cfg_path = write_cfg(tmp_path, "engine: llamacpp\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="models"):
        load_config(cfg_path)


def test_unsupported_engine_lists_adapters(tmp_path):
    # Arrange
    cfg = apply_defaults({"engine": "ncnn", "models": [{"id": "m", "path": "x"}]})

    # Act
    problems = validate(cfg)

    # Assert
    assert any("ncnn" in p and "llamacpp" in p for p in problems)


def test_adb_requires_adb_and_device(tmp_path):
    # Arrange
    cfg = apply_defaults({
        "engine": "llamacpp",
        "platform": {"type": "adb"},  # missing adb/device_id
        "models": [{"id": "m", "path": "x.gguf"}],
    })

    # Act
    problems = validate(cfg)

    # Assert
    joined = " ".join(problems)
    assert "adb" in joined
    assert "device_id" in joined


def test_invalid_yaml_rejected(tmp_path):
    # Arrange
    cfg_path = tmp_path / "mobench.yml"
    cfg_path.write_text("engine: [unclosed", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ConfigError, match="YAML"):
        load_yaml(str(cfg_path))


def test_invalid_model_kind(tmp_path):
    # Arrange
    cfg = apply_defaults({
        "engine": "llamacpp",
        "models": [{"id": "m", "path": "x", "kind": "audio"}],
    })

    # Act
    problems = validate(cfg)

    # Assert
    assert any("kind" in p for p in problems)


def test_builtin_engines_exposed():
    # Assert
    assert "llamacpp" in BUILTIN_ENGINES
    assert "mnn" in BUILTIN_ENGINES


# -- CPU 绑核 / 锁频配置（add-cpu-pin-lockfreq） --------------------------


def test_methodology_defaults_injected(tmp_path):
    """未声明 methodology 的既有档位也默认受控（cpu_pin/lock_freq 为真）。"""
    # Arrange
    cfg_path = write_cfg(tmp_path, VALID)

    # Act
    cfg = load_config(cfg_path)

    # Assert
    assert cfg["methodology"]["cpu_pin"] is True
    assert cfg["methodology"]["lock_freq"] is True
    assert cfg["methodology"]["pin_cpus"] is None
    assert cfg["methodology"]["env_control"] == "auto"


def test_methodology_partial_keeps_other_defaults(tmp_path):
    """只声明 env_control 时，绑核/锁频默认值仍被注入。"""
    # Arrange
    body = VALID + "methodology:\n  env_control: none\n"
    cfg_path = write_cfg(tmp_path, body)

    # Act
    cfg = load_config(cfg_path)

    # Assert
    assert cfg["methodology"]["env_control"] == "none"
    assert cfg["methodology"]["cpu_pin"] is True
    assert cfg["methodology"]["lock_freq"] is True


def test_pin_cpus_range_string_parsed():
    # Act
    from mobench.config import parse_cpus

    cpus = parse_cpus("4-7")

    # Assert
    assert cpus == [4, 5, 6, 7]


def test_pin_cpus_list_sorted_and_deduped():
    # Act
    from mobench.config import parse_cpus

    cpus = parse_cpus([7, 4, 4, 5])

    # Assert
    assert cpus == [4, 5, 7]


def test_pin_cpus_csv_string_parsed():
    # Act
    from mobench.config import parse_cpus

    cpus = parse_cpus("4,5,6,7")

    # Assert
    assert cpus == [4, 5, 6, 7]


def test_pin_cpus_invalid_format_rejected(tmp_path):
    # Arrange
    body = VALID + "methodology:\n  pin_cpus: abc\n"
    cfg_path = write_cfg(tmp_path, body)

    # Act / Assert
    with pytest.raises(ConfigError, match="pin_cpus"):
        load_config(cfg_path)


def test_pin_cpus_negative_rejected(tmp_path):
    # Arrange
    body = VALID + "methodology:\n  pin_cpus: [-1]\n"
    cfg_path = write_cfg(tmp_path, body)

    # Act / Assert
    with pytest.raises(ConfigError, match="pin_cpus"):
        load_config(cfg_path)


def test_pin_cpus_reversed_range_rejected(tmp_path):
    # Arrange
    body = VALID + "methodology:\n  pin_cpus: \"7-4\"\n"
    cfg_path = write_cfg(tmp_path, body)

    # Act / Assert
    with pytest.raises(ConfigError, match="pin_cpus"):
        load_config(cfg_path)


def test_methodology_non_bool_switch_rejected(tmp_path):
    # Arrange
    body = VALID + "methodology:\n  cpu_pin: maybe\n"
    cfg_path = write_cfg(tmp_path, body)

    # Act / Assert
    with pytest.raises(ConfigError, match="cpu_pin"):
        load_config(cfg_path)


def test_mnn_n_repeat_accepts_positive_int(tmp_path):
    # Arrange: 一轮一测（T4 档用），llm_bench 侧 -rep 1
    cfg_path = write_cfg(tmp_path, VALID + "mnn:\n  n_repeat: 1\n")

    # Act
    cfg = load_config(cfg_path)

    # Assert
    assert cfg["mnn"]["n_repeat"] == 1


def test_mnn_n_repeat_rejects_non_positive(tmp_path):
    # Arrange
    cfg_path = write_cfg(tmp_path, VALID + "mnn:\n  n_repeat: 0\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="n_repeat"):
        load_config(cfg_path)


def test_mnn_n_repeat_rejects_non_int(tmp_path):
    # Arrange: bool 是 int 的子类，须单独挡掉
    cfg_path = write_cfg(tmp_path, VALID + "mnn:\n  n_repeat: true\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="n_repeat"):
        load_config(cfg_path)


# -- validation 段（add-correctness-validation）-----------------------------

VALIDATION_OK = """
validation:
  prompts:
    - prompts/long-01.txt
  gen_tokens: 64
  ppl:
    dataset: datasets/wiki.test.raw
    ctx: 1024
    chunks: 8
"""


def test_validation_absent_is_backward_compatible(tmp_path):
    # Arrange: 既有档位不含 validation 段
    cfg_path = write_cfg(tmp_path, VALID)

    # Act
    cfg = load_config(cfg_path)

    # Assert: 无校验能力（不注入默认值）
    assert not cfg.get("validation")


def test_validation_section_parsed(tmp_path):
    # Arrange
    cfg_path = write_cfg(tmp_path, VALID + VALIDATION_OK)

    # Act
    cfg = load_config(cfg_path)

    # Assert
    assert cfg["validation"]["gen_tokens"] == 64
    assert cfg["validation"]["ppl"]["ctx"] == 1024


def test_validation_ppl_only_is_allowed(tmp_path):
    # Arrange: 仅 ppl 仪器（decode 覆盖由工件层标 NOT COVERED）
    body = "validation:\n  ppl:\n    dataset: d.raw\n"
    cfg_path = write_cfg(tmp_path, VALID + body)

    # Act
    cfg = load_config(cfg_path)

    # Assert
    assert "prompts" not in cfg["validation"]


def test_validation_empty_section_rejected(tmp_path):
    # Arrange: 声明了段却没有仪器
    cfg_path = write_cfg(tmp_path, VALID + "validation:\n  gen_tokens: 64\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="instrument"):
        load_config(cfg_path)


def test_validation_prompts_empty_list_rejected(tmp_path):
    # Arrange
    cfg_path = write_cfg(tmp_path, VALID + "validation:\n  prompts: []\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="prompts"):
        load_config(cfg_path)


def test_validation_gen_tokens_non_positive_rejected(tmp_path):
    # Arrange
    cfg_path = write_cfg(
        tmp_path, VALID + "validation:\n  prompts: [p.txt]\n  gen_tokens: 0\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="gen_tokens"):
        load_config(cfg_path)


def test_validation_ppl_requires_dataset(tmp_path):
    # Arrange
    cfg_path = write_cfg(tmp_path, VALID + "validation:\n  ppl:\n    ctx: 1024\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="dataset"):
        load_config(cfg_path)


def test_validation_ppl_ctx_invalid_rejected(tmp_path):
    # Arrange
    cfg_path = write_cfg(
        tmp_path,
        VALID + "validation:\n  ppl:\n    dataset: d.raw\n    ctx: -4\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="ctx"):
        load_config(cfg_path)


def test_flash_attn_unquoted_yaml_bool_normalized(tmp_path):
    # Arrange: YAML 1.1 把 off 解析为 False；必须还原为 "off"（否则 -fa 被静默丢弃）
    cfg_path = write_cfg(
        tmp_path, VALID + "workload:\n  threads: [4]\n  flash_attn: off\n")

    # Act
    cfg = load_config(cfg_path)

    # Assert
    assert cfg["workload"]["flash_attn"] == "off"


def test_schema_validates_all_repo_examples_and_templates():
    import glob
    import json
    from pathlib import Path
    import yaml

    jsonschema = pytest.importorskip("jsonschema")
    repo = Path(__file__).resolve().parents[1]
    schema = json.loads((repo / "schema" / "mobench.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft7Validator.check_schema(schema)

    files = sorted(
        list(repo.glob("examples/**/*.yml")) + list((repo / "configs" / "templates").glob("*.yml"))
    )
    assert files
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=schema)
        cfg = load_config(str(path))
        assert cfg["engine"] in BUILTIN_ENGINES
