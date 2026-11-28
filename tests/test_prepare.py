"""Unit tests for model preparation (spec: benchmark-model-prep)."""
import io

import pytest

import mobench.prepare as prepare_mod
from mobench.prepare import PrepareError, prepare_model


def model_cfg(tmp_path, **over):
    base = {"engine": "llamacpp", "build": {"source_dir": str(tmp_path)}}
    base.update(over)
    return base


def test_self_provided_model_ready(tmp_path):
    # Arrange
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF" * 16)
    cfg = model_cfg(tmp_path)
    model = {"id": "m", "path": str(path), "kind": "llm"}

    # Act
    info = prepare_model(cfg, model)

    # Assert
    assert info["status"] == "ready"
    assert info["path"] == str(path)
    assert len(info["sha256"]) == 64


def test_missing_self_provided_raises(tmp_path):
    # Arrange
    cfg = model_cfg(tmp_path)
    model = {"id": "m", "path": str(tmp_path / "nope.gguf")}

    # Act / Assert
    with pytest.raises(PrepareError, match="missing"):
        prepare_model(cfg, model)


def test_download_ok_and_sha_verified(tmp_path, monkeypatch):
    # Arrange
    dest = tmp_path / "models" / "tiny.gguf"
    data = b"downloaded-content"
    expected = __import__("hashlib").sha256(data).hexdigest()
    seen_timeouts = []

    def fake_urlopen(_url, timeout=None):
        seen_timeouts.append(timeout)
        return io.BytesIO(data)

    monkeypatch.setattr(prepare_mod.urllib.request, "urlopen", fake_urlopen)
    cfg = model_cfg(tmp_path)
    model = {"id": "m", "path": str(dest), "source": {"url": "https://x/y.gguf",
                                                       "sha256": expected}}

    # Act
    info = prepare_model(cfg, model)

    # Assert
    assert info["status"] == "ready"
    assert info["sha256"] == expected
    assert seen_timeouts == [prepare_mod.DEFAULT_DOWNLOAD_TIMEOUT]


def test_download_sha_mismatch_discards(tmp_path, monkeypatch):
    # Arrange
    dest = tmp_path / "models" / "bad.gguf"

    def fake_urlopen(_url, timeout=None):
        return io.BytesIO(b"tampered")

    monkeypatch.setattr(prepare_mod.urllib.request, "urlopen", fake_urlopen)
    cfg = model_cfg(tmp_path)
    model = {"id": "m", "path": str(dest),
             "source": {"url": "https://x/bad.gguf",
                        "sha256": "0" * 64}}

    # Act / Assert
    with pytest.raises(PrepareError, match="sha256 mismatch"):
        prepare_model(cfg, model)
    assert not dest.exists()  # artifact discarded


def test_download_rejects_non_http_scheme(tmp_path):
    dest = tmp_path / "models" / "local.gguf"
    cfg = model_cfg(tmp_path)
    model = {"id": "m", "path": str(dest), "source": {"url": "file:///etc/passwd"}}

    with pytest.raises(PrepareError, match="unsupported URL scheme"):
        prepare_model(cfg, model)
    assert not dest.exists()


def test_download_error_cleans_up_part_file(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "partial.gguf"

    class BrokenStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _n):
            raise OSError("network drop")

    monkeypatch.setattr(prepare_mod.urllib.request, "urlopen", lambda _u, timeout=None: BrokenStream())

    with pytest.raises(OSError, match="network drop"):
        prepare_mod.download("https://x/partial.gguf", str(dest))
    assert not dest.exists()
    assert not (tmp_path / "models" / "partial.gguf.part").exists()


def test_convert_llama_cpp_uses_engine_tool(tmp_path):
    # Arrange: engine repo with convert_hf_to_gguf.py
    tool = tmp_path / "convert_hf_to_gguf.py"
    tool.write_text(
        "import sys, pathlib\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('-o') + 1])\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_bytes(b'GGUF' * 8)\n",
        encoding="utf-8",
    )
    cfg = model_cfg(tmp_path)
    out = tmp_path / "models" / "converted.gguf"
    model = {"id": "m", "path": str(out), "kind": "llm",
             "source": {"convert_with": "llama.cpp",
                        "dir": str(tmp_path / "hf"),
                        "outtype": "q4_k_m"}}

    # Act
    info = prepare_model(cfg, model)

    # Assert
    assert info["status"] == "ready"
    assert out.exists()
    assert "convert_hf_to_gguf.py" in info["source_cmd"]


def test_convert_missing_tool_raises(tmp_path):
    # Arrange
    cfg = model_cfg(tmp_path)  # no MNNConvert anywhere
    model = {"id": "m", "path": str(tmp_path / "out.mnn"),
             "source": {"convert_with": "mnn", "dir": "x.onnx"}}

    # Act / Assert
    with pytest.raises(PrepareError, match="MNNConvert"):
        prepare_model(cfg, model)


def test_unsupported_convert_with_raises(tmp_path):
    # Arrange
    cfg = model_cfg(tmp_path)
    model = {"id": "m", "path": str(tmp_path / "o"),
             "source": {"convert_with": "tvm"}}

    # Act / Assert
    with pytest.raises(PrepareError, match="convert_with"):
        prepare_model(cfg, model)


def test_convert_delegates_to_adapter_and_wraps_subprocess_error(tmp_path):
    tool = tmp_path / "convert_hf_to_gguf.py"
    tool.write_text("import sys; sys.stderr.write('convert boom'); sys.exit(2)\n", encoding="utf-8")
    cfg = model_cfg(tmp_path)
    model = {
        "id": "m",
        "path": str(tmp_path / "out.gguf"),
        "source": {"convert_with": "llama.cpp", "dir": str(tmp_path / "hf")},
    }
    with pytest.raises(PrepareError, match="convert boom"):
        prepare_model(cfg, model)
