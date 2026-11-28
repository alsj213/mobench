"""artifacts_hash 行为测试：验证二进制与同目录动态库哈希锚定。"""
from mobench.artifacts_hash import artifact_set_md5


def test_artifact_set_md5_deterministic_for_single_binary(tmp_path):
    binary = tmp_path / "llama-bench"
    binary.write_bytes(b"ELF-BINARY-v1")

    h1 = artifact_set_md5(str(binary))
    h2 = artifact_set_md5(str(binary))
    assert h1 == h2
    assert len(h1) == 32

    binary.write_bytes(b"ELF-BINARY-v2")
    assert artifact_set_md5(str(binary)) != h1


def test_artifact_set_md5_tracks_adjacent_so_changes_and_renames(tmp_path):
    binary = tmp_path / "llama-bench"
    binary.write_bytes(b"ELF-BINARY")
    h_base = artifact_set_md5(str(binary))

    so_file = tmp_path / "libggml.so"
    so_file.write_bytes(b"SO-v1")
    h_with_so = artifact_set_md5(str(binary))
    assert h_with_so != h_base

    so_file.write_bytes(b"SO-v2")
    h_modified_so = artifact_set_md5(str(binary))
    assert h_modified_so != h_with_so

    renamed = tmp_path / "libggml_alt.so"
    so_file.rename(renamed)
    assert artifact_set_md5(str(binary)) != h_modified_so


def test_artifact_set_md5_ignores_non_so_siblings(tmp_path):
    binary = tmp_path / "llama-bench"
    binary.write_bytes(b"ELF-BINARY")
    h_base = artifact_set_md5(str(binary))

    (tmp_path / "README.txt").write_text("notes", encoding="utf-8")
    (tmp_path / "libggml.a").write_bytes(b"STATIC")
    assert artifact_set_md5(str(binary)) == h_base
