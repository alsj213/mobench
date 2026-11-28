"""共享流式哈希、原子 JSON 写入与冲突安全目录生成工具。"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
from typing import Any


def _file_digest(path: str | os.PathLike, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path: str | os.PathLike) -> str:
    """流式计算本地文件 MD5 hex（1 MiB 分块）。"""
    return _file_digest(path, "md5")


def file_sha256(path: str | os.PathLike) -> str:
    """流式计算本地文件 SHA256 hex（1 MiB 分块）。"""
    return _file_digest(path, "sha256")


def atomic_write_json(path: str | os.PathLike, data: Any) -> None:
    """以同目录临时文件 + os.replace 原子落盘 JSON，避免中断留下半截文件。"""
    abs_path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(abs_path)
    os.makedirs(parent, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{os.path.basename(abs_path)}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = fh.name
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
        os.replace(tmp_path, abs_path)
    except BaseException:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def make_run_dir(results_dir: str | os.PathLike, prefix: str) -> tuple[str, str]:
    """创建不冲突的 ``<results_dir>/<prefix>_<YYYYMMDD_HHMMSS>[_NN]`` 目录。"""
    base_dir = os.fspath(results_dir)
    os.makedirs(base_dir, exist_ok=True)
    base_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seq = 0
    while True:
        ts = base_ts if seq == 0 else f"{base_ts}_{seq:02d}"
        path = os.path.join(base_dir, f"{prefix}_{ts}")
        try:
            os.makedirs(path, exist_ok=False)
            return ts, path
        except FileExistsError:
            seq += 1
