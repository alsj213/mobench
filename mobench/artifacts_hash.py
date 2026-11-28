"""Artifact-set hashing for binary + shared-library integrity anchoring.

The measured engine binary is often a thin dynamic loader: the kernels and
CPU code actually under test live in shared libraries next to it. Hashing
the executable alone cannot detect kernel changes in adjacent ``*.so`` or
versioned ``*.so.*`` files. The artifact-set hash therefore covers the
binary plus every shared library in the same directory, sorted by filename
and hashed by name and content.

Wire format::

    md5( concat( for f in [binary] + sorted(same_dir/*.so*):
                     utf8(f.name) + ascii(hex(md5(f))) ) )
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from mobench.io_utils import file_md5

# 只接受真正的共享库：``*.so`` 或以数字版本结尾的 ``*.so.N[.N...]``。
# 这样 ``libggml.so.bak`` / ``libggml.so.swp`` 等编辑器/备份残留不会污染哈希。
_VERSIONED_SO_RE = re.compile(r"\.so\.\d+(?:\.\d+)*$")


def artifact_set_md5(binary_path: str) -> str:
    p = Path(binary_path)
    shared_libs = sorted(
        (
            f
            for f in p.parent.iterdir()
            if f.is_file()
            and f != p
            and (f.name.endswith(".so") or bool(_VERSIONED_SO_RE.search(f.name)))
        ),
        key=lambda f: f.name,
    )
    files = [p] + shared_libs
    h = hashlib.md5()
    for f in files:
        h.update(f.name.encode())
        h.update(file_md5(f).encode())
    return h.hexdigest()
