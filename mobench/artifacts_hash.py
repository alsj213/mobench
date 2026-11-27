"""Artifact-set hashing for binary + shared-library integrity anchoring.

The measured engine binary is often a thin dynamic loader: the kernels and
CPU code actually under test live in shared libraries next to it. Hashing
the executable alone cannot detect kernel changes in adjacent ``*.so`` files.
The artifact-set hash therefore covers the binary plus every ``*.so`` in the
same directory, sorted by filename and hashed by name and content.

Wire format::

    md5( concat( for f in [binary] + sorted(same_dir/*.so):
                     utf8(f.name) + ascii(hex(md5(f))) ) )
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_set_md5(binary_path: str) -> str:
    p = Path(binary_path)
    files = [p] + sorted(p.parent.glob("*.so"), key=lambda f: f.name)
    h = hashlib.md5()
    for f in files:
        h.update(f.name.encode())
        h.update(_file_md5(f).encode())
    return h.hexdigest()
