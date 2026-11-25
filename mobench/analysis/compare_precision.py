"""Precision comparison against an ONNX Runtime reference (spec:
benchmark-core 内置 MNN 适配器 — precision reference).

Only meaningful when the same graph is expressed as an ONNX model and a local
ORT reference output is available. Otherwise the caller labels
``Precision reference unavailable`` — no cosine is fabricated.
"""
from __future__ import annotations

import math


def cosine_similarity(ref, cand):
    """Cosine similarity between two equal-length float sequences."""
    if len(ref) != len(cand) or not ref:
        return None
    dot = sum(a * b for a, b in zip(ref, cand))
    norm_a = math.sqrt(sum(a * a for a in ref))
    norm_b = math.sqrt(sum(b * b for b in cand))
    if norm_a == 0 or norm_b == 0:
        return None
    return dot / (norm_a * norm_b)


def mean_abs_error(ref, cand):
    if len(ref) != len(cand) or not ref:
        return None
    return sum(abs(a - b) for a, b in zip(ref, cand)) / len(ref)


def max_abs_error(ref, cand):
    if len(ref) != len(cand) or not ref:
        return None
    return max(abs(a - b) for a, b in zip(ref, cand))


def precision_status(ref_available, ref=None, cand=None):
    """Return a comparison dict; never fabricate when the reference is absent."""
    if not ref_available:
        return {"status": "unavailable",
                "label": "Precision reference unavailable",
                "cosine": None, "mae": None, "max_err": None}
    sim = cosine_similarity(ref, cand)
    return {
        "status": "ok" if sim is not None else "unavailable",
        "cosine": round(sim, 6) if sim is not None else None,
        "mae": round(mean_abs_error(ref, cand), 6) if sim is not None else None,
        "max_err": round(max_abs_error(ref, cand), 6) if sim is not None else None,
    }
