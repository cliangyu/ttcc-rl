"""Canonical curve parser — single source of truth.

Used by:
- inference post-processing (``scripts/postprocess.py``)
- GRPO/RLOO reward plugin (``go_viral_overlay/.../ttcc_ibs_plugin.py``)
- CoT distillation pipeline (``scripts/cot_distill.py``)

Robust to (a) JSON-fenced ``{"R": [1.0, ...]}``, (b) bare ``R = [...]`` or
``R: [...]``, (c) text with prose before/after, (d) truncated output where
the closing ``]``/``}`` never arrives. Always returns a length ``T+1`` list
of floats with ``R[0] == 1.0`` and the sequence clipped to ``[0, 1]`` and
forced monotone non-increasing via running min from the left.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Sequence

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_R_KEY_RE = re.compile(r'(?:"R(?:\(0\))?"|\bR)\s*[:=]\s*\[')


def _coerce_length(R: list[float], T: int) -> list[float]:
    """Pad-with-last-value or truncate; enforce R[0] = 1, monotone, [0, 1]."""
    if not R:
        raise ValueError("empty R list")
    if len(R) < T + 1:
        R = R + [R[-1]] * (T + 1 - len(R))
    elif len(R) > T + 1:
        R = R[: T + 1]
    R[0] = 1.0
    for i in range(1, len(R)):
        if R[i] > R[i - 1]:
            R[i] = R[i - 1]
        R[i] = max(0.0, min(1.0, R[i]))
    return R


def parse_curve(text: str, T: int) -> Optional[list[float]]:
    """Extract an R curve of length ``T + 1`` from ``text``.

    Returns ``None`` if no recognisable R-list is found. Use callers'
    discretion for how to handle: the reward plugin returns ``0.0`` reward,
    the post-processor drops the row.
    """
    cleaned = text.replace("```json", "").replace("```", "")

    # Pass 1 — balanced JSON object containing "R".
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(cleaned)):
            ch = cleaned[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = cleaned[start : end + 1]
                    try:
                        obj = json.loads(blob)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "R" in obj and isinstance(obj["R"], list):
                        try:
                            nums = [float(x) for x in obj["R"]]
                            return _coerce_length(nums, T)
                        except (TypeError, ValueError):
                            pass
                    break
        start = cleaned.find("{", start + 1)

    # Pass 2 — bare ``R = [...]`` / ``R: [...]`` / ``"R": [...]``.
    m = _R_KEY_RE.search(cleaned)
    if m is not None:
        tail = cleaned[m.end():]
        end_bracket = tail.find("]")
        body = tail if end_bracket == -1 else tail[:end_bracket]
        nums = [float(s) for s in _NUM_RE.findall(body)]
        if nums:
            return _coerce_length(nums, T)

    return None
