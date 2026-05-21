"""Convert ms-swift infer JSONL → ttcc-eval predictions parquet.

The swift JSONL has ``response``/``videos`` keys but lacks ``ad_id``/``T``/
``R_true``. We join those back from the test-split JSONL using video path.

Usage (module mode):
    python -m ttcc_rl.postprocess <infer.jsonl> <test.jsonl> <out.parquet> <method-tag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ttcc_rl.parser import parse_curve


def convert(infer_jsonl: Path, test_jsonl: Path, out_parquet: Path, method: str) -> int:
    lookup = {}
    with open(test_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lookup[r["videos"][0]] = (r["ad_id"], r["T"], r["R_true"])

    ad_ids: list[str] = []
    curves: list[list[float]] = []
    with open(infer_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            video = (r.get("videos") or [r.get("video", "")])[0]
            if video not in lookup:
                continue
            ad_id, T, _ = lookup[video]
            curve = parse_curve(r.get("response", ""), T)
            if curve is None:
                print(f"PARSE FAIL ad={ad_id}", file=sys.stderr)
                continue
            ad_ids.append(ad_id)
            curves.append(curve)

    table = pa.table({
        "ad_id":  pa.array(ad_ids, type=pa.string()),
        "R_hat":  pa.array(curves, type=pa.list_(pa.float64())),
        "method": pa.array([method] * len(ad_ids), type=pa.string()),
        "seed":   pa.array([0] * len(ad_ids), type=pa.int64()),
    })
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_parquet)
    print(f"wrote {len(ad_ids)} predictions → {out_parquet}")
    return len(ad_ids)


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: python -m ttcc_rl.postprocess <infer.jsonl> <test.jsonl> <out.parquet> <method>", file=sys.stderr)
        return 1
    return 0 if convert(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
