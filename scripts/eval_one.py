"""Evaluate one predictions parquet under the docs/07 revised protocol.

Reports IBS / calibration_slope / AUC-ρ / legacy ρ_hook+ρ_comp + paired BCa
diff vs. a set of reference baselines (B1 train-mean is mandatory).

Usage:
    python scripts/eval_one.py PRED.parquet [--name TAG] [--vs B1 SFT ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", type=Path)
    ap.add_argument("--name", default=None, help="display tag")
    ap.add_argument("--report", type=Path, default=None, help="optional JSON output path")
    ap.add_argument("--vs", nargs="*", default=["B1"],
                    help="paired BCa references (key or parquet path)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/home/ssm-user/work/hf-cache")
    sys.path.insert(0, "/home/ssm-user/work/ttcc-eval/src")
    from ttcc_eval.eval import evaluate, paired_compare

    KNOWN = {
        "B1":    "/home/ssm-user/work/work-out/B1_train_mean.parquet",
        "B2":    "/home/ssm-user/work/work-out/B2_linear_T.parquet",
        "SFT":   "/home/ssm-user/work/work-out/preds_sft.parquet",
        "GRPO":  "/home/ssm-user/work/work-out/preds_grpo.parquet",
        "RLOO":  "/home/ssm-user/work/work-out/preds_rloo.parquet",
        "iter1": "/home/ssm-user/work/work-out/qwen25_omni_3b_seed0_iter1.parquet",
        "iter2": "/home/ssm-user/work/work-out/qwen25_omni_3b_seed0_iter2.parquet",
        "OLD":   "/home/ssm-user/work/work-out/qwen25_omni_3b_seed0_modeCollapse.parquet",
    }

    rep = evaluate(args.pred, B=10000, seed=0)
    name = args.name or args.pred.stem
    m = rep.metrics

    def fmt(d): return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"

    print(f"=== {name} ===")
    print(f"  IBS              = {fmt(m['ibs'])}")
    print(f"  calibration slope= {fmt(m['calibration_slope'])}")
    print(f"  AUC-ρ            = {fmt(m['auc_spearman'])}")
    print(f"  (legacy) ρ_hook  = {fmt(m['hook_spearman'])}")
    print(f"  (legacy) ρ_comp  = {fmt(m['completion_spearman'])}")

    for ref in args.vs:
        ref_path = Path(KNOWN.get(ref, ref))
        if not ref_path.exists():
            print(f"  [skip] reference {ref} not found at {ref_path}")
            continue
        cmp = paired_compare(ref_path, args.pred, B=10000, seed=0)
        d = cmp["ibs"]["diff"]
        verdict = ("BEATS"   if d["point"] < 0 and d["hi"] < 0 else
                   "loses to" if d["point"] > 0 and d["lo"] > 0 else
                   "tied")
        ref_tag = ref if ref in KNOWN else ref_path.stem
        print(f"  paired ΔIBS({name} − {ref_tag}) = {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  ← {verdict}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep.to_dict(), indent=2, default=float))
        print(f"  report → {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
