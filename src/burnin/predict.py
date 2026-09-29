"""Screen a CSV with the trained pipeline and write a QA report.

Usage:
    python -m burnin.predict data/burnin_test_24h.csv -o reports/predictions.csv

The output has one row per part and parameter: status (PASS/INSPECT/REJECT),
the V168 forecast with its 90% upper bound, scores and plain-language reasons.
If a part has several parameters, a per-part roll-up (worst status) is also
written next to the output as *_parts.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .decision import summarize_parts
from .features import small_lots

OUTPUT_COLS = ["lot_id", "part_id", "parameter", "unit", "status", "anomaly_score", "max_lot_z",
               "iforest_pct", "V168_pred", "V168_upper", "forecast_confidence", "pred_slope", "z_pred_drift",
               "trig_static", "trig_lot_z", "trig_iforest", "trig_drift_slope", "trig_guard_band",
               "reasons"]


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("reports/predictions.csv"))
    ap.add_argument("--model", type=Path, default=Path("models/pipeline.joblib"))
    args = ap.parse_args(argv)

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found. Run `python -m burnin.evaluate` first to train it.")
    pipe = joblib.load(args.model)
    df = pd.read_csv(args.input)
    res = pipe.run(df)

    tiny = small_lots(res)
    if len(tiny):
        print(f"WARNING: {len(tiny)} lot(s) have fewer than 30 parts, so their dynamic limits are less "
              f"reliable: {', '.join(f'{lot}/{p} ({n})' for (lot, p), n in tiny.items())}")

    extra = [c for c in res.columns if c.startswith("V") and c[1:].isdigit()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res[OUTPUT_COLS[:4] + extra + OUTPUT_COLS[4:]].to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Screened {len(res)} rows: {res['status'].value_counts().to_dict()}  ->  {args.output}")

    if res.groupby(["lot_id", "part_id"])["parameter"].nunique().max() > 1:
        parts_path = args.output.with_name(args.output.stem + "_parts.csv")
        parts = summarize_parts(res)
        parts.to_csv(parts_path, index=False, encoding="utf-8-sig")
        print(f"Per-part roll-up ({len(parts)} parts): {parts['status'].value_counts().to_dict()}  ->  {parts_path}")
    return res


if __name__ == "__main__":
    main()
