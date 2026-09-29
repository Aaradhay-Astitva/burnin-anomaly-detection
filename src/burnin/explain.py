"""Plain-language reason codes a QA inspector can check by hand.

Every statement quotes the measured value, the lot reference and the
threshold, so each flag can be verified against the raw data sheet.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import GROUP_COLS, TIMES
from .module_b_drift import safety_slope

MAX_Z_REASONS = 3
SEP = " | "


def _lot_median(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby(GROUP_COLS, sort=False)[col].transform("median")


def reason_codes(df, raw, f_all, sev_all, dec, th) -> list[str]:
    lot_med = {}
    for t in TIMES:
        if f"V{t}" in df.columns and f"logV{t}" in f_all.columns:
            lot_med[f"V{t}"] = _lot_median(df, f"V{t}")
    for c in f_all.columns:
        if c.startswith("lr_") or c == "curv":
            lot_med[c] = _lot_median(f_all, c)
    safe = safety_slope(raw, df["V0"], th.slope_k)
    maxz = sev_all.max(axis=1)
    v168_measured = "logV168" in f_all.columns

    out = []
    for i in df.index:
        row = df.loc[i]
        p, u, lot, lim = row["parameter"], row["unit"], row["lot_id"], row["datasheet_max"]
        reasons = []

        if dec.at[i, "trig_static"]:
            for t in TIMES:
                col = f"V{t}"
                if col in lot_med and row[col] > lim:
                    reasons.append(f"STATIC FAIL: {p}@{t}h = {row[col]:.1f} {u} exceeds datasheet max {lim:g} {u}.")
                    break

        hits = sev_all.loc[i]
        hits = hits[hits >= th.z_inspect].sort_values(ascending=False).head(MAX_Z_REASONS)
        for feat, z in hits.items():
            if m := re.fullmatch(r"z_lvl_(\d+)", feat):
                t = int(m.group(1))
                v, med = row[f"V{t}"], lot_med[f"V{t}"][i]
                note = (f"; still under datasheet max {lim:g} {u}, so a static limit would pass it"
                        if v <= lim else "")
                reasons.append(f"LOT OUTLIER: {p}@{t}h = {v:.1f} {u} is {z:.1f} robust-sigma above "
                               f"lot {lot} median ({med:.1f} {u}){note}.")
            elif m := re.fullmatch(r"z_drift_(\d+)_(\d+)", feat):
                a, b = m.groups()
                lr = f"lr_{a}_{b}"
                pct, lpct = np.expm1(f_all.at[i, lr]) * 100, np.expm1(lot_med[lr][i]) * 100
                reasons.append(f"ABNORMAL DRIFT: {p} changed {pct:+.0f}% from {a}h to {b}h vs lot "
                               f"median {lpct:+.0f}% ({z:.1f} robust-sigma).")
            elif feat == "z_curv":
                direction = "accelerating" if f_all.at[i, "curv"] > lot_med["curv"][i] else "decelerating"
                reasons.append(f"DRIFT SHAPE: drift rate is {direction} over burn-in, {z:.1f} robust-sigma "
                               f"from the lot norm (wear-out signature).")

        if dec.at[i, "trig_iforest"] and not len(hits):
            pct = max(raw.at[i, c] for c in raw.columns if c.startswith("if_pct_"))
            reasons.append(f"UNUSUAL TRAJECTORY: combined level/drift pattern is more isolated than "
                           f"{pct * 100:.1f}% of reference parts (Isolation Forest), though no single "
                           f"reading crosses {th.z_inspect:g} robust-sigma.")

        if dec.at[i, "trig_drift_slope"]:
            reasons.append(f"EARLY REJECT (drift forecast): predicted {p}@168h = {raw.at[i, 'V168_pred']:.1f} {u} "
                           f"from 0h/24h readings; drift slope {raw.at[i, 'pred_slope']:.3f} {u}/h exceeds "
                           f"lot safety slope {safe[i]:.3f} {u}/h ({raw.at[i, 'z_pred_drift']:.1f} robust-sigma).")
        if dec.at[i, "trig_guard_band"]:
            up = raw.at[i, "V168_upper"]
            reasons.append(f"GUARD BAND: 90% upper forecast for {p}@168h is {up:.1f} {u} = "
                           f"{up / lim * 100:.0f}% of datasheet max (guard band {th.guard_frac * 100:.0f}%).")

        if not reasons:
            tail = ("full trajectory inside the lot envelope" if v168_measured
                    else "predicted 168h drift inside the lot safety envelope")
            reasons.append(f"OK: within lot-relative limits (max {maxz[i]:.1f} robust-sigma < {th.z_inspect:g}); "
                           f"{tail}.")
        out.append(SEP.join(reasons))
    return out
