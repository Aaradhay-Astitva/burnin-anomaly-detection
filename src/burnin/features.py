"""Input validation and per-lot robust feature engineering.

Everything is lot-relative. A part is compared against the median and MAD of
its own lot at the same read point. That comparison is the "dynamic limit"
that replaces a static datasheet limit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import GROUP_COLS, TIMES

MAD_TO_SIGMA = 1.4826
EPS = 1e-9
# Floor on the robust scale in log space (~0.5 % relative). This stops very
# tight lots or quantised meter readings from producing absurd z-scores.
MIN_LOG_SCALE = 0.005

REQUIRED_COLS = ["lot_id", "part_id", "V0", "V24", "datasheet_max"]
MIN_LOT_SIZE = 30   # below this, lot median/MAD get noisy and dynamic limits are unreliable


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns, fill optional ones, coerce types."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    df = df.copy()
    if "parameter" not in df.columns:
        df["parameter"] = "Iddq"
    if "unit" not in df.columns:
        df["unit"] = "uA"
    df["lot_id"] = df["lot_id"].astype(str)
    df["part_id"] = df["part_id"].astype(str)
    for t in TIMES:
        col = f"V{t}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["V0", "V24"]].isna().any().any():
        raise ValueError("V0 and V24 must be present for every part.")
    if (df[["V0", "V24"]] <= 0).any().any():
        raise ValueError("Readings must be positive (leakage / current values).")
    dup = df.duplicated(["lot_id", "part_id", "parameter"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (lot_id, part_id, parameter) rows, e.g. "
                         f"{df.loc[dup, 'part_id'].iloc[0]!r}. Each part needs one row per parameter.")
    return df.reset_index(drop=True)


def small_lots(df: pd.DataFrame, min_parts: int = MIN_LOT_SIZE) -> pd.Series:
    """Lots too small for reliable robust statistics (median/MAD)."""
    sizes = df.groupby(GROUP_COLS).size()
    return sizes[sizes < min_parts]


def robust_center_scale(x, min_scale: float = MIN_LOG_SCALE) -> tuple[float, float]:
    """Median and MAD-based sigma estimate."""
    x = np.asarray(x, dtype=float)
    med = float(np.nanmedian(x))
    scale = MAD_TO_SIGMA * float(np.nanmedian(np.abs(x - med)))
    return med, max(scale, min_scale)


def robust_z(x, min_scale: float = MIN_LOG_SCALE) -> np.ndarray:
    med, scale = robust_center_scale(x, min_scale)
    return (np.asarray(x, dtype=float) - med) / scale


def lot_transform(frame: pd.DataFrame, col: str, func) -> pd.Series:
    return frame.groupby(GROUP_COLS, sort=False)[col].transform(lambda s: func(s.to_numpy()))


def lot_robust_z(frame: pd.DataFrame, col: str) -> pd.Series:
    return lot_transform(frame, col, robust_z)


def lot_center_scale(frame: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    med = lot_transform(frame, col, lambda x: np.full(len(x), robust_center_scale(x)[0]))
    scale = lot_transform(frame, col, lambda x: np.full(len(x), robust_center_scale(x)[1]))
    return med, scale


def times_available(df: pd.DataFrame, stage: int | None = None) -> list[int]:
    ts = [t for t in TIMES if f"V{t}" in df.columns and df[f"V{t}"].notna().all()]
    if stage is not None:
        ts = [t for t in ts if t <= stage]
    return ts


def build_features(df: pd.DataFrame, stage: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Lot-relative features built from the read points up to `stage` hours.

    Returns the feature frame and the names of its robust-z columns:
      z_lvl_{t}      level at read point t (log space)
      z_drift_{a}_{b} log-ratio change between consecutive read points
      z_drift_0_{T}  total drift (when there are more than 2 read points)
      z_curv         change in drift rate, last interval vs first (>= 3 points)
    """
    ts = times_available(df, stage)
    if ts[:2] != [0, 24]:
        raise ValueError("Need V0 and V24 readings to build features.")

    f = df[GROUP_COLS].copy()
    zcols: list[str] = []
    for t in ts:
        f[f"logV{t}"] = np.log(df[f"V{t}"].clip(lower=EPS))
        f[f"margin_{t}"] = df[f"V{t}"] / df["datasheet_max"]
        f[f"z_lvl_{t}"] = lot_robust_z(f, f"logV{t}")
        zcols.append(f"z_lvl_{t}")

    pairs = list(zip(ts[:-1], ts[1:]))
    if len(ts) > 2:
        pairs.append((0, ts[-1]))
    for a, b in pairs:
        f[f"lr_{a}_{b}"] = f[f"logV{b}"] - f[f"logV{a}"]
        f[f"z_drift_{a}_{b}"] = lot_robust_z(f, f"lr_{a}_{b}")
        zcols.append(f"z_drift_{a}_{b}")

    if len(ts) >= 3:
        (a0, b0), (a1, b1) = pairs[0], pairs[len(ts) - 2]
        f["curv"] = f[f"lr_{a1}_{b1}"] / (b1 - a1) - f[f"lr_{a0}_{b0}"] / (b0 - a0)
        f["z_curv"] = lot_robust_z(f, "curv")
        zcols.append("z_curv")

    return f, zcols


def severity(f: pd.DataFrame, zcols: list[str]) -> pd.DataFrame:
    """Turn z-scores into anomaly severities.

    Levels are one-sided: an unusually *low* leakage is not a reliability
    risk. Drift and curvature are two-sided, because erratic parts can move
    either way.
    """
    sev = pd.DataFrame(index=f.index)
    for c in zcols:
        sev[c] = f[c].clip(lower=0) if c.startswith("z_lvl_") else f[c].abs()
    return sev
