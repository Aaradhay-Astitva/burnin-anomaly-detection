"""Synthetic burn-in data generator.

Each lot has its own log-normal Iddq baseline (lot-to-lot process variation).
Normal parts drift slowly: V(t) = V0 * (1 + a * log(1 + t)).
A small fraction of parts get one of six injected defect signatures. Apart
from gross fails, every defect is kept under the datasheet limit, so static
pass/fail screening cannot see it.

Usage:  python -m burnin.generate
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import TIMES

DEFECT_TYPES = (
    "gross_fail",       # exceeds datasheet max -- static limits catch these
    "lot_outlier",      # high from 0h relative to its lot, but under the limit
    "latent_linear",    # normal at 0h, then a steep linear drift
    "accelerating",     # exponential wear-out, barely visible at 24h
    "step_jump",        # sudden shift at 96h
    "erratic",          # noisy, non-monotonic
)

LATENT_CAP_FRACTION = 0.98   # latent defects stay just under the datasheet max


def generate(
    n_lots: int = 20,
    parts_per_lot: int = 500,
    defect_rate: float = 0.04,
    datasheet_max: float = 50.0,
    seed: int = 42,
    lot_prefix: str = "L",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.asarray(TIMES, dtype=float)
    frames = []

    for lot in range(n_lots):
        lot_id = f"{lot_prefix}{lot + 1:02d}"
        n = parts_per_lot
        lot_median = rng.uniform(5.0, 15.0)      # uA
        lot_sigma = rng.uniform(0.10, 0.20)      # log-space spread
        lot_drift = rng.uniform(0.01, 0.04)      # drift coefficient a

        v0 = lot_median * np.exp(rng.normal(0.0, lot_sigma, n))
        a = lot_drift * np.exp(rng.normal(0.0, 0.3, n))
        V = v0[:, None] * (1.0 + a[:, None] * np.log1p(t))
        V *= np.exp(rng.normal(0.0, 0.01, V.shape))          # measurement noise

        defect = np.full(n, "none", dtype=object)
        n_def = rng.binomial(n, defect_rate)
        idx = rng.choice(n, size=n_def, replace=False)
        defect[idx] = rng.choice(DEFECT_TYPES, size=n_def)
        cap = LATENT_CAP_FRACTION * datasheet_max

        for i in idx:
            kind = defect[i]
            base = V[i].copy()
            if kind == "gross_fail":
                V[i] = rng.uniform(1.05, 1.6) * datasheet_max * (1 + a[i] * np.log1p(t))
                continue
            if kind == "lot_outlier":
                growth = base / base[0]
                target_v0 = min(rng.uniform(3.0, 4.5) * lot_median, 0.95 * cap / growth[-1])
                V[i] = target_v0 * growth
            elif kind == "latent_linear":
                total = rng.uniform(0.6, 2.0) * v0[i]
                V[i] = base + total * t / t[-1]
            elif kind == "accelerating":
                tau = 60.0
                total = rng.uniform(0.8, 2.5) * v0[i]
                V[i] = base + total * np.expm1(t / tau) / np.expm1(t[-1] / tau)
            elif kind == "step_jump":
                V[i] = base * np.where(t >= 96, rng.uniform(1.6, 2.5), 1.0)
            elif kind == "erratic":
                jitter = np.exp(rng.normal(0.0, 0.25, len(t)))
                jitter[0] = 1.0
                V[i] = base * jitter
            V[i] = np.minimum(V[i], cap)

        frame = pd.DataFrame({
            "lot_id": lot_id,
            "part_id": [f"{lot_id}-P{j + 1:04d}" for j in range(n)],
            "parameter": "Iddq",
            "unit": "uA",
        })
        for k, tt in enumerate(TIMES):
            frame[f"V{tt}"] = V[:, k].round(4)
        frame["datasheet_max"] = datasheet_max
        frame["label"] = (defect != "none").astype(int)
        frame["defect_type"] = defect
        frames.append(frame)

    return pd.DataFrame(pd.concat(frames, ignore_index=True))


def main(out_dir: str | Path = "data") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = generate(n_lots=20, seed=42, lot_prefix="L")
    test = generate(n_lots=8, seed=7, lot_prefix="T")
    train.to_csv(out / "burnin_train.csv", index=False)
    test.to_csv(out / "burnin_test.csv", index=False)
    # Early-screening demo file: only the 0h / 24h readings, no labels.
    test[["lot_id", "part_id", "parameter", "unit", "V0", "V24", "datasheet_max"]].to_csv(
        out / "burnin_test_24h.csv", index=False)

    for name, d in (("train", train), ("test", test)):
        print(f"{name}: {len(d)} parts, {d['lot_id'].nunique()} lots, "
              f"{d['label'].sum()} defects")
        print(d["defect_type"].value_counts().to_string(), "\n")
    print(f"Wrote CSVs to {out.resolve()}")


if __name__ == "__main__":
    main()
