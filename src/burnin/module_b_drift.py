"""Module B: time-series drift predictor.

Forecasts V168 from the 0h and 24h readings only (plus lot context that is
also derived from 0h/24h), then compares the predicted drift against a
lot-relative safety slope.

The target is the relative drift log(V168 / V0). Leakage is log-normal and
drift is multiplicative, so a linear model in log space fits the physics and
stays interpretable. All inputs are ratios or lot-relative z-scores, which
makes the model scale-free: it transfers across parameters, units and lots
with different baselines (e.g. Iddq in uA, propagation delay in ns).

Limitation: a defect that is completely silent at 24h cannot be forecast from
V0/V24. Module A on the 96h/168h read points is the backstop for those parts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import GROUP_COLS
from .features import build_features, lot_center_scale

B_FEATURES = ["lr_0_24", "lot_med_lr_0_24", "z_lvl_0", "z_lvl_24", "z_drift_0_24"]
HORIZON_H = 168.0


def _make_model(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0)
    if name == "gbm":
        return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                         random_state=0)
    raise ValueError(f"Unknown drift model {name!r}; use ridge, rf or gbm")


def drift_features(df: pd.DataFrame) -> pd.DataFrame:
    f, _ = build_features(df, stage=24)
    grp = f.groupby(GROUP_COLS, sort=False)
    f["lot_med_lr_0_24"] = grp["lr_0_24"].transform("median")
    return f


class DriftPredictor:
    def __init__(self, model: str = "ridge", quantile: float = 0.9):
        self.model = model
        self.quantile = quantile

    def fit(self, df: pd.DataFrame) -> "DriftPredictor":
        if "V168" not in df.columns or df["V168"].isna().any():
            raise ValueError("Training the drift predictor needs V168 for every part.")
        f = drift_features(df)
        X = f[B_FEATURES].to_numpy()
        y = np.log(df["V168"].to_numpy()) - f["logV0"].to_numpy()
        self.model_ = _make_model(self.model).fit(X, y)
        self.upper_ = GradientBoostingRegressor(
            loss="quantile", alpha=self.quantile, n_estimators=200, max_depth=3,
            learning_rate=0.05, random_state=0).fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        f = drift_features(df)
        X = f[B_FEATURES].to_numpy()
        v0 = df["V0"].to_numpy()
        pred_lr = self.model_.predict(X)
        pred = v0 * np.exp(pred_lr)
        upper = np.maximum(v0 * np.exp(self.upper_.predict(X)), pred)

        out = pd.DataFrame(index=df.index)
        out["V168_pred"] = pred
        out["V168_upper"] = upper
        out["pred_slope"] = (pred - v0) / HORIZON_H          # units per hour
        out["upper_margin"] = upper / df["datasheet_max"].to_numpy()

        # Lot-relative safety envelope on the predicted relative drift.
        tmp = df[GROUP_COLS].copy()
        tmp["pred_lr"] = pred_lr
        med, scale = lot_center_scale(tmp, "pred_lr")
        out["pred_lr"] = tmp["pred_lr"]
        out["lot_med_pred_lr"] = med
        out["lot_scale_pred_lr"] = scale
        out["z_pred_drift"] = (tmp["pred_lr"] - med) / scale
        return out

    def coefficients(self) -> pd.Series | None:
        """Standardised Ridge coefficients (target: log V168/V0), for explainability."""
        if self.model != "ridge":
            return None
        ridge = self.model_[-1]
        return pd.Series(ridge.coef_, index=B_FEATURES).sort_values(key=np.abs, ascending=False)


def safety_slope(pred: pd.DataFrame, v0: pd.Series, k: float) -> pd.Series:
    """Per-part safety slope in uA/h: the drift allowed at lot median + k robust sigma."""
    allowed_lr = pred["lot_med_pred_lr"] + k * pred["lot_scale_pred_lr"]
    return v0 * np.expm1(allowed_lr) / HORIZON_H


def naive_extrapolation(df: pd.DataFrame) -> np.ndarray:
    """Baseline: straight line through V0 and V24, extended to 168h."""
    return (df["V0"] + (df["V24"] - df["V0"]) * (HORIZON_H / 24.0)).to_numpy()
