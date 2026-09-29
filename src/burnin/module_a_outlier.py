"""Module A: dynamic (lot-relative) outlier detection.

Three layers, OR-combined in the decision layer to keep false negatives low:
  1. Static datasheet limit (baseline).
  2. Robust z-scores per lot, per read point, per drift interval. These are
     univariate and easy to explain.
  3. Isolation Forest on the lot-normalised severities. It catches unusual
     trajectory *shapes* that no single z-score flags. Because it only sees
     lot-normalised inputs it works on unseen lots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import build_features, severity, times_available

Z_CLIP = 25.0   # cap severities so a single huge outlier doesn't dominate the forest


class DynamicOutlierDetector:
    def __init__(self, stage: int | None = None, n_estimators: int = 300, random_state: int = 0):
        self.stage = stage
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> "DynamicOutlierDetector":
        f, zcols = build_features(df, self.stage)
        self.zcols_ = zcols
        self.times_ = times_available(df, self.stage)
        X = severity(f, zcols).clip(upper=Z_CLIP).to_numpy()
        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.random_state).fit(X)
        # Reference distribution for turning raw scores into percentiles.
        self.train_scores_ = np.sort(-self.iforest_.score_samples(X))
        return self

    def matches(self, df: pd.DataFrame) -> bool:
        """True if `df` has exactly the read points this detector was fit on."""
        return times_available(df, self.stage) == self.times_

    def score(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns (scores, features, severities) indexed like `df`.

        scores columns: max_z, max_z_feature, if_pct (percentile of the
        isolation score vs the training population), static_fail.
        """
        f, zcols = build_features(df, self.stage)
        if zcols != self.zcols_:
            raise ValueError(f"Data does not have the read points this detector was fit on: "
                             f"expected {self.zcols_}, got {zcols}")
        sev = severity(f, zcols)
        raw = -self.iforest_.score_samples(sev.clip(upper=Z_CLIP).to_numpy())
        if_pct = np.searchsorted(self.train_scores_, raw, side="right") / len(self.train_scores_)

        margins = f[[c for c in f.columns if c.startswith("margin_")]]
        scores = pd.DataFrame({
            "max_z": sev.max(axis=1),
            "max_z_feature": sev.idxmax(axis=1),
            "if_pct": if_pct,
            "static_fail": (margins > 1.0).any(axis=1),
        }, index=df.index)
        return scores, f, sev
