"""Risk decision layer: combines the static limit, Module A and Module B into
PASS / INSPECT / REJECT.

  REJECT  : static datasheet fail, OR lot-relative z >= z_reject,
            OR predicted drift above the safety slope (early rejection)
  INSPECT : lot-relative z >= z_inspect, OR Isolation-Forest percentile
            >= if_pct, OR predicted 168h upper bound inside the guard band
  PASS    : none of the above

The Module B rules (safety slope, guard band) apply only while V168 has not
been measured yet. That is the early-rejection use case. Once the real 168h
reading exists, Module A judges the actual trajectory instead of a forecast.

Both INSPECT and REJECT count as "caught" when scoring recall.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import times_available, validate_input
from .module_a_outlier import DynamicOutlierDetector
from .module_b_drift import DriftPredictor

STAGES = (24, 96, 168)


@dataclass
class Thresholds:
    z_inspect: float = 3.5
    z_reject: float = 5.0
    if_pct: float = 0.99
    slope_k: float = 4.0
    guard_frac: float = 0.8

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        return cls(**json.loads(Path(path).read_text()))


def _stage_max(raw: pd.DataFrame, prefix: str, stages) -> pd.Series:
    cols = [f"{prefix}_{s}" for s in stages if f"{prefix}_{s}" in raw.columns]
    if not cols:
        return pd.Series(0.0, index=raw.index)
    return raw[cols].max(axis=1, skipna=True).fillna(0.0)


def decide(raw: pd.DataFrame, th: Thresholds, stages=STAGES,
           use_static: bool = True, use_a: bool = True, use_b: bool = True) -> pd.DataFrame:
    """Vectorised decision rule over raw scores (see ScreeningPipeline.raw_scores).

    The ablation switches let evaluate.py score each module on its own.
    """
    false = pd.Series(False, index=raw.index)
    maxz = _stage_max(raw, "max_z", stages)
    ifp = _stage_max(raw, "if_pct", stages)
    static_cols = [f"static_fail_{s}" for s in stages if f"static_fail_{s}" in raw.columns]

    static = raw[static_cols].fillna(False).astype(bool).any(axis=1) if (use_static and static_cols) else false
    a_reject = (maxz >= th.z_reject) if use_a else false
    a_inspect = ((maxz >= th.z_inspect) | (ifp >= th.if_pct)) if use_a else false
    # The V168 forecast drives decisions only until V168 is actually measured;
    # after that, Module A judges the real trajectory.
    v168_measured = 168 in stages and "max_z_168" in raw.columns
    if use_b and "z_pred_drift" in raw.columns and not v168_measured:
        b_reject = raw["z_pred_drift"] >= th.slope_k
        b_inspect = raw["upper_margin"] >= th.guard_frac
    else:
        b_reject = b_inspect = false

    reject = static | a_reject | b_reject
    inspect = a_inspect | b_inspect
    out = pd.DataFrame({
        "status": np.where(reject, "REJECT", np.where(inspect, "INSPECT", "PASS")),
        "trig_static": static,
        "trig_lot_z": (maxz >= th.z_inspect) & use_a,
        "trig_iforest": (ifp >= th.if_pct) & use_a,
        "trig_drift_slope": b_reject,
        "trig_guard_band": b_inspect,
    }, index=raw.index)
    return out


SEVERITY = {"PASS": 0, "INSPECT": 1, "REJECT": 2}


def summarize_parts(res: pd.DataFrame) -> pd.DataFrame:
    """One row per physical part: the worst status across its parameters."""
    r = res.assign(_sev=res["status"].map(SEVERITY),
                   _why=res["parameter"] + ": " + res["reasons"])
    flagged_why = r[r["status"] != "PASS"].groupby(["lot_id", "part_id"])["_why"].agg(" || ".join)
    out = r.groupby(["lot_id", "part_id"]).agg(
        parameters=("parameter", "nunique"), worst_sev=("_sev", "max"),
        max_anomaly_score=("anomaly_score", "max"))
    out["status"] = out.pop("worst_sev").map({v: k for k, v in SEVERITY.items()})
    out["flagged_parameters"] = flagged_why.reindex(out.index).fillna("")
    return out.reset_index()


class ScreeningPipeline:
    def __init__(self, thresholds: Thresholds | None = None, drift_model: str = "ridge"):
        self.thresholds = thresholds or Thresholds()
        self.drift_model = drift_model

    def fit(self, df: pd.DataFrame) -> "ScreeningPipeline":
        df = validate_input(df)
        avail = times_available(df)
        self.detectors_ = {s: DynamicOutlierDetector(stage=s).fit(df) for s in STAGES if s in avail}
        self.drift_ = DriftPredictor(self.drift_model).fit(df)
        return self

    def raw_scores(self, df: pd.DataFrame, return_details: bool = False):
        """Label-free scores for every part: Module A per stage + Module B."""
        df = validate_input(df)
        avail = times_available(df)
        raw = pd.DataFrame(index=df.index)
        feats, sevs = [], []
        for s in STAGES:
            if s not in avail:
                continue
            det = self.detectors_.get(s)
            if det is None or not det.matches(df):
                # Different read-point set from training (e.g. no V96). Module A is
                # unsupervised and lot-relative, so fit it on the lots being screened.
                det = DynamicOutlierDetector(stage=s).fit(df)
            scores, f, sev = det.score(df)
            for c in scores.columns:
                raw[f"{c}_{s}"] = scores[c]
            feats.append(f)
            sevs.append(sev)
        pred = self.drift_.predict(df)
        raw = raw.join(pred)
        if not return_details:
            return raw
        # Features/severities merged across stages. Columns with the same name
        # are identical, because they are computed from the same readings.
        f_all = pd.concat(feats, axis=1)
        f_all = f_all.loc[:, ~f_all.columns.duplicated()]
        sev_all = pd.concat(sevs, axis=1)
        sev_all = sev_all.loc[:, ~sev_all.columns.duplicated()]
        return raw, f_all, sev_all, df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full screening: scores, decision and plain-language reasons per part."""
        from .explain import reason_codes

        raw, f_all, sev_all, df = self.raw_scores(df, return_details=True)
        dec = decide(raw, self.thresholds)
        maxz = _stage_max(raw, "max_z", STAGES)
        ifp = _stage_max(raw, "if_pct", STAGES)
        res = df.copy()
        res["status"] = dec["status"]
        res["anomaly_score"] = np.maximum(np.clip(maxz / (2 * self.thresholds.z_inspect), 0, 1), ifp).round(3)
        res["max_lot_z"] = maxz.round(2)
        res["iforest_pct"] = ifp.round(3)
        res["V168_pred"] = raw["V168_pred"].round(3)
        res["V168_upper"] = raw["V168_upper"].round(3)
        res["pred_slope"] = raw["pred_slope"].round(4)
        res["z_pred_drift"] = raw["z_pred_drift"].round(2)
        # The forecast extrapolates when the 0-24h behaviour is itself far outside the lot.
        early = sev_all[[c for c in ("z_lvl_24", "z_drift_0_24") if c in sev_all.columns]].max(axis=1)
        res["forecast_confidence"] = np.where(early >= self.thresholds.z_inspect, "low", "normal")
        res = res.join(dec.drop(columns="status"))
        res["reasons"] = reason_codes(df, raw, f_all, sev_all, dec, self.thresholds)
        return res
