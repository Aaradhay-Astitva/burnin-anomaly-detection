"""Evaluation, threshold tuning and model export.

1. GroupKFold (by lot) on the training set gives out-of-fold scores. Those
   scores are used to tune the decision thresholds, targeting recall >= 0.98
   at the lowest false-positive rate, and to compare the drift regressors.
2. The final pipeline is fit on all training lots and scored on a separate,
   independently generated held-out test set (unseen lots).
3. Models, thresholds, metrics and per-part results are written to models/
   and reports/.

Usage:  python -m burnin.evaluate [--target-recall 0.98]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import generate
from .decision import ScreeningPipeline, Thresholds, decide
from .features import validate_input
from .module_b_drift import DriftPredictor, naive_extrapolation

DRIFT_MODELS = ("ridge", "rf", "gbm")

GRID = {
    "z_inspect": [3.0, 3.5, 4.0, 4.5, 5.0],
    "if_pct": [0.97, 0.98, 0.99, 0.995, 1.01],     # 1.01 = Isolation Forest disabled
    "slope_k": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
    "guard_frac": [0.7, 0.8, 0.9, 1.01],            # 1.01 = guard band disabled
}

ABLATIONS = {
    "Static datasheet limits (baseline)": dict(use_a=False, use_b=False),
    "Module A @24h only": dict(stages=(24,), use_b=False),
    "Module B only (24h forecast)": dict(stages=(24,), use_static=False, use_a=False),
    "Early screen @24h (static + A + B)": dict(stages=(24,)),
    "Full run @168h (static + A; forecast superseded)": dict(),
}


# ---------------------------------------------------------------- metrics ---
def classification_metrics(y: np.ndarray, flagged: np.ndarray) -> dict:
    y, flagged = np.asarray(y, bool), np.asarray(flagged, bool)
    tp, fn = int((y & flagged).sum()), int((y & ~flagged).sum())
    fp, tn = int((~y & flagged).sum()), int((~y & ~flagged).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
    return dict(recall=round(recall, 4), false_negatives=fn, precision=round(precision, 4),
                f2=round(f2, 4), false_positive_rate=round(fp / max(fp + tn, 1), 4),
                tp=tp, fp=fp, tn=tn)


def ablation_tables(raw, y, types, th) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, per_type = {}, {}
    for name, kw in ABLATIONS.items():
        flagged = decide(raw, th, **kw)["status"].ne("PASS").to_numpy()
        summary[name] = classification_metrics(y, flagged)
        per_type[name] = pd.Series(flagged[y == 1]).groupby(types[y == 1]).mean()
    per_type = pd.DataFrame(per_type).T.round(3)
    per_type.columns.name = "recall by defect type"
    return pd.DataFrame(summary).T, per_type


def drift_mae_table(truth: np.ndarray, preds: dict, types: np.ndarray) -> pd.DataFrame:
    rows = {}
    for name, p in preds.items():
        err = np.abs(np.asarray(p) - truth)
        row = {"MAE_all": err.mean(), "RMSE_all": np.sqrt((err ** 2).mean()),
               "MAE_good_parts": err[types == "none"].mean()}
        for t in sorted(set(types) - {"none"}):
            row[f"MAE_{t}"] = err[types == t].mean()
        rows[name] = row
    return pd.DataFrame(rows).T.round(3)


# ------------------------------------------------------------ CV + tuning ---
def oof_scores(df: pd.DataFrame, n_splits: int = 5) -> dict[str, pd.DataFrame]:
    """Out-of-fold raw scores (Module A + each drift model's Module B), grouped by lot."""
    raws = {m: [] for m in DRIFT_MODELS}
    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=df["lot_id"]):
        train, test = df.iloc[tr], df.iloc[te]
        pipe = ScreeningPipeline(drift_model="ridge").fit(train)
        r = pipe.raw_scores(test)
        r.index = test.index
        a_cols = [c for c in r.columns if c.startswith(("max_z_", "if_pct_", "static_fail_"))]
        raws["ridge"].append(r)
        for m in DRIFT_MODELS[1:]:
            pred = DriftPredictor(m).fit(train).predict(test)
            pred.index = test.index
            raws[m].append(r[a_cols].join(pred))
    return {m: pd.concat(v).sort_index() for m, v in raws.items()}


def tune_thresholds(raw: pd.DataFrame, y: np.ndarray, target_recall: float) -> Thresholds:
    """Two-step grid search.

    1. Module A thresholds on the full burn-in run: meet the recall target at
       the lowest false-positive rate (otherwise maximise F2).
    2. Module B thresholds on the 24h early screen: maximise F2. Early recall
       has a hard ceiling, since defects that are silent at 24h cannot be
       forecast, so a recall target there would only buy false rejects.
    """
    best_key, th_a = None, None
    for zi, ifp in itertools.product(GRID["z_inspect"], GRID["if_pct"]):
        th = Thresholds(z_inspect=zi, z_reject=max(5.0, zi + 1.5), if_pct=ifp)
        m = classification_metrics(y, decide(raw, th)["status"].ne("PASS").to_numpy())
        key = (0, m["false_positive_rate"], -zi) if m["recall"] >= target_recall else (1, -m["f2"], 0)
        if best_key is None or key < best_key:
            best_key, th_a = key, th

    best_key, best = None, None
    for k, g in itertools.product(GRID["slope_k"], GRID["guard_frac"]):
        th = Thresholds(th_a.z_inspect, th_a.z_reject, th_a.if_pct, slope_k=k, guard_frac=g)
        m = classification_metrics(y, decide(raw, th, stages=(24,))["status"].ne("PASS").to_numpy())
        key = (-m["f2"], m["false_positive_rate"])
        if best_key is None or key < best_key:
            best_key, best = key, th
    return best


# ------------------------------------------------------------------- main ---
def _load_or_generate(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (data_dir / "burnin_train.csv").exists() or not (data_dir / "burnin_test.csv").exists():
        generate.main(data_dir)
    train = validate_input(pd.read_csv(data_dir / "burnin_train.csv"))
    test = validate_input(pd.read_csv(data_dir / "burnin_test.csv"))
    return train, test


def _show(title: str, table) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string() if hasattr(table, "to_string") else table)


def main(target_recall: float = 0.98, drift_model: str = "auto",
         data_dir="data", model_dir="models", report_dir="reports") -> dict:
    data_dir, model_dir, report_dir = Path(data_dir), Path(model_dir), Path(report_dir)
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    train, test = _load_or_generate(data_dir)
    y_tr, types_tr = train["label"].to_numpy(), train["defect_type"].to_numpy()
    y_te, types_te = test["label"].to_numpy(), test["defect_type"].to_numpy()

    # 1) Cross-validation on training lots.
    print(f"Cross-validating on {train['lot_id'].nunique()} training lots (GroupKFold by lot)...")
    raws_cv = oof_scores(train)
    preds_cv = {m: r["V168_pred"].to_numpy() for m, r in raws_cv.items()}
    preds_cv["naive_linear_extrapolation"] = naive_extrapolation(train)
    cv_mae = drift_mae_table(train["V168"].to_numpy(), preds_cv, types_tr)
    _show("CV drift prediction: V168 error in uA", cv_mae)

    if drift_model == "auto":
        drift_model = cv_mae.loc[list(DRIFT_MODELS), "MAE_all"].idxmin()
    print(f"\nDrift model for the final pipeline: {drift_model} "
          f"(Ridge coefficients are still reported for explainability)")

    raw_cv = raws_cv[drift_model]
    th = tune_thresholds(raw_cv, y_tr, target_recall)
    th.save(model_dir / "thresholds.json")
    _show("Tuned thresholds", pd.Series(th.__dict__))
    cv_summary, _ = ablation_tables(raw_cv, y_tr, types_tr, th)
    _show("CV detection (thresholds tuned on these folds, so slightly optimistic)", cv_summary)

    # 2) Final model on all training lots -> held-out test lots.
    pipe = ScreeningPipeline(thresholds=th, drift_model=drift_model).fit(train)
    joblib.dump(pipe, model_dir / "pipeline.joblib")

    raw_te = pipe.raw_scores(test)
    te_summary, te_types = ablation_tables(raw_te, y_te, types_te, th)
    _show(f"HELD-OUT TEST detection ({test['lot_id'].nunique()} unseen lots, {len(test)} parts)", te_summary)
    _show("HELD-OUT TEST recall by defect type", te_types)

    fitted = {m: (pipe.drift_ if m == drift_model else DriftPredictor(m).fit(train)) for m in DRIFT_MODELS}
    test_preds = {m: p.predict(test)["V168_pred"].to_numpy() for m, p in fitted.items()}
    test_preds["naive_linear_extrapolation"] = naive_extrapolation(test)
    te_mae = drift_mae_table(test["V168"].to_numpy(), test_preds, types_te)
    _show("HELD-OUT TEST drift prediction: V168 error in uA", te_mae)

    coefs = fitted["ridge"].coefficients()
    _show("Ridge coefficients (standardised inputs, target = log V168/V0)", coefs.round(4))

    results = pipe.run(test)
    results.to_csv(report_dir / "test_results.csv", index=False, encoding="utf-8-sig")
    missed = results[(results["label"] == 1) & (results["status"] == "PASS")]
    _show(f"Missed defects on test (false negatives: {len(missed)})",
          missed[["part_id", "defect_type", "V0", "V24", "V96", "V168", "max_lot_z", "z_pred_drift"]]
          if len(missed) else "none")
    _show("Status counts on test", results["status"].value_counts())

    metrics = {
        "target_recall": target_recall,
        "thresholds": th.__dict__,
        "drift_model": drift_model,
        "cv": {"detection": cv_summary.to_dict(orient="index"), "drift_mae": cv_mae.to_dict(orient="index")},
        "test": {"detection": te_summary.to_dict(orient="index"),
                 "recall_by_type": te_types.to_dict(orient="index"),
                 "drift_mae": te_mae.to_dict(orient="index")},
        "ridge_coefficients": coefs.to_dict(),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nSaved {model_dir / 'pipeline.joblib'}, {model_dir / 'thresholds.json'}, "
          f"{report_dir / 'metrics.json'}, {report_dir / 'test_results.csv'}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--drift-model", default="auto", choices=("auto",) + DRIFT_MODELS,
                    help="auto = lowest cross-validated V168 MAE")
    args = ap.parse_args()
    main(args.target_recall, args.drift_model)
