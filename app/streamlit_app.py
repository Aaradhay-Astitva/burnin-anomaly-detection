"""Burn-in screening dashboard.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from burnin import GROUP_COLS, TIMES, generate  # noqa: E402
from burnin.decision import ScreeningPipeline, Thresholds, summarize_parts  # noqa: E402
from burnin.evaluate import classification_metrics  # noqa: E402
from burnin.explain import SEP  # noqa: E402
from burnin.features import MIN_LOT_SIZE, robust_center_scale, small_lots, times_available, validate_input  # noqa: E402

DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"

# Reference palette (dataviz skill): status colours always ship with a text label.
STATUS_COLORS = {"PASS": "#0ca30c", "INSPECT": "#fab219", "REJECT": "#d03b3b"}
STATUS_ICON = {"PASS": "✅", "INSPECT": "⚠️", "REJECT": "⛔"}
C_PART, C_FORECAST, C_MEDIAN = "#2a78d6", "#eb6834", "#898781"
C_BAND, C_CONTEXT, C_LIMIT = "rgba(42,120,214,0.14)", "rgba(137,135,129,0.25)", "#d03b3b"

st.set_page_config(page_title="Burn-in Anomaly Screening", page_icon="🔬", layout="wide")


# ----------------------------------------------------------------- loading ---
@st.cache_resource
def load_pipeline() -> ScreeningPipeline:
    path = MODELS / "pipeline.joblib"
    if path.exists():
        return joblib.load(path)
    if not (DATA / "burnin_train.csv").exists():
        generate.main(DATA)
    th_path = MODELS / "thresholds.json"
    th = Thresholds.load(th_path) if th_path.exists() else Thresholds()
    return ScreeningPipeline(th).fit(pd.read_csv(DATA / "burnin_train.csv"))


@st.cache_data(show_spinner="Screening parts...")
def screen(df: pd.DataFrame, th: tuple) -> pd.DataFrame:
    pipe = copy.copy(load_pipeline())
    pipe.thresholds = Thresholds(*th)
    return pipe.run(df)


def load_metrics() -> dict | None:
    p = REPORTS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------- sidebar ---
pipe = load_pipeline()
base_th = pipe.thresholds

st.sidebar.title("🔬 Burn-in Screening")
source = st.sidebar.radio("Data", ["Demo: held-out test lots (0–168 h)",
                                   "Demo: early screen (0 h + 24 h only)",
                                   "Upload CSV"])
if source == "Upload CSV":
    up = st.sidebar.file_uploader(
        "CSV with lot_id, part_id, V0, V24, datasheet_max (+ optional V96, V168, parameter, unit)",
        type="csv")
    if up is None:
        st.info("Upload a CSV to screen it, or pick a demo dataset in the sidebar.")
        st.stop()
    raw_df = pd.read_csv(up)
else:
    name = "burnin_test.csv" if source.startswith("Demo: held") else "burnin_test_24h.csv"
    if not (DATA / name).exists():
        generate.main(DATA)
    raw_df = pd.read_csv(DATA / name)

try:
    df = validate_input(raw_df)
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.sidebar.expander("Decision thresholds", expanded=False):
    st.caption("Defaults were tuned by lot-grouped cross-validation for recall ≥ 0.98.")
    z_inspect = st.slider("Lot-relative z → INSPECT", 2.0, 8.0, float(base_th.z_inspect), 0.5)
    z_reject = st.slider("Lot-relative z → REJECT", z_inspect, 12.0, max(float(base_th.z_reject), z_inspect), 0.5)
    if_pct = st.slider("Isolation Forest percentile → INSPECT", 0.90, 1.01, float(base_th.if_pct), 0.005,
                       help="1.01 disables the Isolation Forest layer")
    st.caption("Safety slope and guard band apply only in early screening (before V168 is measured).")
    slope_k = st.slider("Safety slope (k robust-sigma above lot drift)", 2.0, 10.0, float(base_th.slope_k), 0.5)
    guard = st.slider("Guard band (fraction of datasheet max)", 0.5, 1.0, float(base_th.guard_frac), 0.05)
th = (z_inspect, z_reject, if_pct, slope_k, guard)

res = screen(df, th)
has_labels = "label" in res.columns
stage = max(times_available(df))
multi_param = res["parameter"].nunique() > 1
tiny = small_lots(res)
if len(tiny):
    st.warning(f"{len(tiny)} lot(s) have fewer than {MIN_LOT_SIZE} parts, so their lot-relative limits are "
               f"less reliable: " + ", ".join(f"{lot}/{p} ({n} parts)" for (lot, p), n in tiny.items()))
st.sidebar.caption(f"{len(res)} parts · {res['lot_id'].nunique()} lots · read points up to {stage} h · "
                   f"drift model: {pipe.drift_model}")

# -------------------------------------------------------------------- tabs ---
t_over, t_flag, t_part, t_perf = st.tabs(["Overview", "Flagged parts", "Part inspector", "Model performance"])

with t_over:
    counts = res["status"].value_counts()
    static_rej = int(res["trig_static"].sum())
    flagged = int((res["status"] != "PASS").sum())
    c = st.columns(5)
    c[0].metric("Parts screened", f"{len(res):,}")
    c[1].metric(f"{STATUS_ICON['PASS']} PASS", int(counts.get("PASS", 0)))
    c[2].metric(f"{STATUS_ICON['INSPECT']} INSPECT", int(counts.get("INSPECT", 0)))
    c[3].metric(f"{STATUS_ICON['REJECT']} REJECT", int(counts.get("REJECT", 0)))
    c[4].metric("Caught beyond static limits", flagged - static_rej,
                help="Parts that pass the datasheet limit but were flagged by the dynamic / drift models")
    if stage < 168:
        st.info(f"Early-screening mode: only read points up to {stage} h are present. Module A uses the lot "
                f"statistics at those read points and Module B forecasts V168 from V0/V24.")

    by_lot = res.groupby(["lot_id", "status"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for s in ("REJECT", "INSPECT"):
        if s in by_lot:
            fig.add_bar(x=by_lot.index, y=by_lot[s], name=s, marker_color=STATUS_COLORS[s],
                        marker_line=dict(color="rgba(255,255,255,0.9)", width=1),
                        hovertemplate="Lot %{x}<br>" + s + ": %{y} parts<extra></extra>")
    fig.update_layout(barmode="stack", title="Flagged parts per lot (PASS counts in the table below)", height=360,
                      yaxis_title="Flagged parts", xaxis_title="Lot", legend_title_text="Status",
                      margin=dict(t=50, b=40, l=40, r=10))
    st.plotly_chart(fig, width="stretch")

    if multi_param:
        parts = summarize_parts(res)
        st.markdown(f"**Per-part roll-up** ({res['parameter'].nunique()} parameters per part; worst status wins): "
                    + ", ".join(f"{STATUS_ICON[k]} {k} {v}" for k, v in parts["status"].value_counts().items()))
    lot_tab = res.groupby(GROUP_COLS).agg(
        parts=("part_id", "size"), median_V0=("V0", "median"), median_V24=("V24", "median"),
        passed=("status", lambda s: int((s == "PASS").sum())),
        inspect=("status", lambda s: int((s == "INSPECT").sum())),
        reject=("status", lambda s: int((s == "REJECT").sum()))).round(2)
    st.dataframe(lot_tab, width="stretch")

with t_flag:
    c1, c2 = st.columns([1, 2])
    status_sel = c1.multiselect("Status", ["REJECT", "INSPECT", "PASS"], default=["REJECT", "INSPECT"])
    lot_sel = c2.multiselect("Lots", sorted(res["lot_id"].unique()))
    view = res[res["status"].isin(status_sel)]
    if lot_sel:
        view = view[view["lot_id"].isin(lot_sel)]
    cols = ["part_id", "lot_id"] + (["parameter"] if multi_param else []) + ["status", "anomaly_score", "max_lot_z"] + \
           [f"V{t}" for t in TIMES if f"V{t}" in res.columns] + ["V168_pred", "reasons"]
    if has_labels:
        cols.insert(3, "defect_type")
    st.caption(f"{len(view)} parts. Reasons quote the measured value, the lot reference and the threshold.")
    view = view.assign(_sev=view["status"].map({"REJECT": 0, "INSPECT": 1, "PASS": 2}))
    view = view.sort_values(["_sev", "anomaly_score", "max_lot_z"], ascending=[True, False, False])
    num = [c for c in cols if c.startswith("V") and c != "V168_pred"] + ["V168_pred"]
    st.dataframe(view[cols].round({c: 2 for c in num}), width="stretch",
                 hide_index=True, column_config={"reasons": st.column_config.TextColumn(width="large")})
    st.download_button("Download full QA report (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                       "burnin_qa_report.csv", "text/csv")

with t_part:
    order = res.sort_values(["status", "anomaly_score", "max_lot_z"], key=lambda s: s.map(
        {"REJECT": 0, "INSPECT": 1, "PASS": 2}) if s.name == "status" else -s).drop_duplicates("part_id")
    status_of = dict(zip(order["part_id"], order["status"]))   # worst status across parameters
    part_id = st.selectbox("Part (flagged parts listed first)", order["part_id"],
                           format_func=lambda p: f"{p} · {status_of[p]}")
    part_rows = res[res["part_id"] == part_id]
    if len(part_rows) > 1:
        param_sel = st.radio("Parameter", part_rows["parameter"].tolist(), horizontal=True,
                             format_func=lambda p: f"{p} · {part_rows.set_index('parameter').at[p, 'status']}")
        part_rows = part_rows[part_rows["parameter"] == param_sel]
    row = part_rows.iloc[0]
    unit, param = row["unit"], row["parameter"]
    st.subheader(f"{STATUS_ICON[row['status']]} {part_id}: {row['status']}")
    if has_labels:
        st.caption(f"Ground truth (synthetic): {row['defect_type']}")
    for r in row["reasons"].split(SEP):
        st.markdown(f"- {r}")

    lot = res[(res["lot_id"] == row["lot_id"]) & (res["parameter"] == param)]
    ts = [t for t in TIMES if f"V{t}" in res.columns and res[f"V{t}"].notna().all()]
    med, lo, hi = [], [], []
    for t in ts:
        m, s = robust_center_scale(np.log(lot[f"V{t}"]))
        med.append(np.exp(m))
        lo.append(np.exp(m - z_inspect * s))
        hi.append(np.exp(m + z_inspect * s))

    fig = go.Figure()
    sample = lot[lot["part_id"] != part_id].sample(min(60, len(lot) - 1), random_state=0)
    for i, (_, r) in enumerate(sample.iterrows()):
        fig.add_scatter(x=ts, y=[r[f"V{t}"] for t in ts], mode="lines", line=dict(color=C_CONTEXT, width=1),
                        hoverinfo="skip", showlegend=i == 0, name="Other parts in lot (sample)")
    fig.add_scatter(x=ts, y=lo, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=ts, y=hi, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C_BAND,
                    name=f"Lot envelope (median ± {z_inspect:g} robust-sigma)", hoverinfo="skip")
    fig.add_scatter(x=ts, y=med, mode="lines", line=dict(color=C_MEDIAN, width=2, dash="dot"),
                    name="Lot median", hovertemplate="%{x} h · lot median %{y:.2f} " + unit + "<extra></extra>")
    fig.add_scatter(x=ts, y=[row[f"V{t}"] for t in ts], mode="lines+markers", name=part_id,
                    line=dict(color=C_PART, width=2), marker=dict(size=9, line=dict(color="white", width=2)),
                    hovertemplate="%{x} h · " + param + " %{y:.2f} " + unit + "<extra>" + part_id + "</extra>")
    fig.add_scatter(x=[24, 168], y=[row["V24"], row["V168_pred"]], mode="lines", hoverinfo="skip",
                    line=dict(color=C_FORECAST, width=1.5, dash="dot"), showlegend=False)
    fig.add_scatter(x=[168], y=[row["V168_pred"]], mode="markers", name="Forecast V168 from 0 h + 24 h (90% upper)",
                    marker=dict(color=C_FORECAST, size=11, symbol="diamond", line=dict(color="white", width=2)),
                    error_y=dict(type="data", symmetric=False, array=[row["V168_upper"] - row["V168_pred"]],
                                 arrayminus=[0], color=C_FORECAST, thickness=2),
                    hovertemplate="Forecast 168 h: %{y:.2f} " + unit + "<extra></extra>")
    fig.add_hline(y=row["datasheet_max"], line=dict(color=C_LIMIT, width=1.5, dash="dash"),
                  annotation_text=f"Datasheet max {row['datasheet_max']:g} {unit}",
                  annotation_position="top left")
    st.markdown(f"**{param} trajectory vs lot {row['lot_id']}**")
    fig.update_layout(height=480, hovermode="closest",
                      xaxis=dict(title="Burn-in time (h)", tickvals=list(TIMES)),
                      yaxis=dict(title=f"{param} ({unit})", rangemode="tozero"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      margin=dict(t=70, b=40, l=50, r=10))
    st.plotly_chart(fig, width="stretch")
    if row["forecast_confidence"] == "low":
        st.caption("⚠️ Low-confidence forecast: this part's 0–24 h behaviour is far outside its lot, so the "
                   "168 h forecast is an extrapolation. The decision relies on the lot-relative checks above.")

    d = st.columns(4)
    d[0].metric("Max lot-relative z", f"{row['max_lot_z']:.1f}")
    d[1].metric("Isolation Forest percentile", f"{row['iforest_pct'] * 100:.1f}%")
    d[2].metric(f"Forecast V168 ({unit})", f"{row['V168_pred']:.2f}",
                delta=None if "V168" not in res.columns else f"actual {row['V168']:.2f}", delta_color="off")
    d[3].metric("Predicted drift (robust-sigma vs lot)", f"{row['z_pred_drift']:.1f}")

with t_perf:
    if has_labels:
        st.subheader("On the data currently loaded")
        y = res["label"].to_numpy()
        m = classification_metrics(y, (res["status"] != "PASS").to_numpy())
        c = st.columns(4)
        c[0].metric("Recall (defects caught)", f"{m['recall']:.3f}")
        c[1].metric("False negatives (escapes)", m["false_negatives"])
        c[2].metric("Precision", f"{m['precision']:.3f}")
        c[3].metric("False-positive rate", f"{m['false_positive_rate']:.4f}")
        cm = pd.DataFrame([[m["tp"], m["false_negatives"]], [m["fp"], m["tn"]]],
                          index=["Actual defect", "Actual good"], columns=["Flagged", "Passed"])
        c1, c2 = st.columns(2)
        c1.markdown("**Confusion matrix**")
        c1.dataframe(cm, width="stretch")
        defects = res[res["label"] == 1]
        c2.markdown("**Recall by defect type**")
        c2.dataframe(defects.groupby("defect_type")["status"].apply(lambda s: round((s != "PASS").mean(), 3))
                     .rename("recall"), width="stretch")
        if "V168" in res.columns:
            mae = (res["V168_pred"] - res["V168"]).abs()
            st.metric(f"Drift prediction MAE on V168 ({res['unit'].iloc[0]})", f"{mae.mean():.3f}",
                      help="Forecast uses only V0 and V24")
    else:
        st.info("No labels in this data. The saved evaluation results are shown below.")

    metrics = load_metrics()
    if metrics:
        st.subheader("Saved evaluation (held-out test lots)")
        st.markdown("**Detection: ablation of each module**")
        st.dataframe(pd.DataFrame(metrics["test"]["detection"]).T, width="stretch")
        st.markdown("**Recall by defect type**")
        st.dataframe(pd.DataFrame(metrics["test"]["recall_by_type"]).T, width="stretch")
        st.markdown("**V168 forecast error (from V0 + V24)**")
        st.dataframe(pd.DataFrame(metrics["test"]["drift_mae"]).T, width="stretch")
        st.markdown("**Ridge coefficients** (standardised inputs, target = log V168/V0): how the "
                    "interpretable model weighs each input")
        st.dataframe(pd.Series(metrics["ridge_coefficients"], name="coefficient").round(4),
                     width="stretch")
