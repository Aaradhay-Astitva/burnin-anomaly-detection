"""Static presentation figures (PNG) from the held-out test results.

Usage:  python -m burnin.figures      (run `python -m burnin.evaluate` first)
Writes reports/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import TIMES  # noqa: E402
from .features import robust_center_scale  # noqa: E402

# Reference palette (light surface).
BLUE, ORANGE, GRAY, LIGHT_GRAY = "#2a78d6", "#eb6834", "#898781", "#c3c2b7"
INK, INK2, GRID, CRITICAL, BAND = "#0b0b0b", "#52514e", "#e1e0d9", "#d03b3b", "#cde2fb"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
    "axes.edgecolor": LIGHT_GRAY, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": GRAY, "ytick.color": GRAY, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2,
})

PRETTY = {"lot_outlier": "Lot outlier", "latent_linear": "Latent linear drift",
          "accelerating": "Accelerating wear-out", "step_jump": "Step jump at 96 h",
          "erratic": "Erratic", "gross_fail": "Gross fail"}


def _envelope(lot: pd.DataFrame, z: float):
    med, lo, hi = [], [], []
    for t in TIMES:
        m, s = robust_center_scale(np.log(lot[f"V{t}"]))
        med.append(np.exp(m))
        lo.append(np.exp(m - z * s))
        hi.append(np.exp(m + z * s))
    return med, lo, hi


def fig_static_vs_dynamic(res: pd.DataFrame, z: float, out: Path) -> None:
    part = res[(res["defect_type"] == "lot_outlier")].sort_values("V24").iloc[-1]
    lot = res[res["lot_id"] == part["lot_id"]]
    m, s = robust_center_scale(np.log(lot["V24"]))
    dyn = np.exp(m + z * s)
    lim = part["datasheet_max"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.hist(lot["V24"], bins=np.linspace(0, lim * 1.1, 90), color=BLUE, edgecolor="#fcfcfb", linewidth=0.6)
    ymax = ax.get_ylim()[1]
    ax.axvline(lim, color=CRITICAL, lw=2, ls="--")
    ax.text(lim, ymax * 0.95, f" Static datasheet limit\n {lim:g} µA", color=CRITICAL, va="top", fontsize=10)
    ax.axvline(dyn, color=INK, lw=2)
    ax.text(dyn, ymax * 0.62, f" Dynamic lot limit\n {dyn:.1f} µA\n (median + {z:g} robust-σ)",
            color=INK, va="top", fontsize=10)
    ax.annotate(f"{part['part_id']}\n{part['V24']:.1f} µA → passes static limit,\n"
                f"but {part['max_lot_z']:.1f} robust-σ above its lot",
                xy=(part["V24"], 1), xytext=(dyn + lim * 0.02, ymax * 0.33),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5), color=INK, fontsize=10)
    ax.plot(part["V24"], 1, "o", color=ORANGE, ms=10, mec="white", mew=2, zorder=5)
    ax.set_title(f"Static vs dynamic limits: lot {part['lot_id']}, Iddq at 24 h ({len(lot)} parts)")
    ax.set_xlabel("Iddq at 24 h (µA)")
    ax.set_ylabel("Parts")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_drift_signatures(res: pd.DataFrame, z: float, out: Path) -> None:
    kinds = ["lot_outlier", "latent_linear", "accelerating", "step_jump", "erratic", "gross_fail"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, kind in zip(axes.flat, kinds):
        part = res[res["defect_type"] == kind].iloc[0]
        lot = res[res["lot_id"] == part["lot_id"]]
        ctx = lot[lot["label"] == 0].sample(40, random_state=0)
        for _, r in ctx.iterrows():
            ax.plot(TIMES, [r[f"V{t}"] for t in TIMES], color=LIGHT_GRAY, lw=0.7, alpha=0.6)
        med, lo, hi = _envelope(lot, z)
        ax.fill_between(TIMES, lo, hi, color=BAND, alpha=0.7, lw=0)
        ax.plot(TIMES, med, color=GRAY, lw=1.5, ls=":")
        ax.plot(TIMES, [part[f"V{t}"] for t in TIMES], color=BLUE, lw=2.2, marker="o", ms=6,
                mec="white", mew=1.5, zorder=5)
        ax.plot(168, part["V168_pred"], "D", color=ORANGE, ms=8, mec="white", mew=1.5, zorder=6)
        ax.axhline(part["datasheet_max"], color=CRITICAL, lw=1.2, ls="--")
        ax.set_title(f"{PRETTY[kind]} → {part['status']}", fontsize=11)
        tag = part["reasons"].split(":")[0].title()
        ax.text(0.98, 0.04, f"Flagged by: {tag} ({part['max_lot_z']:.0f} robust-σ)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=INK2,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec=GRID))
        ax.set_xticks(TIMES)
        ax.set_ylim(0, max(part["datasheet_max"] * 1.1, max(part[f"V{t}"] for t in TIMES) * 1.1))
    for ax in axes[1]:
        ax.set_xlabel("Burn-in time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Iddq (µA)")
    handles = [plt.Line2D([], [], color=BLUE, lw=2.2, marker="o", label="Defective part"),
               plt.Line2D([], [], color=LIGHT_GRAY, lw=1, label="Good parts, same lot"),
               plt.Rectangle((0, 0), 1, 1, color=BAND, label=f"Lot envelope (± {z:g} robust-σ)"),
               plt.Line2D([], [], color=ORANGE, marker="D", lw=0, label="168 h forecast from 0 h + 24 h"),
               plt.Line2D([], [], color=CRITICAL, ls="--", label="Datasheet max")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Defect signatures vs their lot: only the gross fail breaks the static limit "
                 "(band = level envelope; drift is scored per interval)",
                 x=0.01, ha="left", fontweight="bold", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_recall(metrics: dict, out: Path) -> None:
    det = metrics["test"]["detection"]
    rows = [("Static datasheet limits (today)", "Static datasheet limits (baseline)"),
            ("Early screen at 24 h (A + B)", "Early screen @24h (static + A + B)"),
            ("Full run at 168 h (A)", next(k for k in det if k.startswith("Full run")))]
    labels = [r[0] for r in rows][::-1]
    rec = [det[r[1]]["recall"] for r in rows][::-1]
    fn = [int(det[r[1]]["false_negatives"]) for r in rows][::-1]
    fpr = [det[r[1]]["false_positive_rate"] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    bars = ax.barh(labels, rec, color=[BLUE, BLUE, LIGHT_GRAY], height=0.55)   # rows are reversed: full, early, static
    for b, r, f, p in zip(bars, rec, fn, fpr):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{r:.0%} caught · {f} escaped · {p:.1%} false alarms", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, 1.55)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Recall: share of defective parts caught (held-out lots)")
    n_def = det[rows[0][1]]["tp"] + det[rows[0][1]]["false_negatives"]
    ax.set_title(f"Defects caught on {int(n_def)} defective parts in unseen lots")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_mae(metrics: dict, out: Path) -> None:
    mae = metrics["test"]["drift_mae"]
    names = {"naive_linear_extrapolation": "Naive line through V0, V24", "ridge": "Ridge (interpretable)",
             "rf": "Random Forest", "gbm": "Gradient Boosting"}
    order = sorted(names, key=lambda k: -mae[k]["MAE_all"])
    chosen = metrics.get("drift_model")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    vals = [mae[k]["MAE_all"] for k in order]
    bars = ax.barh([names[k] + ("  ◀ used" if k == chosen else "") for k in order], vals,
                   color=[BLUE if k == chosen else LIGHT_GRAY for k in order], height=0.55)
    for b, v, k in zip(bars, vals, order):
        ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.2f} µA  (good parts {mae[k]['MAE_good_parts']:.2f})", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean absolute error of the V168 forecast (µA), held-out lots")
    ax.set_title("Module B: forecasting Iddq at 168 h from the 0 h and 24 h readings")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_forecast_scatter(res: pd.DataFrame, out: Path) -> None:
    good, bad = res[res["label"] == 0], res[res["label"] == 1]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(good["V168"], good["V168_pred"], s=8, color=BLUE, alpha=0.35, lw=0, label="Good parts")
    ax.scatter(bad["V168"], bad["V168_pred"], s=22, color=ORANGE, edgecolor="white", lw=0.6,
               label="Defective parts")
    lo, hi = res[["V168", "V168_pred"]].min().min() * 0.9, res[["V168", "V168_pred"]].max().max() * 1.1
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=1, ls="--", label="Perfect forecast")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual Iddq at 168 h (µA)")
    ax.set_ylabel("Forecast from 0 h + 24 h (µA)")
    ax.set_title("Forecast vs actual, held-out lots")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main(report_dir="reports") -> None:
    report_dir = Path(report_dir)
    res = pd.read_csv(report_dir / "test_results.csv")
    metrics = json.loads((report_dir / "metrics.json").read_text())
    z = metrics["thresholds"]["z_inspect"]
    out = report_dir / "figures"
    out.mkdir(exist_ok=True)
    fig_static_vs_dynamic(res, z, out / "1_static_vs_dynamic.png")
    fig_drift_signatures(res, z, out / "2_defect_signatures.png")
    fig_recall(metrics, out / "3_recall_by_screen.png")
    fig_mae(metrics, out / "4_forecast_mae.png")
    fig_forecast_scatter(res, out / "5_forecast_vs_actual.png")
    print(f"Wrote 5 figures to {out.resolve()}")


if __name__ == "__main__":
    main()
