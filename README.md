# AI-Driven Anomaly Detection in Component Burn-In & Screening

Static pass/fail limits miss **latent defects**: parts that stay under the datasheet maximum but
behave abnormally *for their lot* or drift abnormally during burn-in. This project flags those parts
with two modules, a risk decision layer and a plain-language reason for every flag.

```
data ──► features (per-lot median/MAD, log space)
            ├─► Module A: static limit + robust z-score + Isolation Forest   (dynamic outliers)
            └─► Module B: V0, V24 ──► forecast V168 + safety slope           (early rejection)
                         └─► Risk decision layer: PASS / INSPECT / REJECT + reason codes
                                   └─► Streamlit dashboard (Plotly drift-envelope plots)
```

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r requirements.txt && pip install -e .
python -m burnin.generate      # synthetic lots -> data/
python -m burnin.evaluate      # CV + tuning + held-out test -> models/, reports/
python -m burnin.figures       # presentation PNGs -> reports/figures/
streamlit run app/streamlit_app.py
pytest
```

Screen any CSV with the trained model (writes status, V168 forecast and reasons per part):

```bash
python -m burnin.predict path/to/lots.csv -o reports/predictions.csv
```

## Data format

One row per part: `lot_id, part_id, V0, V24, datasheet_max` are required. `V96`, `V168`,
`parameter` (default `Iddq`) and `unit` (default `uA`) are optional. `label` and `defect_type` are
used only for scoring. A file with only V0/V24 runs in **early-screening mode**.

- **Several parameters per part** (e.g. Iddq, leakage, propagation delay) are supported as one row
  per part and parameter. Each parameter is judged against its own lot. `burnin.predict` and the
  dashboard also give a per-part roll-up, where the worst status wins.
- **Different read-point sets** (e.g. no V96) are handled: Module A is refit, unsupervised, on the
  lots being screened.
- **Lots with fewer than 30 parts** produce a warning.

No dataset was provided, so `burnin.generate` simulates lots with log-normal baselines that differ
from lot to lot. Six defect signatures are injected into about 4% of parts: `gross_fail`,
`lot_outlier`, `latent_linear`, `accelerating`, `step_jump` and `erratic`. Every signature except
`gross_fail` stays under the datasheet limit. The model is trained on 20 lots, and all headline
numbers come from 8 **separately generated, unseen lots**.

## How it works

**Module A: dynamic outlier detection** (`src/burnin/module_a_outlier.py`)
- Each part is scored against its own lot at every read point: robust z = (x − lot median) / (1.4826·MAD), computed in log space.
- The same scoring is applied to every drift interval (for example 0→24h and 24→96h) and to drift curvature.
- A 45 µA part in a 10 µA lot scores ≈7σ, even though the datasheet limit is 50 µA.
- An Isolation Forest on the lot-normalised severities catches odd trajectory *shapes*. Because it only sees normalised inputs, it works on new lots.

**Module B: drift predictor** (`src/burnin/module_b_drift.py`)
- Forecasts the relative drift log(V168/V0) from the 0→24h drift and lot-relative z-scores.
- Every input is a ratio or a lot-relative z-score, so the model is **scale-free**: it transfers across parameters, units and lot baselines. When Iddq readings were rescaled ×100, the error stayed at about 3%, whereas a model on absolute values was off by about 90%.
- Ridge, Random Forest and Gradient Boosting are compared with lot-grouped CV. The one with the lowest MAE is used (currently GBM).
- A GBM quantile model gives a 90% upper bound.
- The part is rejected early if its predicted drift slope exceeds the **lot safety slope** (lot median + k robust-σ). It is sent to inspection if the upper bound enters the guard band (a set fraction of the datasheet max).
- The forecast drives decisions **only until V168 is measured**. After that, Module A judges the real trajectory.
- If a part's 0–24h behaviour is itself far outside its lot, its forecast is an extrapolation and is marked `forecast_confidence = low`.

**Decision layer** (`src/burnin/decision.py`)
- Thresholds are tuned by GroupKFold CV (grouped by lot, so there is no leakage).
- Module A thresholds: recall ≥ 0.98 at the lowest false-positive rate.
- Module B thresholds: best F2 at the 24h screen.

**Explainability** (`src/burnin/explain.py`)
- Every flag quotes the measured value, the lot reference and the threshold, for example:
  > LOT OUTLIER: Iddq@0h = 39.7 uA is 6.9 robust-sigma above lot T01 median (11.0 uA); still under datasheet max 50 uA, so a static limit would pass it.
  > EARLY REJECT (drift forecast): predicted Iddq@168h = 34.7 uA from 0h/24h readings; drift slope 0.129 uA/h exceeds lot safety slope 0.030 uA/h.
- Ridge coefficients are published, and the dashboard plots each part against its lot envelope.

## Results (held-out test: 8 unseen lots, 4,000 parts, 170 defects)

| Screen | Recall | Missed defects | False-positive rate |
|---|---|---|---|
| Static datasheet limits (baseline) | 0.171 | 141 | 0.000 |
| Early screen @24h (static + A + B) | 0.612 | 66 | 0.007 |
| Full run @168h (static + A) | **1.000** | **0** | 0.001 |

- **V168 forecast MAE:** 0.43 µA (GBM), 0.45 µA (Random Forest), 0.49 µA (Ridge), 4.33 µA (naive linear extrapolation).
- On good parts the forecast MAE is 0.19 µA.
- Presentation figures are in `reports/figures/`.
- The full tables are in `reports/evaluate_log.txt` and `reports/metrics.json`.

## Honest limitations

- **The data is synthetic.** Its defect signatures are clean, so full-run recall of 1.0 is optimistic. Expect lower numbers on real burn-in data, and re-run `burnin.evaluate` on it to re-tune the thresholds.
- **Some defects cannot be forecast from V0/V24.** `step_jump` (appears at 96h) and most `accelerating` parts look normal at 24h, which caps early-screen recall at about 0.64. These parts are caught by Module A once the 96h/168h readings exist.
- **Robust lot statistics need a reasonable lot size**, roughly 30 parts or more.

## Project layout

```
src/burnin/   generate · features · module_a_outlier · module_b_drift · decision · explain
              evaluate (CV, tuning, held-out test) · predict (CLI) · figures (PNG)
app/          Streamlit dashboard
tests/        pytest suite
data/         generated CSVs          models/   trained pipeline + thresholds
reports/      metrics, per-part results, figures
```
