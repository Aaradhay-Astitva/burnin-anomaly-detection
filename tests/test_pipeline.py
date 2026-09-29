import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from burnin.decision import ScreeningPipeline, Thresholds, decide, summarize_parts
from burnin.features import MAD_TO_SIGMA, build_features, robust_z, validate_input
from burnin.generate import generate


@pytest.fixture(scope="module")
def small():
    return validate_input(generate(n_lots=4, parts_per_lot=200, seed=1))


def test_robust_z_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    z = robust_z(x, min_scale=0.0)
    # median 3, MAD 1 -> sigma 1.4826
    assert z[2] == 0.0
    assert z[4] == pytest.approx(97 / MAD_TO_SIGMA)


def test_robust_z_ignores_outlier_in_scale():
    base = np.random.default_rng(0).normal(0, 1, 500)
    z_clean = robust_z(base)
    z_dirty = robust_z(np.append(base, 1e6))[:-1]
    assert np.allclose(z_clean, z_dirty, atol=0.05)


def test_features_are_lot_relative(small):
    # Scaling one whole lot must not change its z-scores (dynamic, not static, limits).
    scaled = small.copy()
    mask = scaled["lot_id"] == "L01"
    for c in ("V0", "V24", "V96", "V168"):
        scaled.loc[mask, c] *= 3.0
    f1, cols = build_features(small)
    f2, _ = build_features(scaled)
    assert np.allclose(f1[cols], f2[cols])


def test_group_kfold_has_no_lot_leakage(small):
    for tr, te in GroupKFold(n_splits=4).split(small, groups=small["lot_id"]):
        assert not set(small.iloc[tr]["lot_id"]) & set(small.iloc[te]["lot_id"])


def test_thresholds_roundtrip(tmp_path):
    th = Thresholds(z_inspect=4.0, z_reject=6.0, if_pct=0.98, slope_k=5.0, guard_frac=0.9)
    th.save(tmp_path / "t.json")
    assert Thresholds.load(tmp_path / "t.json") == th


def test_pipeline_end_to_end(small):
    pipe = ScreeningPipeline().fit(small)
    res = pipe.run(small)
    assert set(res["status"]) <= {"PASS", "INSPECT", "REJECT"}
    assert res["reasons"].str.len().gt(0).all()
    # Every static-limit failure is rejected.
    assert (res.loc[res["V168"] > res["datasheet_max"], "status"] == "REJECT").all()
    assert (res.loc[res["label"] == 1, "status"] != "PASS").mean() > 0.9


def test_early_screen_needs_only_v0_v24(small):
    pipe = ScreeningPipeline().fit(small)
    early = small[["lot_id", "part_id", "V0", "V24", "datasheet_max"]]
    res = pipe.run(early)
    assert len(res) == len(early) and res["V168_pred"].gt(0).all()


def test_forecast_rule_superseded_once_v168_measured():
    raw = pd.DataFrame({"max_z_24": [0.0], "max_z_168": [0.0], "if_pct_24": [0.5], "if_pct_168": [0.5],
                        "static_fail_24": [False], "static_fail_168": [False],
                        "z_pred_drift": [50.0], "upper_margin": [0.95]})
    th = Thresholds()
    assert decide(raw, th)["status"].iloc[0] == "PASS"
    assert decide(raw, th, stages=(24,))["status"].iloc[0] == "REJECT"


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="missing"):
        validate_input(pd.DataFrame({"lot_id": ["a"], "V0": [1.0]}))


def test_missing_v96_still_screens(small):
    pipe = ScreeningPipeline().fit(small)
    res = pipe.run(small.drop(columns=["V96"]))
    assert (res.loc[res["label"] == 1, "status"] != "PASS").mean() > 0.9


def test_second_parameter_in_other_units(small):
    # Same parts, a second parameter at 1000x the scale (e.g. ns instead of uA):
    # all features are lot-relative / ratios, so the screening must be unchanged.
    pipe = ScreeningPipeline().fit(small)
    other = small.copy()
    other["parameter"], other["unit"] = "Tpd", "ns"
    for c in ("V0", "V24", "V96", "V168", "datasheet_max"):
        other[c] = other[c] * 1000
    res = pipe.run(pd.concat([small, other], ignore_index=True))
    a = res[res["parameter"] == "Iddq"]["status"].to_numpy()
    b = res[res["parameter"] == "Tpd"]["status"].to_numpy()
    assert (a == b).all()
    parts = summarize_parts(res)
    assert len(parts) == len(small) and (parts["parameters"] == 2).all()


def test_duplicate_rows_rejected(small):
    with pytest.raises(ValueError, match="duplicate"):
        validate_input(pd.concat([small.head(3), small.head(1)]))


def test_predict_cli(tmp_path, small):
    import joblib
    from burnin import predict
    joblib.dump(ScreeningPipeline().fit(small), tmp_path / "p.joblib")
    small[["lot_id", "part_id", "V0", "V24", "datasheet_max"]].to_csv(tmp_path / "in.csv", index=False)
    predict.main([str(tmp_path / "in.csv"), "-o", str(tmp_path / "out.csv"), "--model", str(tmp_path / "p.joblib")])
    out = pd.read_csv(tmp_path / "out.csv")
    assert len(out) == len(small) and {"status", "V168_pred", "reasons"} <= set(out.columns)
