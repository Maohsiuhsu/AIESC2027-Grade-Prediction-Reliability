"""Unit tests of the AIESC 2027 analysis on synthetic data (no real data needed).

Run: python -m pytest -q research/aiesc2027/tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.aiesc2027 import bootstrap as B  # noqa: E402
from research.aiesc2027 import config as C  # noqa: E402
from research.aiesc2027 import metrics as M  # noqa: E402
from research.aiesc2027 import uncertainty as U  # noqa: E402


@pytest.fixture
def synth():
    rng = np.random.default_rng(0)
    n = 3000
    y = np.clip(np.round(rng.normal(75, 15, n)), 1, 100)
    p = np.clip(np.round(0.5 * y + 37 + rng.normal(0, 8, n)), 0, 100)      # rounded -> many ties
    groups = rng.integers(0, 400, n)
    return y, p, groups


def test_ranking_matches_sklearn_with_ties_and_weights(synth):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p, _ = synth
    pos = y < 60
    w = np.random.default_rng(1).integers(0, 4, len(y)).astype(float)
    r = M.ranking(pos, -p)
    assert r["roc_auc"] == pytest.approx(roc_auc_score(pos, -p), abs=1e-12)
    assert r["pr_auc"] == pytest.approx(average_precision_score(pos, -p), abs=1e-12)
    rw = M.ranking(pos, -p, w)
    assert rw["roc_auc"] == pytest.approx(roc_auc_score(pos, -p, sample_weight=w), abs=1e-12)
    assert rw["pr_auc"] == pytest.approx(average_precision_score(pos, -p, sample_weight=w), abs=1e-12)


def test_classification_matches_sklearn(synth):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    y, p, _ = synth
    a, f = y < 60, p < 60
    c = M.classification(a, f)
    tn, fp, fn, tp = confusion_matrix(a, f).ravel()
    assert (c["TP"], c["FP"], c["FN"], c["TN"]) == (tp, fp, fn, tn)
    assert c["precision"] == pytest.approx(precision_score(a, f))
    assert c["recall"] == pytest.approx(recall_score(a, f))
    assert c["f1"] == pytest.approx(f1_score(a, f))
    assert c["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(a, f))
    assert c["specificity"] == pytest.approx(tn / (tn + fp))


def test_no_flag_gives_not_available_precision():
    c = M.classification(np.array([1, 0, 1], bool), np.zeros(3, bool))
    assert np.isnan(c["precision"]) and c["recall"] == 0.0


def test_regression_and_band_bias(synth):
    y, p, _ = synth
    r = M.regression(y, p)
    assert r["MAE"] == pytest.approx(np.mean(np.abs(p - y)))
    assert r["RMSE"] == pytest.approx(np.sqrt(np.mean((p - y) ** 2)))
    assert r["mean_signed_error"] == pytest.approx(np.mean(p - y))
    t = M.band_bias(y, p, C.GRADE_BANDS)
    assert t["n"].sum() == len(y)
    assert t.loc[t.grade_range == "0-59", "mean_actual"].iat[0] < 60


def test_band_boundaries():
    b = M.band_of(np.array([1, 59, 60, 69, 70, 89, 90, 100]), C.GRADE_BANDS)
    assert list(b) == ["0-59", "0-59", "60-69", "60-69", "70-79", "80-89", "90-100", "90-100"]
    with pytest.raises(ValueError):
        M.band_of(np.array([101.0]), C.GRADE_BANDS)


def test_bootstrap_point_reproducible_and_paired(synth):
    y, p, g = synth
    preds = {"a": p, "b": np.clip(p + 1.5, 0, 100), "same": p.copy()}
    bs = B.PairedClusterBootstrap(g, y, preds, 60.0)
    pt = bs.point()
    assert pt["a"]["MAE"] == pytest.approx(M.regression(y, p)["MAE"])
    assert pt["a"]["recall"] == pytest.approx(M.classification(y < 60, p < 60)["recall"])
    assert pt["a"]["pr_auc"] == pytest.approx(M.ranking(y < 60, -p)["pr_auc"])
    r1, r2 = bs.run(50, 42), bs.run(50, 42)
    pd.testing.assert_frame_equal(r1["a"], r2["a"])
    assert np.allclose(r1["same"]["MAE"], r1["a"]["MAE"])          # identical predictions -> zero difference
    per, diff = B.summarise(pt, r1, "a", ["MAE", "recall"], {"MAE"}, 0.05, {"MAE": 0.5, "recall": 0.05})
    d = diff[(diff.model == "same") & (diff.metric == "MAE")].iloc[0]
    assert d["difference"] == 0 and d["p_boot"] == 1.0


def test_bootstrap_resamples_clusters_not_rows():
    # two clusters with very different errors: cluster resampling must give a wide MAE distribution
    y = np.full(200, 70.0)
    p = np.r_[np.full(100, 70.0), np.full(100, 90.0)]
    g = np.r_[np.zeros(100), np.ones(100)]
    reps = B.PairedClusterBootstrap(g, y, {"m": p}, 60.0).run(400, 0)["m"]["MAE"]
    assert set(np.round(reps.unique(), 6)) <= {0.0, 10.0, 20.0}


def test_holm():
    adj = B.holm([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])


def test_conformal_coverage_and_no_test_label_use():
    rng = np.random.default_rng(3)
    n = 5000
    loc = rng.uniform(50, 90, n)
    scale = rng.uniform(5, 15, n)
    y = loc + scale * rng.standard_normal(n)
    fold = rng.integers(0, 5, n)
    cat = rng.choice(np.array(["x", "y"]), n)
    iv = U.intervals(y, loc, scale, np.ones(n), fold, cat, 0.9, 100)
    for k in ("normal_raw", "cf_absolute", "cf_normalised", "cf_mondrian"):
        lo, hi = iv[k]
        assert abs(np.mean((y >= lo) & (y <= hi)) - 0.9) < 0.02, k
    y2 = y.copy()
    y2[fold == 0] += 1000                                            # corrupt test-fold labels
    iv2 = U.intervals(y2, loc, scale, np.ones(n), fold, cat, 0.9, 100)
    for k in ("cf_absolute", "cf_normalised", "cf_mondrian"):
        assert np.allclose(iv[k][0][fold == 0], iv2[k][0][fold == 0]), k


def test_prob_below_and_scores():
    assert U.prob_below(60, np.array([60.0]), np.array([10.0]))[0] == pytest.approx(0.5)
    y = np.array([50.0, 70.0])
    assert U.nll_normal(y, np.array([50.0, 70.0]), np.array([1.0, 1.0])) == pytest.approx(0.5 * np.log(2 * np.pi))
    assert U.crps_normal(np.array([0.0]), np.array([0.0]), np.array([1.0])) == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi))


def test_fold_consistency_detects_wrong_folds():
    from research.aiesc2027.io_data import FEATURES, fold_consistency
    rng = np.random.default_rng(5)
    n = 2000
    fr = pd.DataFrame({c: (rng.integers(0, 3, n) if i < 2 else np.zeros(n, int)).astype(str)
                       for i, c in enumerate(FEATURES)})       # 9 distinct feature keys, many rows each
    fold = rng.integers(0, 5, n)
    key = fr.astype(str).agg("|".join, axis=1)
    codes = pd.factorize(key)[0]
    pred = codes * 0.37 + fold * 0.011                     # one deterministic model per fold
    ok = fold_consistency(fr, fold, pred)
    assert ok["within_fold_identical"] == 1.0 and ok["across_fold_distinct"] == 1.0
    bad = fold_consistency(fr, rng.permutation(fold), pred)
    assert bad["within_fold_identical"] < 0.5


def test_features_exclude_target_and_derived():
    from research.aiesc2027.io_data import FEATURES
    assert not set(FEATURES) & C.FORBIDDEN_PREDICTORS
    assert "pass_threshold" in C.FORBIDDEN_PREDICTORS and "term_grade" in C.FORBIDDEN_PREDICTORS


def test_reliability_ece():
    p = np.array([0.1] * 10 + [0.9] * 10)
    a = np.array([0] * 9 + [1] + [1] * 9 + [0])
    t, ece = M.reliability(a, p)
    assert ece == pytest.approx(0.0)
    assert t["n"].sum() == 20
