"""Unit tests of the decision-reliability helpers (tie-aware budgets, strata, budget tables); synthetic data only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.aiesc2027 import decision_reliability as D  # noqa: E402


def test_capture_counts_tied_cutoff_fractionally():
    score = np.array([3, 2, 2, 2, 1, 0, 0, 0, 0, 0], float)
    pos = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], float)
    fold = np.zeros(10, int)
    fl, cap, P = D.FoldBlocks(score, fold).capture(pos, np.ones(10), [0.2, 0.4])
    assert P == 3
    np.testing.assert_allclose(fl, [2, 4])
    # 20%: record 0 plus 1 of the 3 tied records (1 positive among 3) -> 1 + 1/3
    np.testing.assert_allclose(cap, [1 + 1 / 3, 2])


def test_capture_is_per_fold_and_full_budget_recalls_all():
    rng = np.random.default_rng(0)
    n = 400
    score = rng.integers(0, 15, n).astype(float)
    pos = (rng.random(n) < 0.2).astype(float)
    fold = rng.integers(0, 5, n)
    fb = D.FoldBlocks(score, fold)
    fl, cap, P = fb.capture(pos, np.ones(n), [1.0])
    assert fl[0] == pytest.approx(n)
    assert cap[0] == pytest.approx(P)
    fl, cap, _ = fb.capture(pos, np.ones(n), [0.1])
    per_fold = sum(D.FoldBlocks(score[fold == k], np.zeros((fold == k).sum(), int)).capture(
        pos[fold == k], np.ones((fold == k).sum()), [0.1])[1][0] for k in range(5))
    assert cap[0] == pytest.approx(per_fold)
    assert fl[0] == pytest.approx(0.1 * n)


def test_capture_matches_expected_value_of_random_tie_breaking():
    rng = np.random.default_rng(1)
    score = rng.integers(0, 6, 60).astype(float)
    pos = (rng.random(60) < 0.3).astype(float)
    fold = np.zeros(60, int)
    _, cap, _ = D.FoldBlocks(score, fold).capture(pos, np.ones(60), [0.25])
    k = 15
    sims = []
    for _ in range(4000):
        order = np.lexsort((rng.random(60), -score))
        sims.append(pos[order[:k]].sum())
    assert cap[0] == pytest.approx(np.mean(sims), abs=0.05)


def test_capture_with_bootstrap_weights_equals_replicated_data():
    score = np.array([5, 4, 4, 3, 2, 1], float)
    pos = np.array([1, 0, 1, 0, 1, 0], float)
    w = np.array([2, 1, 0, 3, 1, 1], float)
    fl, cap, P = D.FoldBlocks(score, np.zeros(6, int)).capture(pos, w, [0.5])
    rep = np.repeat(np.arange(6), w.astype(int))
    fl2, cap2, P2 = D.FoldBlocks(score[rep], np.zeros(len(rep), int)).capture(pos[rep], np.ones(len(rep)), [0.5])
    np.testing.assert_allclose([fl[0], cap[0], P], [fl2[0], cap2[0], P2])


def test_strata_keep_ties_together_and_follow_shares():
    score = np.r_[np.arange(100, 0, -1), np.zeros(100)].astype(float)
    fold = np.zeros(200, int)
    st = D.strata_of(score, fold)
    assert (np.diff(st[:100]) >= 0).all()
    assert len(set(st[score == 0])) == 1
    np.testing.assert_array_equal(np.bincount(st[:100], minlength=5), [10, 10, 20, 40, 20])
    assert st[0] == 0 and st[-1] == 4


def test_strata_are_computed_within_each_fold():
    score = np.r_[np.linspace(1, 0, 50), np.linspace(101, 100, 50)]
    fold = np.r_[np.zeros(50, int), np.ones(50, int)]
    st = D.strata_of(score, fold)
    np.testing.assert_array_equal(st[:50], st[50:])


def test_budget_point_lift_and_records_per_find():
    rng = np.random.default_rng(2)
    n = 1000
    y = rng.uniform(30, 100, n)
    fold = rng.integers(0, 5, n)
    s = D.Score("m", "M", "original_oof", -y + rng.normal(0, 10, n), "ranking", reference=True)
    rows, curve = D.budget_point(y, fold, [s])
    prev = (y < 60).mean()
    np.testing.assert_allclose(rows.lift, rows.precision / prev)
    np.testing.assert_allclose(rows.records_per_low_grade_found, 1 / rows.precision)
    np.testing.assert_allclose(rows.low_grade_captured + rows.low_grade_missed, rows.low_grade_total)
    assert curve.recall.is_monotonic_increasing
    assert curve.recall.iloc[-1] == pytest.approx(1.0)


def test_budget_tables_contrast_is_paired_difference():
    rng = np.random.default_rng(3)
    n = 600
    y = rng.uniform(30, 100, n)
    fold = rng.integers(0, 5, n)
    students = rng.integers(0, 150, n).astype(str)
    a = D.Score("a", "A", "original_oof", -y + rng.normal(0, 5, n), "ranking", reference=True)
    b = D.Score("b", "B", "original_oof", -y + rng.normal(0, 25, n), "ranking")
    r = D.Score("r", "R", "refit_80", -y, "ranking", reference=True)
    pt, _ = D.budget_point(y, fold, [a, b, r])
    reps = D.budget_bootstrap(y, fold, students, [a, b, r], n_boot=40, seed=1)
    tab, con = D.budget_tables(pt, reps, [a, b, r])
    c = con[con.score_key == "b"]
    assert len(c) == len(D.BUDGETS)
    np.testing.assert_allclose(c.recall_difference, c.recall - c.reference_recall)
    assert (tab.recall_ci_low <= tab.recall_ci_high).all()
    assert set(con.configuration) == {"original_oof"}
