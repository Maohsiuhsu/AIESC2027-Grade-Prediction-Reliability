"""Tests of follow-up 2 (risk-oriented rules and objectives; run in the pinned environment, see README)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.aiesc2027 import metrics as M  # noqa: E402
from research.aiesc2027 import risk_analysis as R  # noqa: E402


def test_op1_flags_the_calibration_prevalence():
    rng = np.random.default_rng(0)
    score = rng.normal(size=1000)
    pos = rng.random(1000) < 0.1
    t = R.threshold_op1(score, pos)
    assert (score >= t).sum() == round(pos.mean() * 1000)


def test_op2_is_the_largest_threshold_reaching_target_recall():
    rng = np.random.default_rng(1)
    score = rng.normal(size=2000)
    pos = rng.random(2000) < 0.2
    t = R.threshold_op2(score, pos, 0.5)
    assert (score[pos] >= t).mean() >= 0.5
    sp = score[pos]
    higher = sp[sp > t]
    assert len(higher) == 0 or (sp >= higher.min()).mean() < 0.5


def test_class_weights_balance_the_classes():
    from research.aiesc2027.risk_fit import class_weights
    y = np.array([50.0] * 10 + [80.0] * 90)
    w, w_low = class_weights(y)
    assert abs(w[y < 60].sum() - w[y >= 60].sum()) < 1e-9 and abs(w_low - 9.0) < 1e-12


def test_zero_only_students_assignment_is_seeded_and_excludes_known_students():
    from research.aiesc2027.risk_fit import zero_only_assignment
    sb = np.array(["a", "b", "c", "z1", "z2", "z2"])
    a1, a2 = zero_only_assignment(sb, {"a", "b", "c"}), zero_only_assignment(sb, {"a", "b", "c"})
    assert list(a1["student"]) == ["z1", "z2"] and a1.equals(a2)
    assert a1["fold"].between(0, 4).all()


def test_bootstrap_point_equals_direct_metrics():
    rng = np.random.default_rng(2)
    n = 600
    students = np.repeat(np.arange(200), 3)
    y = rng.uniform(30, 100, n)
    pos = y < 60
    score = -y + rng.normal(0, 15, n)
    flags = {("m", op): score >= np.quantile(score, q) for op, q in (("native", 0.8), ("OP1", 0.7), ("OP2", 0.6))}
    bs = R.RiskBootstrap(students, pos, {"m": score}, flags)
    p = bs.point()["m"]
    assert abs(p["pr_auc"] - M.ranking(pos, score)["pr_auc"]) < 1e-12
    assert abs(p["recall_OP1"] - M.classification(pos, flags[("m", "OP1")])["recall"]) < 1e-12
    reps = bs.run(50, 0)["m"]
    assert len(reps) == 50 and reps["f1_OP2"].between(0, 1).all()
