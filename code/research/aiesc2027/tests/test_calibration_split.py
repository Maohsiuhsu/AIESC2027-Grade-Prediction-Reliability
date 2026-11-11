"""Tests of the calibration-split follow-up (run in the pinned environment, see README)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.aiesc2027 import calibration_analysis as CA  # noqa: E402
from research.aiesc2027.uncertainty import conformal_quantile  # noqa: E402


def test_split_conformal_marginal_coverage():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(200):
        r_cal, r_te = np.abs(rng.standard_t(3, 500)), np.abs(rng.standard_t(3, 500))
        covs.append(np.mean(r_te <= conformal_quantile(r_cal, 0.1)))
    assert np.mean(covs) >= 0.895


def test_mondrian_uses_group_quantile_and_pooled_fallback():
    score = np.concatenate([np.full(200, 1.0), np.full(200, 5.0), np.full(10, 50.0)])
    g = np.array(["a"] * 200 + ["b"] * 200 + ["c"] * 10)
    q = CA.mondrian_q(score, g, np.array(["a", "b", "c", "d"]), 0.1, min_calib=100)
    pooled = conformal_quantile(score, 0.1)
    assert q[0] == 1.0 and q[1] == 5.0 and q[2] == pooled and q[3] == pooled


def test_cqr_reorders_crossed_quantiles_and_covers():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 4000)
    y = 50 + 20 * x + rng.normal(0, 1 + 5 * x)
    ql, qh = 50 + 20 * x - 3, 50 + 20 * x + 3
    lo, hi = CA.cqr(qh[:2000], ql[:2000], y[:2000], qh[2000:], ql[2000:], 0.1)   # crossed on purpose
    assert (hi >= lo).all()
    assert np.mean((y[2000:] >= lo) & (y[2000:] <= hi)) >= 0.88


def test_calibration_partition_is_student_level_and_seeded():
    from research.aiesc2027.calibration_fit import calibration_partition
    students = np.repeat(np.array([f"s{i:03d}" for i in range(500)]), 3)
    m1, m2 = calibration_partition(students, 0), calibration_partition(students, 0)
    assert np.array_equal(m1, m2)
    assert not np.array_equal(m1, calibration_partition(students, 1))
    per_student = {s: set(m1[students == s]) for s in np.unique(students)}
    assert all(len(v) == 1 for v in per_student.values())
    assert sum(True in v for v in per_student.values()) == 100


def test_predicted_range_clips_out_of_range_means():
    assert list(CA.predicted_range(np.array([-5.0, 59.9, 60.0, 100.0, 104.0]))) == \
        ["0-59", "0-59", "60-69", "90-100", "90-100"]
