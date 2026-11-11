"""Predictive uncertainty of NGBoost (Normal) on the student split, from the saved per-fold loc / scale.

Interval methods, all evaluated on the SAME 83,494 out-of-fold rows at the SAME nominal levels:
  normal_raw         loc +/- z * scale                          (as saved; reproduces the IR study)
  normal_rescaled    loc +/- z * scale * m_k, m_k estimated on the 10% grouped validation part of the
                     outer TRAINING fold k (saved by the IR study; no test label involved)
  cf_absolute        cross-fold conformal, |y - loc| quantile from the other four folds (constant width)
  cf_normalised      cross-fold conformal, |y - loc| / scale quantile from the other four folds
  cf_mondrian        cf_absolute with the quantile computed per course category (>= 100 rows, else global)
All intervals are clipped to [0, 100], as in the IR study.

The cross-fold conformal variants are POST-HOC recalibrations: the quantile for test fold k uses only
rows of folds != k (student-disjoint from fold k, so no test label enters its own threshold), but the
residuals of fold j were produced by a model whose training data included fold k. There is no separate
calibration set, so the finite-sample split-conformal guarantee does not apply (cross-conformal setting,
coverage guarantee only approximately 1 - 2*alpha). Nothing is tuned on these results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import metrics as M


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Same finite-sample quantile as research/family/intervals.py."""
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[k - 1]) if n else np.inf


def nll_normal(y, loc, scale) -> float:
    return float(-np.mean(norm.logpdf(y, loc, scale)))


def crps_normal(y, loc, scale) -> float:
    z = (np.asarray(y) - loc) / scale
    return float(np.mean(scale * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def intervals(y, loc, scale, mult, fold, groups, level: float, min_calib: int) -> dict[str, tuple]:
    a = 1 - level
    z = norm.ppf(0.5 + level / 2)
    out = {"normal_raw": (loc - z * scale, loc + z * scale),
           "normal_rescaled": (loc - z * scale * mult, loc + z * scale * mult)}
    abs_res, norm_res = np.abs(y - loc), np.abs(y - loc) / scale
    lo_a, hi_a, lo_n, hi_n, lo_m, hi_m = (np.full(len(y), np.nan) for _ in range(6))
    for k in np.unique(fold):
        te, cal = fold == k, fold != k
        qa, qn = conformal_quantile(abs_res[cal], a), conformal_quantile(norm_res[cal], a)
        lo_a[te], hi_a[te] = loc[te] - qa, loc[te] + qa
        lo_n[te], hi_n[te] = loc[te] - qn * scale[te], loc[te] + qn * scale[te]
        for g in np.unique(groups[te]):
            cg = cal & (groups == g)
            q = conformal_quantile(abs_res[cg], a) if cg.sum() >= min_calib else qa
            t = te & (groups == g)
            lo_m[t], hi_m[t] = loc[t] - q, loc[t] + q
    out.update({"cf_absolute": (lo_a, hi_a), "cf_normalised": (lo_n, hi_n), "cf_mondrian": (lo_m, hi_m)})
    return {k: (np.clip(lo, 0, 100), np.clip(hi, 0, 100)) for k, (lo, hi) in out.items()}


def interval_table(y, loc, scale, mult, fold, course_cat, bands, levels, min_calib) -> pd.DataFrame:
    band = M.band_of(y, bands)
    rows = []
    for lv in levels:
        for method, (lo, hi) in intervals(y, loc, scale, mult, fold, course_cat, lv, min_calib).items():
            cover = (y >= lo) & (y <= hi)
            width = hi - lo
            strata = [("all", "all", np.ones(len(y), bool))]
            strata += [("grade_range", b, band == b) for b, _, _ in bands]
            strata += [("course_category", c, course_cat == c) for c in sorted(np.unique(course_cat))]
            for by, g, m in strata:
                cov = float(cover[m].mean())
                rows.append({"method": method, "nominal": lv, "group_by": by, "group": g, "n": int(m.sum()),
                             "coverage": cov, "coverage_error": cov - lv,
                             "mean_width": float(width[m].mean()), "median_width": float(np.median(width[m]))})
    return pd.DataFrame(rows)


def scoring_table(y, loc, scale, mult) -> pd.DataFrame:
    return pd.DataFrame([
        {"version": "normal_raw", "NLL": nll_normal(y, loc, scale), "CRPS": crps_normal(y, loc, scale)},
        {"version": "normal_rescaled", "NLL": nll_normal(y, loc, scale * mult), "CRPS": crps_normal(y, loc, scale * mult)},
    ])


def pit(y, loc, scale) -> np.ndarray:
    return norm.cdf((np.asarray(y) - loc) / scale)


def prob_below(threshold, loc, scale) -> np.ndarray:
    """P(Y < threshold) under the NGBoost Normal predictive distribution (threshold may be per row)."""
    return norm.cdf((np.asarray(threshold, float) - loc) / scale)
