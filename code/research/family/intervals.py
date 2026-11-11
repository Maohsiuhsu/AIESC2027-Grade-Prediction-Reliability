"""Prediction intervals with a held-out calibration subset inside each outer training fold.

The outer training fold is split by GROUP (families stay together) into a fitting part (80%) and a
calibration part (20%). Models are fitted on the fitting part only; conformity scores come from the
calibration part only; the outer test fold is used solely to measure coverage and width.

Methods (nominal coverage 1 - alpha):
  split      absolute-residual split conformal: pred +/- q
  mondrian   the same, with q computed separately per course category (>= min_calib rows,
             otherwise the global q)
  cqr        conformalised quantile regression (LightGBM quantile models, Romano et al. 2019)
Coverage guarantees assume exchangeability of groups; with course families that is an
approximation, which is why coverage is reported per course group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from research.advanced.course_repr import RepModel

from .nested import inner_split


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[k - 1]) if n else np.inf


class ConformalIntervals(BaseEstimator):
    def __init__(self, split="family", alphas=(0.1, 0.2), calib_frac=0.2, min_calib=100, learner="catboost",
                 text=("tfidf", "embed"), keep_id=False, cfg=None, seed=42):
        self.split, self.alphas, self.calib_frac, self.min_calib = split, alphas, calib_frac, min_calib
        self.learner, self.text, self.keep_id, self.cfg, self.seed = learner, text, keep_id, cfg, seed

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, float)
        k = max(2, int(round(1 / self.calib_frac)))
        fit_idx, cal_idx = inner_split(X, self.split, k, self.seed + 7)[0]   # one grouped hold-out
        self.calib_rows_, self.fit_rows_ = X["row_id"].iloc[cal_idx].tolist(), X["row_id"].iloc[fit_idx].tolist()
        Xf, yf, Xc, yc = X.iloc[fit_idx], y[fit_idx], X.iloc[cal_idx], y[cal_idx]
        self.point_ = RepModel(self.learner, tuple(self.text), self.keep_id, cfg=self.cfg).fit(Xf, yf)
        res = np.abs(yc - np.clip(self.point_.predict(Xc), 0, 100))
        cat = Xc["course_category"].astype(str).to_numpy()
        self.q_, self.q_cat_, self.cqr_ = {}, {}, {}
        for a in self.alphas:
            self.q_[a] = conformal_quantile(res, a)
            self.q_cat_[a] = {c: conformal_quantile(res[cat == c], a) for c in np.unique(cat)
                              if (cat == c).sum() >= self.min_calib}
            lo = RepModel("lightgbm", tuple(self.text), self.keep_id, cfg=self.cfg,
                          params={"objective": "quantile", "alpha": a / 2}).fit(Xf, yf)
            hi = RepModel("lightgbm", tuple(self.text), self.keep_id, cfg=self.cfg,
                          params={"objective": "quantile", "alpha": 1 - a / 2}).fit(Xf, yf)
            s = np.maximum(lo.predict(Xc) - yc, yc - hi.predict(Xc))
            self.cqr_[a] = (lo, hi, conformal_quantile(s, a))
        self.fit_info_ = {"fit_rows": len(fit_idx), "calib_rows": len(cal_idx),
                          "q": {str(a): round(v, 3) for a, v in self.q_.items()}}
        return self

    def predict(self, X):
        return self.point_.predict(X)

    def intervals(self, X: pd.DataFrame) -> dict:
        X = X.reset_index(drop=True)
        p = np.clip(self.point_.predict(X), 0, 100)
        cat = X["course_category"].astype(str).to_numpy()
        out = {}
        for a in self.alphas:
            q = self.q_[a]
            out[("split", a)] = (np.clip(p - q, 0, 100), np.clip(p + q, 0, 100))
            qc = np.array([self.q_cat_[a].get(c, q) for c in cat])
            out[("mondrian", a)] = (np.clip(p - qc, 0, 100), np.clip(p + qc, 0, 100))
            lo, hi, qq = self.cqr_[a]
            out[("cqr", a)] = (np.clip(lo.predict(X) - qq, 0, 100), np.clip(hi.predict(X) + qq, 0, 100))
        return out
