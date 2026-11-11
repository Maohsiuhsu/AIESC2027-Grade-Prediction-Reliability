"""Candidate estimators for the pre-registered exploratory study (see prereg.json)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold

from research.advanced.common import SPLIT_GROUP
from research.config import COURSE_CATEGORICAL, COURSE_NUMERIC, STUDENT_CATEGORICAL, ResearchConfig
from research.models import make_model

from .encoders import HierarchicalTargetEncoder, indicators

BASE_CAT = COURSE_CATEGORICAL + STUDENT_CATEGORICAL          # identical to the frozen baselines
BASE_NUM = list(COURSE_NUMERIC)
LOW_CARD = ["required_elective", "course_category", "student_college", "student_dept", "division", "program"]
M_GRID = (5.0, 20.0, 100.0)


def _cfg(cfg: ResearchConfig, learner: str, params: dict | None) -> ResearchConfig:
    if not params:
        return cfg
    key = "catboost_params" if learner == "catboost" else "lightgbm_params"
    return ResearchConfig(**{**cfg.__dict__, key: {**getattr(cfg, key), **params}})


class GBM(BaseEstimator, RegressorMixin):
    """Frozen-baseline features with an optional objective override (Experiment C3-C6)."""

    def __init__(self, learner="catboost", params=None, cfg=None):
        self.learner, self.params, self.cfg = learner, params, cfg

    def fit(self, X, y):
        self.model_ = make_model(self.learner, BASE_CAT, BASE_NUM, _cfg(self.cfg, self.learner, self.params))
        self.cols_ = list(dict.fromkeys(BASE_CAT + BASE_NUM + ["course_key"]))
        self.model_.fit(X[self.cols_].reset_index(drop=True), np.asarray(y, float))
        return self

    def predict(self, X):
        return self.model_.predict(X[self.cols_].reset_index(drop=True))


class HTEModel(BaseEstimator, RegressorMixin):
    """Experiment B: hierarchical target encodings (B1-B3) or prediction-time indicators (B4-B5).

    mode: "replace"  course_key categorical replaced by encodings (B1, B2)
          "add"      course_key categorical kept, encodings added (B3)
          "indicators" frozen features + indicators only (B4, B5)
    m is selected inside the outer training fold by 3-fold inner grouped CV (modes replace/add),
    unless `fixed_m` is given."""

    def __init__(self, learner="lightgbm", mode="replace", split="student", cfg=None, fixed_m=None):
        self.learner, self.mode, self.split, self.cfg, self.fixed_m = learner, mode, split, cfg, fixed_m

    def _cols(self):
        cat = [c for c in BASE_CAT if not (self.mode == "replace" and c == "course_key")]
        num = BASE_NUM + (["hte_cat", "hte_fam", "hte_course", "n_fam_train", "n_course_train"]
                          if self.mode in ("replace", "add") else
                          ["ind_required_elective_missing", "ind_course_seen", "ind_family_seen"])
        return cat, num

    def _augment(self, X, enc):
        extra = enc if self.mode in ("replace", "add") else indicators(enc, X)
        return pd.concat([X.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)

    def _fit_once(self, X, y, m):
        g = SPLIT_GROUP[self.split]
        enc = HierarchicalTargetEncoder(m, g, 5, (self.cfg or ResearchConfig()).seed)
        e_tr = enc.fit_transform(X, y)
        cat, num = self._cols()
        model = make_model(self.learner, cat, num, self.cfg or ResearchConfig())
        cols = list(dict.fromkeys(cat + num + ["course_key"]))
        model.fit(self._augment(X, e_tr)[cols], np.asarray(y, float))
        return enc, model, cols

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, float)
        if self.fixed_m is not None or self.mode == "indicators":
            m_sel, scores = (self.fixed_m or 20.0), {}
        else:
            g = X[SPLIT_GROUP[self.split]].astype(str).to_numpy()
            u = np.unique(g)
            perm = dict(zip(u, np.random.default_rng((self.cfg or ResearchConfig()).seed).permutation(len(u))))
            folds = list(GroupKFold(n_splits=3).split(X, groups=[perm[v] for v in g]))
            scores = {}
            for m in M_GRID:
                err = 0.0
                for tr, va in folds:
                    enc, model, cols = self._fit_once(X.iloc[tr].reset_index(drop=True), y[tr], m)
                    Xv = X.iloc[va].reset_index(drop=True)
                    err += np.abs(np.clip(model.predict(self._augment(Xv, enc.transform(Xv))[cols]), 0, 100) - y[va]).sum()
                scores[m] = err / len(y)
            m_sel = min(scores, key=scores.get)
        self.enc_, self.model_, self.cols_ = self._fit_once(X, y, m_sel)
        self.fit_info_ = {"m_selected": m_sel, "inner_mae_by_m": {str(k): round(v, 4) for k, v in scores.items()}}
        return self

    def predict(self, X):
        X = X.reset_index(drop=True)
        return self.model_.predict(self._augment(X, self.enc_.transform(X))[self.cols_])


class EBM(BaseEstimator, RegressorMixin):
    def __init__(self, seed=42, outer_bags=4, n_jobs=32):
        self.seed, self.outer_bags, self.n_jobs = seed, outer_bags, n_jobs

    def _X(self, X):
        Z = X[BASE_CAT + BASE_NUM].copy().reset_index(drop=True)
        for c in BASE_CAT:
            Z[c] = Z[c].astype("string").fillna("__MISSING__").astype(str)
        return Z

    def fit(self, X, y):
        from interpret.glassbox import ExplainableBoostingRegressor
        self.model_ = ExplainableBoostingRegressor(outer_bags=self.outer_bags, random_state=self.seed, n_jobs=self.n_jobs)
        self.model_.fit(self._X(X), np.asarray(y, float))
        return self

    def predict(self, X):
        return self.model_.predict(self._X(X))


class NGB(BaseEstimator, RegressorMixin):
    """NGBoost (Normal). Inputs: group-aware HTE (m=20), ordinal codes, numerics - all in-fold."""

    def __init__(self, split="student", seed=42, n_estimators=400, lr=0.05, cfg=None):
        self.split, self.seed, self.n_estimators, self.lr, self.cfg = split, seed, n_estimators, lr, cfg

    def _matrix(self, X, enc_df):
        cols = []
        for c in LOW_CARD:
            cols.append(X[c].astype("string").fillna("__MISSING__").map(self.codes_[c]).fillna(-1).to_numpy(float))
        num = X[BASE_NUM].apply(pd.to_numeric, errors="coerce").fillna(self.median_).to_numpy(float)
        return np.column_stack(cols + [num, enc_df.to_numpy(float)])

    def fit(self, X, y):
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from sklearn.tree import DecisionTreeRegressor
        X = X.reset_index(drop=True)
        y = np.asarray(y, float)
        g = X[SPLIT_GROUP[self.split]].astype(str).to_numpy()
        ug = np.unique(g)
        rng = np.random.default_rng(self.seed)
        va = np.isin(g, rng.choice(ug, size=max(1, len(ug) // 10), replace=False))
        Xt = X[~va].reset_index(drop=True)
        self.codes_ = {c: {v: i for i, v in enumerate(sorted(Xt[c].astype("string").fillna("__MISSING__").unique()))}
                       for c in LOW_CARD}
        self.median_ = Xt[BASE_NUM].apply(pd.to_numeric, errors="coerce").median()
        self.enc_ = HierarchicalTargetEncoder(20.0, SPLIT_GROUP[self.split], 5, self.seed)
        e_tr = self.enc_.fit_transform(Xt, y[~va])
        Xv = X[va].reset_index(drop=True)
        self.model_ = NGBRegressor(Dist=Normal, Base=DecisionTreeRegressor(max_depth=4, random_state=self.seed),
                                   n_estimators=self.n_estimators, learning_rate=self.lr, random_state=self.seed,
                                   verbose=False)
        self.model_.fit(self._matrix(Xt, e_tr), y[~va], X_val=self._matrix(Xv, self.enc_.transform(Xv)),
                        Y_val=y[va], early_stopping_rounds=20)
        self.fit_info_ = {"best_iteration": int(getattr(self.model_, "best_val_loss_itr", -1) or -1)}
        return self

    def predict(self, X):
        X = X.reset_index(drop=True)
        return self.model_.predict(self._matrix(X, self.enc_.transform(X)))

    def predict_interval(self, X, level=0.9):
        from scipy.stats import norm
        X = X.reset_index(drop=True)
        d = self.model_.pred_dist(self._matrix(X, self.enc_.transform(X)))
        z = norm.ppf(0.5 + level / 2)
        return d.loc - z * d.scale, d.loc + z * d.scale


def factories(cfg: ResearchConfig) -> dict:
    """name -> callable(split) -> estimator, in the pre-registered execution order."""
    from .nn import OfficialNN
    return {
        "C3_catboost_mae": lambda s: GBM("catboost", {"loss_function": "MAE"}, cfg),
        "C4_catboost_huber": lambda s: GBM("catboost", {"loss_function": "Huber:delta=10"}, cfg),
        "C5_lightgbm_l1": lambda s: GBM("lightgbm", {"objective": "l1"}, cfg),
        "C6_lightgbm_huber": lambda s: GBM("lightgbm", {"objective": "huber", "alpha": 10.0}, cfg),
        "B4_missing_lightgbm": lambda s: HTEModel("lightgbm", "indicators", s, cfg),
        "B5_missing_catboost": lambda s: HTEModel("catboost", "indicators", s, cfg),
        "B1_hte_lightgbm": lambda s: HTEModel("lightgbm", "replace", s, cfg),
        "B2_hte_catboost": lambda s: HTEModel("catboost", "replace", s, cfg),
        "B3_hte_plus_cat_lightgbm": lambda s: HTEModel("lightgbm", "add", s, cfg),
        "C1_ebm": lambda s: EBM(cfg.seed, 2 if cfg.quick else 4),
        "C2_ngboost": lambda s: NGB(s, cfg.seed, 40 if cfg.quick else 400, cfg=cfg),
        "A1_tabm": lambda s: OfficialNN("tabm", SPLIT_GROUP[s], cfg.seed, 2 if cfg.quick else 30),
        "A2_ft_transformer": lambda s: OfficialNN("ft", SPLIT_GROUP[s], cfg.seed, 2 if cfg.quick else 30),
    }
