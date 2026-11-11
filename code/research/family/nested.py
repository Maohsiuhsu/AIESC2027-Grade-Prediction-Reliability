"""Nested grouped cross-validation.

`NestedSelector` receives ONLY the outer training fold. Inside it:
  1. grouped inner folds (same grouping unit as the outer split: families stay together);
  2. every candidate (model x feature set) is scored by inner-fold MAE with default parameters;
  3. the inner winner, if tunable, gets an Optuna TPE search on the same inner folds;
     tuned parameters are kept only if they beat the default inner score;
  4. the selected configuration is refitted on the whole outer training fold.
The outer test fold is never seen, so outer performance cannot influence any selection.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold

from research.advanced.common import SPLIT_GROUP
from research.advanced.course_repr import RepModel
from research.advanced.ensemble import search_space


def inner_split(X: pd.DataFrame, split: str, k: int, seed: int):
    g = X[SPLIT_GROUP[split]].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    u = np.unique(g)
    perm = dict(zip(u, rng.permutation(len(u))))
    return list(GroupKFold(n_splits=k).split(X, groups=[perm[v] for v in g]))


# candidate name -> (learner, text, keep_id); "ridge" handled by RidgeCat
CANDIDATES = {
    "catboost:cat": ("catboost", (), True),
    "lightgbm:cat": ("lightgbm", (), True),
    "catboost:cat+tfidf+embed": ("catboost", ("tfidf", "embed"), True),
    "catboost:tfidf+embed": ("catboost", ("tfidf", "embed"), False),
    "catboost:embed": ("catboost", ("embed",), False),
    "lightgbm:tfidf": ("lightgbm", ("tfidf",), False),
    "lightgbm:tfidf+embed": ("lightgbm", ("tfidf", "embed"), False),
}


def build(name: str, cfg, params: dict | None = None):
    learner, text, keep_id = CANDIDATES[name]
    return RepModel(learner, text, keep_id, cfg=cfg, params=params)


def _inner_mae(name, params, X, y, folds, cfg) -> float:
    err = 0.0
    for tr, va in folds:
        m = build(name, cfg, params).fit(X.iloc[tr], y[tr])
        err += np.abs(np.clip(m.predict(X.iloc[va]), 0, 100) - y[va]).sum()
    return float(err / len(y))


class NestedSelector(BaseEstimator, RegressorMixin):
    def __init__(self, split="family", candidates=tuple(CANDIDATES), inner_k=3, n_trials=6, cfg=None, seed=42):
        self.split, self.candidates, self.inner_k, self.n_trials, self.cfg, self.seed = \
            split, candidates, inner_k, n_trials, cfg, seed

    def fit(self, X: pd.DataFrame, y):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        X = X.reset_index(drop=True)
        y = np.asarray(y, float)
        folds = inner_split(X, self.split, self.inner_k, self.seed)
        t0 = time.time()
        scores = {c: _inner_mae(c, None, X, y, folds, self.cfg) for c in self.candidates}
        t_sel = time.time() - t0
        best = min(scores, key=scores.get)
        params, tuned_score, trials = None, None, []
        t1 = time.time()
        if self.n_trials:
            learner = CANDIDATES[best][0]
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed))
            study.optimize(lambda tr: _inner_mae(best, search_space(tr, learner), X, y, folds, self.cfg),
                           n_trials=self.n_trials)
            trials = [round(t.value, 4) for t in study.trials if t.value is not None]
            if study.best_value < scores[best]:
                params, tuned_score = study.best_params, study.best_value
        t_tune = time.time() - t1
        self.selected_, self.params_ = best, params
        self.model_ = build(best, self.cfg, params).fit(X, y)
        self.fit_info_ = {"inner_scores": {k: round(v, 4) for k, v in scores.items()}, "selected": best,
                          "tuned": params is not None, "tuned_params": params,
                          "default_inner_mae": round(scores[best], 4),
                          "tuned_inner_mae": round(tuned_score, 4) if tuned_score is not None else None,
                          "trial_values": trials, "seconds_selection": round(t_sel, 1),
                          "seconds_tuning": round(t_tune, 1), "inner_folds": self.inner_k}
        return self

    def predict(self, X):
        return self.model_.predict(X)
