"""Phase 3 ensemble and Phase 4 tuning. Both only ever see the OUTER TRAINING fold.

OOFStackingEnsemble: inside the outer training fold, grouped inner folds produce out-of-fold
predictions of every base model; non-negative weights (+ intercept) are fitted on those inner OOF
predictions; the base models are then refitted on the whole outer training fold.

TunedModel: an Optuna TPE study whose objective is the inner grouped-CV MAE on the outer training
fold. The outer test fold is never passed in, so it cannot influence the chosen parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression

from research.config import TARGET

from .common import SPLIT_GROUP


def _inner(X: pd.DataFrame, split: str, k: int, seed: int):
    from sklearn.model_selection import GroupKFold
    g = X[SPLIT_GROUP[split]].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    u = np.unique(g)
    perm = dict(zip(u, rng.permutation(len(u))))
    return list(GroupKFold(n_splits=k).split(X, groups=[perm[v] for v in g]))


class OOFStackingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, bases: dict | None = None, split: str = "student", inner_k: int = 3, seed: int = 42):
        self.bases, self.split, self.inner_k, self.seed = bases, split, inner_k, seed

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, float)
        names = list(self.bases)
        Z = np.full((len(X), len(names)), np.nan)
        for tr, va in _inner(X, self.split, self.inner_k, self.seed):
            for j, n in enumerate(names):
                m = clone(self.bases[n]).fit(X.iloc[tr], y[tr])
                Z[va, j] = m.predict(X.iloc[va])
        self.stack_ = LinearRegression(positive=True).fit(Z, y)
        self.fitted_ = {n: clone(self.bases[n]).fit(X, y) for n in names}
        self.fit_info_ = {"weights": dict(zip(names, np.round(self.stack_.coef_, 4).tolist())),
                          "intercept": round(float(self.stack_.intercept_), 4),
                          "inner_oof_mae": {n: round(float(np.mean(np.abs(Z[:, j] - y))), 4) for j, n in enumerate(names)}}
        return self

    def predict(self, X: pd.DataFrame):
        Z = np.column_stack([self.fitted_[n].predict(X) for n in self.bases])
        return self.stack_.predict(Z)


def search_space(trial, learner: str) -> dict:
    if learner == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True)}
    return {"iterations": trial.suggest_int("iterations", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5, log=True)}


class TunedModel(BaseEstimator, RegressorMixin):
    """`factory(params) -> estimator`. Tuning uses only the data passed to fit()."""

    def __init__(self, factory=None, learner: str = "lightgbm", split: str = "student", n_trials: int = 20,
                 inner_k: int = 3, seed: int = 42, timeout: float | None = None):
        self.factory, self.learner, self.split = factory, learner, split
        self.n_trials, self.inner_k, self.seed, self.timeout = n_trials, inner_k, seed, timeout

    def fit(self, X: pd.DataFrame, y):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        X = X.reset_index(drop=True)
        y = np.asarray(y, float)
        folds = _inner(X, self.split, self.inner_k, self.seed)

        def objective(trial):
            params = search_space(trial, self.learner)
            errs = []
            for tr, va in folds:
                m = self.factory(params).fit(X.iloc[tr], y[tr])
                errs.append(np.mean(np.abs(np.clip(m.predict(X.iloc[va]), 0, 100) - y[va])) * len(va))
            return float(np.sum(errs) / len(X))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params_ = study.best_params
        self.model_ = self.factory(self.best_params_).fit(X, y)
        self.fit_info_ = {"best_params": self.best_params_, "best_inner_mae": round(study.best_value, 4),
                          "trials": len(study.trials),
                          "trial_values": [round(t.value, 4) if t.value is not None else None for t in study.trials]}
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(X)
