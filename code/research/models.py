"""Model factories. Every model is an sklearn-compatible estimator taking a DataFrame."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_pipeline.model import CatBoostFramePreprocessor  # noqa: E402  (reused, not modified)

from .config import ResearchConfig  # noqa: E402


class GroupMeanBaseline(BaseEstimator, RegressorMixin):
    """Mean of the target per `group_col` on the training fold; global mean when unseen."""

    def __init__(self, group_col: str | None = None):
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y, float), index=X.index)
        self.global_mean_ = float(y.mean())
        self.means_ = y.groupby(X[self.group_col].astype(str)).mean().to_dict() if self.group_col else {}
        return self

    def predict(self, X: pd.DataFrame):
        if not self.group_col:
            return np.full(len(X), self.global_mean_)
        return X[self.group_col].astype(str).map(self.means_).fillna(self.global_mean_).to_numpy()


def _dense_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """One-hot (rare levels pooled) + median-imputed, scaled numerics with missing indicators."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
    ], sparse_threshold=1.0)


class _CastCategorical(BaseEstimator):
    """Cast categorical columns to str with an explicit missing level (for the one-hot models)."""

    def __init__(self, categorical: list[str]):
        self.categorical = categorical

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for c in self.categorical:
            X[c] = X[c].astype("string").fillna("__MISSING__").astype(str)
        return X


class LightGBMFrame(BaseEstimator, RegressorMixin):
    """LightGBM with pandas categorical dtype; NaN handled natively."""

    def __init__(self, categorical: list[str], numeric: list[str], params: dict, seed: int = 42):
        self.categorical, self.numeric, self.params, self.seed = categorical, numeric, params, seed

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in self.numeric:
            out[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        for c in self.categorical:
            vals = X[c].astype("string").fillna("__MISSING__").astype(str)
            cats = self.categories_[c] if hasattr(self, "categories_") else None
            out[c] = pd.Categorical(vals, categories=cats)
        return out

    def fit(self, X, y):
        self.categories_ = {c: sorted(X[c].astype("string").fillna("__MISSING__").astype(str).unique())
                            for c in self.categorical}
        self.model_ = LGBMRegressor(random_state=self.seed, **self.params)
        self.model_.fit(self._prep(X), np.asarray(y, float), categorical_feature=self.categorical)
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X))


def make_model(name: str, categorical: list[str], numeric: list[str], cfg: ResearchConfig):
    if name == "mean_global":
        return GroupMeanBaseline(None)
    if name == "mean_course":
        return GroupMeanBaseline("course_key")
    if name == "ridge":
        return Pipeline([("cast", _CastCategorical(categorical)), ("prep", _dense_preprocessor(categorical, numeric)),
                         ("model", Ridge(alpha=cfg.ridge_alpha))])
    if name == "mlp":
        # target standardised for the network; predictions are mapped back to the grade scale
        mlp = TransformedTargetRegressor(regressor=MLPRegressor(random_state=cfg.seed, **cfg.mlp_params),
                                         transformer=StandardScaler())
        return Pipeline([("cast", _CastCategorical(categorical)), ("prep", _dense_preprocessor(categorical, numeric)),
                         ("model", mlp)])
    if name == "catboost":
        return Pipeline([("prep", CatBoostFramePreprocessor(numeric, categorical)),
                         ("model", CatBoostRegressor(cat_features=categorical, random_seed=cfg.seed, **cfg.catboost_params))])
    if name == "lightgbm":
        return LightGBMFrame(categorical, numeric, cfg.lightgbm_params, cfg.seed)
    raise ValueError(name)
