"""Reusable scikit-learn pipeline: type-safe preprocessing + CatBoostRegressor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

MISSING_CATEGORY = "__MISSING__"


class CatBoostFramePreprocessor(BaseEstimator, TransformerMixin):
    """Selects and orders feature columns and fixes their dtypes for CatBoost.

    * numeric: coerced to float; NaN is kept and handled natively by CatBoost (`nan_mode`),
      so no value is invented. Missingness is also explicit via features such as
      `has_history` and the `*_coverage` columns.
    * categorical: cast to str; missing values become an explicit `__MISSING__` level
      (e.g. an unknown instructor is its own category, not a guessed instructor).
    """

    def __init__(self, numeric_features: list[str], categorical_features: list[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y=None):
        missing = [c for c in self.numeric_features + self.categorical_features if c not in X]
        if missing:
            raise ValueError(f"Feature columns missing from input: {missing}")
        self.feature_names_out_ = np.array(self.numeric_features + self.categorical_features)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num = X[self.numeric_features].apply(pd.to_numeric, errors="coerce").astype(float)
        cat = X[self.categorical_features].astype("string").fillna(MISSING_CATEGORY).astype(str)
        return pd.concat([num, cat], axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def build_pipeline(numeric: list[str], categorical: list[str], params: dict,
                   random_seed: int = 42) -> Pipeline:
    model = CatBoostRegressor(cat_features=categorical, random_seed=random_seed, **params)
    return Pipeline([("prep", CatBoostFramePreprocessor(numeric, categorical)), ("model", model)])


def fit_with_validation(pipe: Pipeline, X_train, y_train, X_val, y_val) -> Pipeline:
    """Fit with early stopping on the (chronologically later) validation set."""
    prep = pipe.named_steps["prep"].fit(X_train)
    pipe.fit(X_train, y_train, model__eval_set=(prep.transform(X_val), y_val),
             model__use_best_model=True)
    return pipe


def predict_grades(pipe: Pipeline, X: pd.DataFrame, lo: float = 0.0, hi: float = 100.0) -> np.ndarray:
    return np.clip(pipe.predict(X), lo, hi)


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    model: CatBoostRegressor = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    return (pd.DataFrame({"feature": names, "importance": model.get_feature_importance()})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def save_pipeline(pipe: Pipeline, model_dir: Path, spec: dict, extra: dict | None = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "grade_pipeline.joblib")
    pipe.named_steps["model"].save_model(str(model_dir / "catboost_grade_model.cbm"))
    (model_dir / "feature_spec.json").write_text(
        json.dumps({**spec, **(extra or {})}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pipeline(model_dir: Path) -> tuple[Pipeline, dict]:
    pipe = joblib.load(model_dir / "grade_pipeline.joblib")
    spec = json.loads((model_dir / "feature_spec.json").read_text(encoding="utf-8"))
    return pipe, spec
