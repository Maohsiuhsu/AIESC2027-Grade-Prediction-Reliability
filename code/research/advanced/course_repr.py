"""Course representations and the representation-aware model (Phase 1).

`CourseText` turns normalised course names into dense vectors:
  * "tfidf": Chinese character 1-3-gram TF-IDF -> TruncatedSVD
  * "embed": frozen multilingual sentence embedding (cached) -> PCA
The vocabulary, IDF weights, SVD and PCA are fitted on the UNIQUE course names of the training
fold only; test names are only transformed. Names are registration-time information and carry no
grades; the frozen embedding model is never trained on this data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from research.config import (COURSE_CATEGORICAL, COURSE_NUMERIC, STUDENT_CATEGORICAL,
                             ResearchConfig)
from research.models import make_model

from .embed_courses import DEFAULT_MODEL, cache_path

_EMB_CACHE: dict[str, dict[str, np.ndarray]] = {}


def load_embeddings(model: str = DEFAULT_MODEL) -> dict[str, np.ndarray]:
    if model not in _EMB_CACHE:
        z = np.load(cache_path(model))
        _EMB_CACHE[model] = dict(zip(z["names"].tolist(), z["vectors"]))
    return _EMB_CACHE[model]


class CourseText(BaseEstimator, TransformerMixin):
    def __init__(self, methods: tuple[str, ...] = ("tfidf",), n_components: int = 32, seed: int = 42,
                 embed_model: str = DEFAULT_MODEL):
        self.methods, self.n_components, self.seed, self.embed_model = methods, n_components, seed, embed_model

    def fit(self, names: pd.Series, y=None):
        uniq = sorted(pd.Series(names).dropna().unique())
        self.n_train_names_ = len(uniq)
        k = max(1, min(self.n_components, len(uniq) - 1))
        if "tfidf" in self.methods:
            self.tfidf_ = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True).fit(uniq)
            self.svd_ = TruncatedSVD(n_components=min(k, len(self.tfidf_.vocabulary_) - 1),
                                     random_state=self.seed).fit(self.tfidf_.transform(uniq))
        if "embed" in self.methods:
            emb = load_embeddings(self.embed_model)
            missing = [n for n in uniq if n not in emb]
            if missing:
                raise KeyError(f"{len(missing)} course names missing from the embedding cache; rerun embed_courses.py")
            self.pca_ = PCA(n_components=k, random_state=self.seed).fit(np.stack([emb[n] for n in uniq]))
        return self

    def transform(self, names: pd.Series) -> pd.DataFrame:
        names = pd.Series(names).reset_index(drop=True)
        uniq = names.unique()
        blocks = []
        if "tfidf" in self.methods:
            v = self.svd_.transform(self.tfidf_.transform(uniq))
            blocks.append(pd.DataFrame(v, index=uniq, columns=[f"tf_{i}" for i in range(v.shape[1])]))
        if "embed" in self.methods:
            emb = load_embeddings(self.embed_model)
            dim = next(iter(emb.values())).shape[0]
            # a name absent from the cache stays missing (NaN); it is never replaced by a guess
            raw = np.stack([emb.get(n, np.full(dim, np.nan)) for n in uniq])
            ok = ~np.isnan(raw).any(axis=1)
            v = np.full((len(uniq), self.pca_.n_components_), np.nan)
            if ok.any():
                v[ok] = self.pca_.transform(raw[ok])
            blocks.append(pd.DataFrame(v, index=uniq, columns=[f"em_{i}" for i in range(v.shape[1])]))
        table = pd.concat(blocks, axis=1)
        return table.loc[names.to_numpy()].reset_index(drop=True)


def feature_lists(keep_id: bool) -> tuple[list[str], list[str]]:
    course_cat = COURSE_CATEGORICAL if keep_id else [c for c in COURSE_CATEGORICAL if c != "course_key"]
    return course_cat + STUDENT_CATEGORICAL, list(COURSE_NUMERIC)


class RepModel(BaseEstimator, RegressorMixin):
    """Course text representation (fitted in-fold) + CatBoost / LightGBM from research.models."""

    def __init__(self, learner: str = "catboost", text: tuple[str, ...] = (), keep_id: bool = True,
                 n_components: int = 32, cfg: ResearchConfig | None = None, params: dict | None = None):
        self.learner, self.text, self.keep_id = learner, text, keep_id
        self.n_components, self.cfg, self.params = n_components, cfg, params

    def _cfg(self) -> ResearchConfig:
        cfg = self.cfg or ResearchConfig()
        if self.params:
            key = "catboost_params" if self.learner == "catboost" else "lightgbm_params"
            cfg = ResearchConfig(**{**cfg.__dict__, key: {**getattr(cfg, key), **self.params}})
        return cfg

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        base = X.reset_index(drop=True)
        if not self.text:
            return base
        return pd.concat([base, self.text_.transform(base["course_norm"])], axis=1)

    def fit(self, X: pd.DataFrame, y):
        cfg = self._cfg()
        if self.text:
            self.text_ = CourseText(tuple(self.text), self.n_components, cfg.seed).fit(X["course_norm"])
        Z = self._frame(X)
        self.categorical_, numeric = feature_lists(self.keep_id)
        self.numeric_ = numeric + [c for c in Z.columns if c.startswith(("tf_", "em_"))]
        self.model_ = make_model(self.learner, self.categorical_, self.numeric_, cfg)
        cols = list(dict.fromkeys(self.categorical_ + self.numeric_ + ["course_key"]))
        self.cols_ = cols
        self.model_.fit(Z[cols], np.asarray(y, float))
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(self._frame(X)[self.cols_])
