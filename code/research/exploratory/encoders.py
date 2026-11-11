"""Group-aware hierarchical target encoding (Experiment B) and prediction-time indicators.

Hierarchy: global -> course_category -> course_family -> course_key, with additive smoothing m:
    enc_cat    = (sum_cat + m * global)  / (n_cat + m)
    enc_family = (sum_fam + m * enc_cat)  / (n_fam + m)       (prior = the row's category encoding)
    enc_course = (sum_crs + m * enc_family) / (n_crs + m)
An unseen course falls back to its family, an unseen family to its category, an unseen category to
the global mean - no value is invented beyond the training fold's own statistics.

Group-aware: encodings of TRAINING rows are computed out-of-fold over inner folds grouped by the
same unit as the outer split (student, section, course or family), so a training row never sees
statistics from its own group - exactly the situation of a test row. Test rows are encoded with
statistics from the whole outer training fold. Nothing from the outer test fold is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

LEVELS = ("course_category", "course_family", "course_key")
OUT_COLS = ["hte_cat", "hte_fam", "hte_course", "n_fam_train", "n_course_train"]


def _stats(X: pd.DataFrame, y: np.ndarray) -> dict:
    d = pd.DataFrame({c: X[c].astype(str).to_numpy() for c in LEVELS})
    d["y"] = y
    return {"global": float(np.mean(y)),
            **{c: d.groupby(c)["y"].agg(["sum", "count"]) for c in LEVELS}}


def _encode(X: pd.DataFrame, st: dict, m: float) -> pd.DataFrame:
    def look(col):
        t = st[col]
        keys = X[col].astype(str)
        return keys.map(t["sum"]).fillna(0).to_numpy(float), keys.map(t["count"]).fillna(0).to_numpy(float)

    g = st["global"]
    s_c, n_c = look("course_category")
    s_f, n_f = look("course_family")
    s_k, n_k = look("course_key")
    enc_c = (s_c + m * g) / (n_c + m)
    enc_f = (s_f + m * enc_c) / (n_f + m)
    enc_k = (s_k + m * enc_f) / (n_k + m)
    return pd.DataFrame({"hte_cat": enc_c, "hte_fam": enc_f, "hte_course": enc_k,
                         "n_fam_train": n_f, "n_course_train": n_k}, index=X.index)


class HierarchicalTargetEncoder:
    def __init__(self, m: float = 20.0, group_col: str = "student_id", k_inner: int = 5, seed: int = 42):
        self.m, self.group_col, self.k_inner, self.seed = m, group_col, k_inner, seed

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """Out-of-fold (group-aware) encodings for the training rows; also fits full statistics."""
        y = np.asarray(y, float)
        out = pd.DataFrame(np.nan, index=X.index, columns=OUT_COLS)
        groups = X[self.group_col].astype(str).to_numpy()
        u = np.unique(groups)
        rng = np.random.default_rng(self.seed)
        perm = dict(zip(u, rng.permutation(len(u))))
        k = min(self.k_inner, len(u))
        for tr, va in GroupKFold(n_splits=k).split(X, groups=[perm[g] for g in groups]):
            st = _stats(X.iloc[tr], y[tr])
            out.iloc[va] = _encode(X.iloc[va], st, self.m).to_numpy()
        self.stats_ = _stats(X, y)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return _encode(X, self.stats_, self.m)


def indicators(enc: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Prediction-time indicators: required_elective missing (registration data) and whether the
    course / family occurs in the training data (from the group-aware counts)."""
    return pd.DataFrame({"ind_required_elective_missing": X["required_elective"].isna().astype(float).to_numpy(),
                         "ind_course_seen": (enc["n_course_train"].to_numpy() > 0).astype(float),
                         "ind_family_seen": (enc["n_fam_train"].to_numpy() > 0).astype(float)}, index=X.index)
