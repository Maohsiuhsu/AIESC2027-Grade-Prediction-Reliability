"""Shared machinery for the advanced research phases.

* Data: the same pseudonymised 113-2 frame as the earlier research (zero grades excluded as the
  primary policy), plus target-free course-name columns.
* Folds: IDENTICAL to research/validation.py (student / section / course grouped, same seed), plus a
  stricter supplementary course-family split used only in Phase 1.
* Every fit is recorded in a JSONL registry (success or failure, seed, runtime, versions).
"""
from __future__ import annotations

import json
import platform
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from research.config import TARGET, ResearchConfig
from research.data import build_frame, load_cleaned
from research.validation import SPLITS3, folds3, paired_bootstrap

from .names import course_family, normalize_name

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "research" / "advanced"
PRIMARY_SPLITS = ["student", "section", "course"]
SPLIT_GROUP = {**SPLITS3, "family": "course_family", "family_linked": "family_linked"}

# Phase gate: a candidate "measurably improves" on a baseline when the paired group-bootstrap CI
# of MAE(baseline) - MAE(candidate) lies above 0 AND the point estimate is at least this large.
MIN_GAIN = 0.05


def versions() -> dict:
    import catboost
    import lightgbm
    import sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
         "sklearn": sklearn.__version__, "catboost": catboost.__version__, "lightgbm": lightgbm.__version__}
    try:
        import optuna
        v["optuna"] = optuna.__version__
    except Exception:  # noqa: BLE001
        pass
    return v


class Registry:
    """Append-only JSONL log of every experiment, failed or successful."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.versions = versions()

    def log(self, **rec) -> dict:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec, "versions": self.versions}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec


def load_frames(cfg: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load_cleaned(cfg)
    primary, meta = build_frame(df, cfg)
    with_zeros, _ = build_frame(df, replace(cfg, exclude_zero_grades=False))
    for f in (primary, with_zeros):
        f["course_norm"] = f["course_key"].map(normalize_name)
        f["course_family"] = f["course_key"].map(course_family)
    return primary, with_zeros, meta


def get_folds(frame: pd.DataFrame, split: str, cfg: ResearchConfig):
    if split in SPLITS3:
        return folds3(frame, split, cfg)          # identical to the earlier validation experiments
    groups = frame[SPLIT_GROUP[split]].astype(str)
    rng = np.random.default_rng(cfg.seed)
    uniq = groups.unique()
    perm = dict(zip(uniq, rng.permutation(len(uniq))))
    return list(GroupKFold(n_splits=cfg.n_folds).split(frame, groups=groups.map(perm)))


def inner_folds(frame: pd.DataFrame, idx: np.ndarray, split: str, k: int, seed: int):
    """Grouped folds INSIDE an outer training fold (same grouping unit as the outer split)."""
    groups = frame[SPLIT_GROUP[split]].astype(str).to_numpy()[idx]
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = dict(zip(uniq, rng.permutation(len(uniq))))
    return [(idx[a], idx[b]) for a, b in GroupKFold(n_splits=k).split(idx, groups=[perm[g] for g in groups])]


def cross_validate(frame: pd.DataFrame, split: str, make_estimator, cfg: ResearchConfig, reg: Registry,
                   phase: str, name: str, fold_list=None, extra: dict | None = None) -> dict:
    """Out-of-fold predictions; a fresh estimator is built and fitted per outer training fold."""
    fold_list = fold_list or get_folds(frame, split, cfg)
    y = frame[TARGET].to_numpy()
    oof = np.full(len(frame), np.nan)
    t0 = time.time()
    fold_info = []
    try:
        for k, (tr, te) in enumerate(fold_list):
            tf = time.time()
            est = make_estimator(tr)
            est.fit(frame.iloc[tr], y[tr])
            oof[te] = np.clip(est.predict(frame.iloc[te]), 0, 100)
            fold_info.append({"fold": k, "seconds": round(time.time() - tf, 2),
                              **(getattr(est, "fit_info_", None) or {})})
    except Exception as e:  # noqa: BLE001
        reg.log(phase=phase, experiment=name, split=split, status="failed", seed=cfg.seed,
                seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=5), **(extra or {}))
        return {"status": "failed", "oof": None, "error": str(e)}
    mae = float(np.mean(np.abs(y - oof)))
    rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    rec = reg.log(phase=phase, experiment=name, split=split, status="success", seed=cfg.seed, n=len(frame),
                  folds=len(fold_list), MAE=round(mae, 4), RMSE=round(rmse, 4),
                  seconds=round(time.time() - t0, 1), fold_info=fold_info, **(extra or {}))
    return {"status": "success", "oof": oof, "MAE": mae, "RMSE": rmse, "seconds": rec["seconds"]}


def contrast(frame: pd.DataFrame, split: str, oof_base, oof_cand, seed: int) -> dict:
    r = paired_bootstrap(frame, oof_base, oof_cand, SPLIT_GROUP[split], seed=seed)
    r["improves"] = bool(r["ci_low"] > 0 and r["mae_diff"] >= MIN_GAIN)
    return r
