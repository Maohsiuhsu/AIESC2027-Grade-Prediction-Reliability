"""Grouped cross-validation experiments: models x feature sets x split strategies."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import (FAST_MODELS, FEATURE_SETS, MAIN_FEATURE_SET, MIN_GROUP_ROWS, MODELS, SPLITS,
                     TARGET, ResearchConfig)
from .models import make_model

GROUP_COL = {"student": "student_id", "section": "section_key"}


def mae_rmse(y, p) -> tuple[float, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p))), float(np.sqrt(np.mean((y - p) ** 2)))


def folds(frame: pd.DataFrame, split: str, cfg: ResearchConfig):
    """Deterministic grouped folds: no student (or section) appears in both train and test."""
    groups = frame[GROUP_COL[split]].astype(str)
    rng = np.random.default_rng(cfg.seed)
    uniq = groups.unique()
    perm = dict(zip(uniq, rng.permutation(len(uniq))))     # shuffle groups deterministically
    return list(GroupKFold(n_splits=cfg.n_folds).split(frame, groups=groups.map(perm)))


def cross_validate(frame: pd.DataFrame, split: str, feature_set: str, model: str, cfg: ResearchConfig) -> dict:
    fs = FEATURE_SETS[feature_set]
    cols = fs["categorical"] + fs["numeric"] + (["course_key"] if "course_key" not in fs["categorical"] else [])
    X, y = frame[cols], frame[TARGET].to_numpy()
    oof = np.full(len(frame), np.nan)
    t0 = time.time()
    fold_scores = []
    for tr, te in folds(frame, split, cfg):
        m = make_model(model, fs["categorical"], fs["numeric"], cfg)
        m.fit(X.iloc[tr], y[tr])
        p = np.clip(m.predict(X.iloc[te]), 0, 100)
        oof[te] = p
        fold_scores.append(mae_rmse(y[te], p))
    mae, rmse = mae_rmse(y, oof)
    return {"split": split, "feature_set": feature_set, "model": model, "n": int(len(frame)),
            "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAE_fold_std": round(float(np.std([f[0] for f in fold_scores])), 4),
            "seconds": round(time.time() - t0, 1), "oof": oof}


def per_group(frame: pd.DataFrame, oof: dict[str, np.ndarray], group_col: str) -> pd.DataFrame:
    rows = []
    for g, idx in frame.groupby(group_col).indices.items():
        if len(idx) < MIN_GROUP_ROWS:
            continue
        y = frame[TARGET].to_numpy()[idx]
        r = {"group_by": group_col, "group": g, "n": int(len(idx)), "mean_grade": round(float(y.mean()), 2)}
        for name, p in oof.items():
            mae, rmse = mae_rmse(y, p[idx])
            r[f"MAE_{name}"], r[f"RMSE_{name}"] = round(mae, 3), round(rmse, 3)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def run_all(frame: pd.DataFrame, frame_with_zeros: pd.DataFrame | None, cfg: ResearchConfig,
            log=print) -> dict:
    results, oofs = [], {}
    # 1) model comparison on the main feature set, both splits
    for split in SPLITS:
        for model in MODELS:
            r = cross_validate(frame, split, MAIN_FEATURE_SET, model, cfg)
            log(f"  {split:8s} {MAIN_FEATURE_SET:28s} {model:12s} MAE {r['MAE']:.3f} RMSE {r['RMSE']:.3f} ({r['seconds']}s)")
            oofs[(split, model)] = r.pop("oof")
            results.append({**r, "experiment": "model_comparison", "target_policy": "zeros_excluded"})
    # 2) feature ablation with the fast models, both splits
    for split in SPLITS:
        for feature_set in FEATURE_SETS:
            for model in FAST_MODELS:
                if feature_set == MAIN_FEATURE_SET:
                    continue  # already run
                r = cross_validate(frame, split, feature_set, model, cfg)
                log(f"  {split:8s} {feature_set:28s} {model:12s} MAE {r['MAE']:.3f} RMSE {r['RMSE']:.3f} ({r['seconds']}s)")
                r.pop("oof")
                results.append({**r, "experiment": "feature_ablation", "target_policy": "zeros_excluded"})
    # 3) sensitivity: zero grades included
    if frame_with_zeros is not None:
        for split in SPLITS:
            for model in ("mean_global", "mean_course", "catboost"):
                r = cross_validate(frame_with_zeros, split, MAIN_FEATURE_SET, model, cfg)
                log(f"  {split:8s} zeros-included {model:12s} MAE {r['MAE']:.3f} RMSE {r['RMSE']:.3f}")
                r.pop("oof")
                results.append({**r, "experiment": "zero_grade_sensitivity", "target_policy": "zeros_included"})
    groups = []
    for split in SPLITS:
        sub = {m: oofs[(split, m)] for m in MODELS if (split, m) in oofs}
        for col in ("student_college", "student_dept", "program", "course_category"):
            g = per_group(frame, sub, col)
            g.insert(0, "split", split)
            groups.append(g)
    return {"results": pd.DataFrame(results), "per_group": pd.concat(groups, ignore_index=True),
            "oof": {f"{s}:{m}": v for (s, m), v in oofs.items()}}
