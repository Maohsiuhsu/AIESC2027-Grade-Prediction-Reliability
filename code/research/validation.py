"""Validation experiments on the single-semester task.

1. Feature ablation (course / student / course+student) on identical grouped folds.
2. Course-grouped evaluation: entire courses held out; every encoder and target statistic is fit
   inside the training fold only (models are rebuilt per fold), so no fold-level leakage.
3. Section-grouped leakage audit: static check that no student identifier or same-term grade of
   another course is a feature; overlap statistics; a leak-injection contrast; and a
   student-disjoint variant of the section split.
Paired MAE differences carry group-level bootstrap confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import (COURSE_CATEGORICAL, COURSE_NUMERIC, EXCLUDED_COLUMNS, STUDENT_CATEGORICAL,
                     TARGET, ResearchConfig)
from .experiments import GROUP_COL, mae_rmse
from .models import make_model

ABLATION_SETS = {
    "course": {"categorical": COURSE_CATEGORICAL, "numeric": COURSE_NUMERIC},
    "student": {"categorical": STUDENT_CATEGORICAL, "numeric": []},
    "course+student": {"categorical": COURSE_CATEGORICAL + STUDENT_CATEGORICAL, "numeric": COURSE_NUMERIC},
}
ABLATION_MODELS = ["mean_global", "mean_course", "ridge", "catboost", "lightgbm"]
SPLITS3 = {"student": "student_id", "section": "section_key", "course": "course_key"}
N_BOOT = 1000
MIN_TRAIN_ROWS = 20

# What a feature may legitimately depend on. Anything about the student beyond registration
# attributes, or any grade of the same term, is forbidden.
REGISTRATION_TIME_FEATURES = set(COURSE_CATEGORICAL + COURSE_NUMERIC + STUDENT_CATEGORICAL)


def folds3(frame: pd.DataFrame, split: str, cfg: ResearchConfig):
    """Same construction as experiments.folds, extended with a course-grouped split."""
    groups = frame[SPLITS3[split]].astype(str)
    rng = np.random.default_rng(cfg.seed)
    uniq = groups.unique()
    perm = dict(zip(uniq, rng.permutation(len(uniq))))
    return list(GroupKFold(n_splits=cfg.n_folds).split(frame, groups=groups.map(perm)))


def oof_predict(frame: pd.DataFrame, fold_list, feature_set: dict, model: str, cfg: ResearchConfig,
                train_filter=None) -> tuple[np.ndarray, dict]:
    """Out-of-fold predictions. The model (with all its encoders / target statistics) is created
    and fit inside each fold on training rows only. `train_filter(train_idx, test_idx)` may shrink
    the training rows (used by the student-disjoint variant)."""
    cols = list(dict.fromkeys(feature_set["categorical"] + feature_set["numeric"] + ["course_key"]))
    X, y = frame[cols], frame[TARGET].to_numpy()
    oof = np.full(len(frame), np.nan)
    train_sizes, skipped = [], 0
    for tr, te in fold_list:
        if train_filter is not None:
            tr = train_filter(tr, te)
        if len(tr) < MIN_TRAIN_ROWS:          # a filtered fold can be (nearly) empty on small data
            skipped += 1
            continue
        train_sizes.append(len(tr))
        m = make_model(model, feature_set["categorical"], feature_set["numeric"], cfg)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = np.clip(m.predict(X.iloc[te]), 0, 100)
    return oof, {"mean_train_rows": int(np.mean(train_sizes)) if train_sizes else 0, "folds_skipped": skipped}


def paired_bootstrap(frame: pd.DataFrame, oof_a: np.ndarray, oof_b: np.ndarray, group_col: str,
                     n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """MAE(a) - MAE(b) with a group-level bootstrap (resample groups with replacement).
    Positive = b is better. Row-level errors are aggregated per group so the paired structure and
    the within-group correlation are both respected."""
    ok = ~(np.isnan(oof_a) | np.isnan(oof_b))
    y = frame[TARGET].to_numpy()[ok]
    ea, eb = np.abs(y - oof_a[ok]), np.abs(y - oof_b[ok])
    g = frame[group_col].astype(str).to_numpy()[ok]
    codes, inv = np.unique(g, return_inverse=True)
    sum_a = np.bincount(inv, weights=ea)
    sum_b = np.bincount(inv, weights=eb)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    n = len(codes)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        diffs[i] = (w @ sum_a - w @ sum_b) / (w @ cnt)
    point = float((sum_a.sum() - sum_b.sum()) / cnt.sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mae_diff": round(point, 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "significant": bool(lo > 0 or hi < 0), "groups": int(n), "n_boot": n_boot}


# ---- 1 & 2: ablation on three grouped splits -------------------------------------------------------
def run_ablation(frame: pd.DataFrame, cfg: ResearchConfig, log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, cis, oofs = [], [], {}
    for split in SPLITS3:
        fl = folds3(frame, split, cfg)
        for fs_name, fs in ABLATION_SETS.items():
            for model in ABLATION_MODELS:
                if model == "mean_course" and fs_name == "student":
                    continue
                oof, info = oof_predict(frame, fl, fs, model, cfg)
                mae, rmse = mae_rmse(frame[TARGET], oof)
                oofs[(split, fs_name, model)] = oof
                rows.append({"split": split, "feature_set": fs_name, "model": model, "n": len(frame),
                             "MAE": round(mae, 4), "RMSE": round(rmse, 4)})
                log(f"  {split:8s} {fs_name:15s} {model:12s} MAE {mae:.3f} RMSE {rmse:.3f}")
        # paired contrasts with group-level bootstrap CIs
        gcol = SPLITS3[split]
        for model in ("ridge", "catboost", "lightgbm"):
            for a, b, label in (("course", "course+student", "value of student features given course"),
                                ("student", "course+student", "value of course features given student"),
                                (None, "course+student", "course+student vs mean_course baseline")):
                oa = oofs[(split, a, model)] if a else oofs[(split, "course", "mean_course")]
                ob = oofs[(split, b, model)]
                r = paired_bootstrap(frame, oa, ob, gcol, seed=cfg.seed)
                cis.append({"split": split, "model": model, "contrast": label, "baseline": a or "mean_course",
                            "candidate": b, **r})
    return pd.DataFrame(rows), pd.DataFrame(cis), oofs


# ---- 3: section-grouped leakage audit ------------------------------------------------------------
def static_feature_audit() -> dict:
    """No identifier, no per-student derived quantity, no grade of the same term is a feature."""
    used = set().union(*[set(fs["categorical"]) | set(fs["numeric"]) for fs in ABLATION_SETS.values()])
    forbidden_used = sorted(used & set(EXCLUDED_COLUMNS))
    non_registration = sorted(used - REGISTRATION_TIME_FEATURES)
    return {"features_used": sorted(used), "forbidden_used": forbidden_used,
            "non_registration_time": non_registration,
            "student_identifier_used": "student_id" in used,
            "same_term_grades_used": bool({"term_grade", "midterm_score", "daily_score", "final_score"} & used),
            "passed": not forbidden_used and not non_registration}


def overlap_statistics(frame: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    rows = []
    for split in SPLITS3:
        for k, (tr, te) in enumerate(folds3(frame, split, cfg)):
            trs, tes = frame.iloc[tr], frame.iloc[te]
            train_students, train_courses = set(trs["student_id"]), set(trs["course_key"])
            rows.append({
                "split": split, "fold": k, "test_rows": len(te),
                "test_rows_with_student_in_train": round(float(tes["student_id"].isin(train_students).mean()), 4),
                "test_rows_with_course_in_train": round(float(tes["course_key"].isin(train_courses).mean()), 4),
                "test_rows_with_section_in_train": round(float(tes["section_key"].isin(set(trs["section_key"])).mean()), 4),
            })
    return pd.DataFrame(rows)


def leak_injection_contrast(frame: pd.DataFrame, cfg: ResearchConfig, log=print) -> dict:
    """Under the section split, deliberately add the forbidden feature 'student's mean grade in
    the training fold' (computed per fold from training rows only). If the base features already
    carried same-student information, this would add little; a large gain shows the base model
    does NOT have it, and that the metric is sensitive to it."""
    fl = folds3(frame, "section", cfg)
    fs = ABLATION_SETS["course+student"]
    base, _ = oof_predict(frame, fl, fs, "catboost", cfg)
    leaky = np.full(len(frame), np.nan)
    f2 = frame.copy()
    for tr, te in fl:
        stu_mean = f2.iloc[tr].groupby("student_id")[TARGET].mean()
        f2["_leak_student_mean"] = f2["student_id"].map(stu_mean)   # NaN for students not in train
        fs_leak = {"categorical": fs["categorical"], "numeric": fs["numeric"] + ["_leak_student_mean"]}
        cols = list(dict.fromkeys(fs_leak["categorical"] + fs_leak["numeric"] + ["course_key"]))
        m = make_model("catboost", fs_leak["categorical"], fs_leak["numeric"], cfg)
        m.fit(f2.iloc[tr][cols], f2[TARGET].to_numpy()[tr])
        leaky[te] = np.clip(m.predict(f2.iloc[te][cols]), 0, 100)
    ci = paired_bootstrap(frame, base, leaky, "section_key", seed=cfg.seed)
    mb, ml = mae_rmse(frame[TARGET], base)[0], mae_rmse(frame[TARGET], leaky)[0]
    log(f"  leak injection (section split, catboost): base MAE {mb:.3f} -> with same-term student mean {ml:.3f}")
    return {"base_MAE": round(mb, 4), "with_leak_MAE": round(ml, 4), **ci,
            "interpretation": ("a large drop confirms that the base features do not carry same-student, same-term "
                               "information; the base result is not inflated by it")}


def student_disjoint_section_variant(frame: pd.DataFrame, cfg: ResearchConfig, log=print) -> pd.DataFrame:
    """Section-grouped folds where every training row of a student who appears in the test
    sections is removed. This removes same-student overlap entirely at the cost of training rows."""
    fl = folds3(frame, "section", cfg)
    students = frame["student_id"].to_numpy()

    def drop_test_students(tr, te):
        test_students = set(students[te])
        keep = np.array([students[i] not in test_students for i in tr])
        return tr[keep]

    rows = []
    fs = ABLATION_SETS["course+student"]
    for model in ("mean_course", "catboost", "lightgbm"):
        plain, ip = oof_predict(frame, fl, fs, model, cfg)
        disj, idj = oof_predict(frame, fl, fs, model, cfg, train_filter=drop_test_students)
        ok = ~np.isnan(disj)                      # compare on the rows the disjoint variant could predict
        ci = paired_bootstrap(frame, plain, disj, "section_key", seed=cfg.seed)
        mp, md = mae_rmse(frame[TARGET][ok], plain[ok])[0], mae_rmse(frame[TARGET][ok], disj[ok])[0]
        log(f"  student-disjoint section split {model:12s}: plain {mp:.3f} (train {ip['mean_train_rows']:,}) -> "
            f"disjoint {md:.3f} (train {idj['mean_train_rows']:,})")
        rows.append({"model": model, "MAE_section_plain": round(mp, 4), "MAE_section_student_disjoint": round(md, 4),
                     "train_rows_plain": ip["mean_train_rows"], "train_rows_disjoint": idj["mean_train_rows"],
                     "rows_evaluated": int(ok.sum()), "folds_skipped": idj["folds_skipped"],
                     "diff_plain_minus_disjoint": ci["mae_diff"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    return pd.DataFrame(rows)


# ---- zero-grade sensitivity: population shift ----------------------------------------------------
def zero_population_shift(full: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows = [{"group_by": "all", "group": "all", "rows_all": len(full), "rows_primary": len(primary),
             "zeros_removed": len(full) - len(primary),
             "zero_share": round(1 - len(primary) / len(full), 4),
             "mean_all": round(float(full[TARGET].mean()), 2), "mean_primary": round(float(primary[TARGET].mean()), 2),
             "std_all": round(float(full[TARGET].std()), 2), "std_primary": round(float(primary[TARGET].std()), 2)}]
    for col in ("student_college", "program", "course_category"):
        for g, a in full.groupby(col):
            p = primary[primary[col] == g]
            rows.append({"group_by": col, "group": g, "rows_all": len(a), "rows_primary": len(p),
                         "zeros_removed": len(a) - len(p), "zero_share": round(1 - len(p) / len(a), 4) if len(a) else 0,
                         "mean_all": round(float(a[TARGET].mean()), 2), "mean_primary": round(float(p[TARGET].mean()), 2) if len(p) else None,
                         "std_all": round(float(a[TARGET].std()), 2), "std_primary": round(float(p[TARGET].std()), 2) if len(p) else None})
    return pd.DataFrame(rows)


def run_zero_sensitivity(full: pd.DataFrame, cfg: ResearchConfig, log=print) -> pd.DataFrame:
    rows = []
    fs = ABLATION_SETS["course+student"]
    for split in SPLITS3:
        fl = folds3(full, split, cfg)
        for model in ("mean_course", "catboost", "lightgbm"):
            oof, _ = oof_predict(full, fl, fs, model, cfg)
            mae, rmse = mae_rmse(full[TARGET], oof)
            zero = (full[TARGET] == 0).to_numpy()
            mae_nz = mae_rmse(full[TARGET][~zero], oof[~zero])[0]
            mae_z = mae_rmse(full[TARGET][zero], oof[zero])[0] if zero.any() else np.nan
            log(f"  zeros-included {split:8s} {model:12s} MAE {mae:.3f} (non-zero rows {mae_nz:.3f}, zero rows {mae_z:.3f})")
            rows.append({"split": split, "model": model, "n": len(full), "MAE_all": round(mae, 4), "RMSE_all": round(rmse, 4),
                         "MAE_nonzero_rows": round(mae_nz, 4), "MAE_zero_rows": round(float(mae_z), 4)})
    return pd.DataFrame(rows)
