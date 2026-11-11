"""Analysis of the calibration-split follow-up (design: calibration_split_spec.json).

Inputs : outputs/aiesc2027/calibration_split/jobs/fold{k}.npz + .json (written by calibration_fit.py) and the
         aligned frame / original OOF predictions of io_data.load_all().
Every interval and recalibration map for test fold k is computed from the calibration rows of training fold k
only (students disjoint from the proper-training rows and from test fold k).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C
from . import metrics as M
from .uncertainty import conformal_quantile, crps_normal, nll_normal

JOBS = C.OUT / "calibration_split" / "jobs"
METHODS = {
    "ngb_raw": "NGBoost Normal, raw",
    "ngb_rescaled": "NGBoost Normal, inner-validation rescaled",
    "split_abs": "Split conformal, absolute residual",
    "split_norm": "Split conformal, sigma-normalised residual",
    "mondrian_category": "Mondrian conformal by course category",
    "mondrian_predicted_range": "Mondrian conformal by predicted grade range",
    "cqr_lightgbm": "CQR, LightGBM quantiles",
    "split_abs_lightgbm": "Split conformal around LightGBM, absolute residual",
}
CONFORMAL = [m for m in METHODS if m not in ("ngb_raw", "ngb_rescaled")]
PROB_VERSIONS = {"original_full_fold": "Original NGBoost (full training fold), raw",
                 "refit_raw": "Refitted NGBoost (80%), raw",
                 "refit_isotonic": "Refitted NGBoost, isotonic recalibration",
                 "refit_platt": "Refitted NGBoost, logistic (Platt) recalibration"}


# ---- loading ------------------------------------------------------------------------------------------
def load_jobs(n_folds: int = C.EXPECTED_N_FOLDS) -> tuple[dict, list[dict]]:
    jobs, infos = {}, []
    for f in range(n_folds):
        p = JOBS / f"fold{f}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run calibration_fit.py --fold {f} first")
        jobs[f] = dict(np.load(p))
        infos.append(json.loads((JOBS / f"fold{f}.json").read_text()))
    return jobs, infos


def check_jobs(jobs: dict, infos: list[dict], fold: np.ndarray, students: np.ndarray) -> pd.DataFrame:
    rows = []

    def add(name, ok, detail=""):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})

    seen = np.zeros(len(fold), int)
    for f, j in jobs.items():
        te, cal, fit = j["test_idx"], j["cal_idx"], j["fit_idx"]
        add(f"fold {f}: test indices equal the saved partition", np.array_equal(np.sort(te), np.flatnonzero(fold == f)))
        add(f"fold {f}: fit + calibration = training fold",
            np.array_equal(np.sort(np.concatenate([fit, cal])), np.flatnonzero(fold != f)), f"{len(fit)} + {len(cal)}")
        sf, sc, st = (set(students[i]) for i in (fit, cal, te))
        add(f"fold {f}: students disjoint across fit / calibration / test", not (sf & sc or sc & st or sf & st))
        add(f"fold {f}: all predictions finite", all(np.isfinite(v).all() for k, v in j.items() if k.endswith(("_cal", "_te"))))
        add(f"fold {f}: NGBoost scale > 0", bool((j["ngb_scale_cal"] > 0).all() and (j["ngb_scale_te"] > 0).all()))
        add(f"fold {f}: full-run fit (not quick)", not infos[f]["quick"])
        add(f"fold {f}: LightGBM reproduction exact (max |diff| < 1e-6)",
            infos[f]["repro_lightgbm"]["max_abs_diff"] < 1e-6, infos[f]["repro_lightgbm"]["max_abs_diff"])
        seen[te] += 1
    add("test folds form a partition of all rows", bool((seen == 1).all()))
    return pd.DataFrame(rows)


def reproduction_table(infos: list[dict]) -> pd.DataFrame:
    rows = []
    for i in infos:
        r = i["repro_ngboost"]
        rows.append({"fold": i["fold"], "recomputed_partition_equals_saved": i["recomputed_test_idx_equals_saved"],
                     "lightgbm_max_abs_diff": i["repro_lightgbm"]["max_abs_diff"],
                     "ngboost_MAE_refit": r["refit"]["MAE"], "ngboost_MAE_saved": r["saved"]["MAE"],
                     "ngboost_NLL_refit": r["refit"]["NLL"], "ngboost_NLL_saved": r["saved"]["NLL"],
                     "ngboost_cov90_refit": r["refit"]["coverage90"], "ngboost_cov90_saved": r["saved"]["coverage90"],
                     "ngboost_mean_abs_diff_loc": r["mean_abs_diff_loc"],
                     "ngboost_scale_multiplier_refit": r["scale_multiplier"],
                     "ngboost_scale_multiplier_saved": r["saved_scale_multiplier"],
                     **{f"partition_{k}": v for k, v in i["partition"].items()},
                     "ngboost_80pct_best_iteration": i["ngboost_fit"]["best_iteration"],
                     "ngboost_80pct_scale_multiplier": i["ngboost_fit"]["scale_multiplier"],
                     "fit_seconds": i["seconds"]})
    return pd.DataFrame(rows)


# ---- intervals ----------------------------------------------------------------------------------------
def predicted_range(mu) -> np.ndarray:
    return M.band_of(np.clip(mu, 0, 100), C.GRADE_BANDS)


def mondrian_q(score_cal, g_cal, g_te, alpha, min_calib) -> np.ndarray:
    """Per-group conformal quantile for each test row; pooled quantile when a group has < min_calib rows."""
    pooled = conformal_quantile(score_cal, alpha)
    q = np.full(len(g_te), pooled)
    for g in np.unique(g_te):
        c = g_cal == g
        if c.sum() >= min_calib:
            q[g_te == g] = conformal_quantile(score_cal[c], alpha)
    return q


def cqr(ql_cal, qh_cal, y_cal, ql_te, qh_te, alpha) -> tuple[np.ndarray, np.ndarray]:
    """Conformalised quantile regression (Romano et al., 2019). Crossed quantiles are re-ordered first."""
    lc, hc = np.minimum(ql_cal, qh_cal), np.maximum(ql_cal, qh_cal)
    lt, ht = np.minimum(ql_te, qh_te), np.maximum(ql_te, qh_te)
    q = conformal_quantile(np.maximum(lc - y_cal, y_cal - hc), alpha)
    return lt - q, ht + q


def fold_intervals(j: dict, y_cal, cat_cal, cat_te, mult: float, level: float, min_calib: int) -> dict:
    a = 1 - level
    z = norm.ppf(0.5 + level / 2)
    mc, sc, mt, s_t = j["ngb_loc_cal"], j["ngb_scale_cal"], j["ngb_loc_te"], j["ngb_scale_te"]
    r_abs, r_norm = np.abs(y_cal - mc), np.abs(y_cal - mc) / sc
    qa, qn = conformal_quantile(r_abs, a), conformal_quantile(r_norm, a)
    q_cat = mondrian_q(r_norm, cat_cal, cat_te, a, min_calib)
    q_rng = mondrian_q(r_norm, predicted_range(mc), predicted_range(mt), a, min_calib)
    lo_q, hi_q = round((1 - level) / 2, 4), round(1 - (1 - level) / 2, 4)
    q_l = conformal_quantile(np.abs(y_cal - j["lgb_cal"]), a)
    out = {"ngb_raw": (mt - z * s_t, mt + z * s_t),
           "ngb_rescaled": (mt - z * s_t * mult, mt + z * s_t * mult),
           "split_abs": (mt - qa, mt + qa),
           "split_norm": (mt - qn * s_t, mt + qn * s_t),
           "mondrian_category": (mt - q_cat * s_t, mt + q_cat * s_t),
           "mondrian_predicted_range": (mt - q_rng * s_t, mt + q_rng * s_t),
           "cqr_lightgbm": cqr(j[f"q{lo_q}_cal"], j[f"q{hi_q}_cal"], y_cal, j[f"q{lo_q}_te"], j[f"q{hi_q}_te"], a),
           "split_abs_lightgbm": (j["lgb_te"] - q_l, j["lgb_te"] + q_l)}
    return {k: (np.clip(lo, 0, 100), np.clip(hi, 0, 100)) for k, (lo, hi) in out.items()}


def assemble(jobs: dict, infos: list[dict], y: np.ndarray, cat: np.ndarray, level: float) -> dict:
    """OOF interval bounds over all rows for one nominal level: method -> (lo, hi)."""
    n = len(y)
    res = {m: (np.full(n, np.nan), np.full(n, np.nan)) for m in METHODS}
    for f, j in jobs.items():
        te, cal = j["test_idx"], j["cal_idx"]
        iv = fold_intervals(j, y[cal], cat[cal], cat[te], infos[f]["ngboost_fit"]["scale_multiplier"], level,
                            C.MONDRIAN_MIN_CALIB)
        for m, (lo, hi) in iv.items():
            res[m][0][te], res[m][1][te] = lo, hi
    return res


def oof_vector(jobs: dict, key: str, n: int) -> np.ndarray:
    v = np.full(n, np.nan)
    for j in jobs.values():
        v[j["test_idx"]] = j[key]
    return v


def interval_table(jobs, infos, y, cat, fold, levels=C.INTERVAL_LEVELS) -> tuple[pd.DataFrame, dict]:
    mu = oof_vector(jobs, "ngb_loc_te", len(y))
    strata = [("all", "all", np.ones(len(y), bool))]
    band, prange = M.band_of(y, C.GRADE_BANDS), predicted_range(mu)
    strata += [("grade_range", b, band == b) for b, _, _ in C.GRADE_BANDS]
    strata += [("predicted_range", b, prange == b) for b, _, _ in C.GRADE_BANDS]
    strata += [("course_category", c, cat == c) for c in sorted(np.unique(cat))]
    strata += [("fold", str(f), fold == f) for f in sorted(np.unique(fold))]
    rows, bounds = [], {}
    for lv in levels:
        iv = assemble(jobs, infos, y, cat, lv)
        bounds[lv] = iv
        for m, (lo, hi) in iv.items():
            cover = (y >= lo) & (y <= hi)
            below = y < lo
            width = np.maximum(hi - lo, 0)
            for by, g, s in strata:
                if not s.any():
                    continue
                cov = float(cover[s].mean())
                rows.append({"method": m, "method_label": METHODS[m], "nominal": lv, "group_by": by, "group": g,
                             "group_en": C.COURSE_CATEGORY_EN.get(g, g), "n": int(s.sum()), "coverage": cov,
                             "coverage_error": cov - lv, "mean_width": float(width[s].mean()),
                             "median_width": float(np.median(width[s])),
                             "share_actual_below_interval": float(below[s].mean()),
                             "share_actual_above_interval": float((~cover & ~below)[s].mean())})
    return pd.DataFrame(rows), bounds


def scoring_table(jobs, infos, y, d_loc, d_scale, d_mult) -> pd.DataFrame:
    n = len(y)
    mu, sd = oof_vector(jobs, "ngb_loc_te", n), oof_vector(jobs, "ngb_scale_te", n)
    mult = np.full(n, np.nan)
    for f, j in jobs.items():
        mult[j["test_idx"]] = infos[f]["ngboost_fit"]["scale_multiplier"]
    rows = [("original_full_fold_raw", d_loc, d_scale), ("original_full_fold_rescaled", d_loc, d_scale * d_mult),
            ("refit_80pct_raw", mu, sd), ("refit_80pct_rescaled", mu, sd * mult)]
    return pd.DataFrame([{"version": v, "NLL": nll_normal(y, a, b), "CRPS": crps_normal(y, a, b)} for v, a, b in rows])


def point_table(jobs, infos, y, preds) -> pd.DataFrame:
    n = len(y)
    rows = {"NGBoost, original (full training fold)": preds["ngboost"],
            "NGBoost, reproduction refit (full training fold)": np.clip(oof_vector(jobs, "repro_ngb_loc", n), 0, 100),
            "NGBoost, proper-training refit (80%)": np.clip(oof_vector(jobs, "ngb_loc_te", n), 0, 100),
            "LightGBM, original (full training fold)": preds["lightgbm"],
            "LightGBM, reproduction refit (full training fold)": oof_vector(jobs, "repro_lgb", n),
            "LightGBM, proper-training refit (80%)": np.clip(oof_vector(jobs, "lgb_te", n), 0, 100)}
    out = []
    for name, p in rows.items():
        r = M.regression(y, p)
        flag = p < C.RISK_THRESHOLD
        out.append({"model": name, "MAE": r["MAE"], "RMSE": r["RMSE"],
                    **{k: v for k, v in M.classification(y < C.RISK_THRESHOLD, flag).items()
                       if k in ("precision", "recall", "f1")}})
    return pd.DataFrame(out)


# ---- probability recalibration ------------------------------------------------------------------------
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def recalibrated_probabilities(jobs, y, d_loc, d_scale, threshold=C.RISK_THRESHOLD) -> dict[str, np.ndarray]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    n = len(y)
    out = {"original_full_fold": norm.cdf((threshold - d_loc) / d_scale)}
    for k in ("refit_raw", "refit_isotonic", "refit_platt"):
        out[k] = np.full(n, np.nan)
    for j in jobs.values():
        te, cal = j["test_idx"], j["cal_idx"]
        pc = norm.cdf((threshold - j["ngb_loc_cal"]) / j["ngb_scale_cal"])
        pt = norm.cdf((threshold - j["ngb_loc_te"]) / j["ngb_scale_te"])
        ac = (y[cal] < threshold).astype(int)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(pc, ac)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(pc)[:, None], ac)
        out["refit_raw"][te] = pt
        out["refit_isotonic"][te] = iso.predict(pt)
        out["refit_platt"][te] = lr.predict_proba(_logit(pt)[:, None])[:, 1]
    return out


def probability_table(probs: dict, y, fold, threshold=C.RISK_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = y < threshold
    base = np.empty(len(y))
    for f in np.unique(fold):
        base[fold == f] = actual[fold != f].mean()
    rows, rel_all = [], []
    for k, p in probs.items():
        rel, ece = M.reliability(actual, p)
        rel_all.append(rel.assign(version=k))
        bs, bs0 = M.brier(actual, p), M.brier(actual, base)
        rows.append({"version": k, "label": PROB_VERSIONS[k], "n": len(y), "prevalence": float(actual.mean()),
                     "mean_predicted_probability": float(p.mean()), **M.ranking(actual, p), "brier": bs,
                     "brier_reference_train_prevalence": bs0, "brier_skill_score": 1 - bs / bs0,
                     "log_loss": M.log_loss(actual, p), "ece_10_bins": ece,
                     "share_p_ge_0.5": float((p >= 0.5).mean())})
    return pd.DataFrame(rows), pd.concat(rel_all, ignore_index=True)


# ---- student-cluster bootstrap ------------------------------------------------------------------------
def bootstrap(bounds90: dict, probs: dict, y, students, n_boot=C.N_BOOT, seed=C.SEED) -> pd.DataFrame:
    """95% percentile CIs: 90% coverage (all rows, actual 0-59), mean width, Brier score; paired replicates."""
    _, inv = np.unique(np.asarray(students).astype(str), return_inverse=True)
    G = int(inv.max()) + 1
    low = y < C.RISK_THRESHOLD
    agg = lambda v: np.bincount(inv, weights=np.asarray(v, float), minlength=G)  # noqa: E731
    n_all, n_low = agg(np.ones(len(y))), agg(low)
    stats = {}
    for m, (lo, hi) in bounds90.items():
        cover = (y >= lo) & (y <= hi)
        stats[(m, "coverage90_all")] = (agg(cover), n_all)
        stats[(m, "coverage90_actual_0_59")] = (agg(cover & low), n_low)
        stats[(m, "mean_width90_all")] = (agg(np.maximum(hi - lo, 0)), n_all)
    for k, p in probs.items():
        stats[(k, "brier")] = (agg((p - low) ** 2), n_all)
    num = np.column_stack([v[0] for v in stats.values()])
    den = np.column_stack([v[1] for v in stats.values()])
    point = num.sum(0) / den.sum(0)
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, num.shape[1]))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        reps[b] = (w @ num) / (w @ den)
    out = []
    keys = list(stats)
    for i, (m, s) in enumerate(keys):
        lo, hi = np.percentile(reps[:, i], [2.5, 97.5])
        out.append({"method": m, "statistic": s, "estimate": point[i], "ci_low": lo, "ci_high": hi})
    # paired differences of Brier scores against the raw refitted probabilities
    ref = keys.index(("refit_raw", "brier"))
    for k in ("refit_isotonic", "refit_platt"):
        i = keys.index((k, "brier"))
        d = reps[:, i] - reps[:, ref]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out.append({"method": f"{k} - refit_raw", "statistic": "brier_difference",
                    "estimate": point[i] - point[ref], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)
