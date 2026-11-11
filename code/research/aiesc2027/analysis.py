"""All result tables of the AIESC 2027 analysis, computed from an `AlignedData` object (no training)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from . import uncertainty as U
from .io_data import AlignedData

NOT_AVAILABLE = "Not Available"


def label(k: str) -> str:
    return C.MODELS[k]["label"] if k in C.MODELS else k


# ---- Table 1 --------------------------------------------------------------------------------------------
def dataset_table(d: AlignedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    c, y, fr = d.counts, d.y, d.frame
    risk = y < C.RISK_THRESHOLD
    off = y < fr[C.OFFICIAL_THRESHOLD_COL].to_numpy(float)
    rows = [
        ("Records in the source export (IR.csv)", c["raw_rows"], ""),
        ("Records of the study term (113-2)", c["term_rows"], ""),
        ("Excluded: term grade = 0 with no status code (unexplained)", c["zero_rows"], ""),
        ("Analysis records", c["analysis_rows"], "primary population"),
        ("Students", c["students"], "grouping unit of the 5-fold CV"),
        ("Course sections", c["sections"], ""),
        ("Courses", c["courses"], ""),
        ("Actual grade < 60 (fixed research threshold)", int(risk.sum()), f"{risk.mean():.3%}"),
        ("Actual grade >= 60", int((~risk).sum()), f"{(~risk).mean():.3%}"),
        ("Actual grade < official programme pass threshold (60 or 70)", int(off.sum()), f"{off.mean():.3%}"),
        ("Records with official threshold 70 (master's / doctoral)",
         int((fr[C.OFFICIAL_THRESHOLD_COL] == 70).sum()), ""),
    ]
    for name, n in zip(*np.unique(M.band_of(y, C.GRADE_BANDS), return_counts=True)):
        rows.append((f"Actual grade range {name}", int(n), f"{n / len(y):.3%}"))
    t1 = pd.DataFrame(rows, columns=["item", "n", "share_or_note"])
    folds = []
    for k in range(C.EXPECTED_N_FOLDS):
        m = d.fold == k
        folds.append({"fold": k, "records": int(m.sum()), "students": int(fr.loc[m, "student_id"].nunique()),
                      "grade_below_60": int(risk[m].sum()), "prevalence_below_60": float(risk[m].mean()),
                      "mean_grade": float(y[m].mean())})
    return t1, pd.DataFrame(folds)


# ---- Table 2 --------------------------------------------------------------------------------------------
def overall_table(d: AlignedData) -> pd.DataFrame:
    y = d.y
    rows = []
    for k, p in d.preds.items():
        r = M.regression(y, p)
        rows.append({"model": label(k), "key": k, "loss": C.MODELS[k]["loss"], "n": len(y), **r,
                     "median_ae": float(np.median(np.abs(p - y))), "ae_p90": M.ae_quantile(y, p, 0.9),
                     **M.compression(y, p)})
    return pd.DataFrame(rows)


def per_fold_table(d: AlignedData) -> pd.DataFrame:
    y, rows = d.y, []
    for k, p in d.preds.items():
        for f in range(C.EXPECTED_N_FOLDS):
            m = d.fold == f
            rows.append({"model": label(k), "fold": f, **M.regression(y[m], p[m]),
                         **{x: v for x, v in M.classification(y[m] < C.RISK_THRESHOLD, p[m] < C.RISK_THRESHOLD).items()
                            if x in ("precision", "recall", "f1")}})
    return pd.DataFrame(rows)


# ---- Table 3 --------------------------------------------------------------------------------------------
def risk_table(d: AlignedData, threshold=None) -> pd.DataFrame:
    """Fixed-threshold rule (predicted grade < threshold) and ranking metrics (score = threshold - pred).
    `threshold` None -> fixed 60; "official" -> programme-specific pass threshold per row."""
    y = d.y
    thr = (d.frame[C.OFFICIAL_THRESHOLD_COL].to_numpy(float) if threshold == "official"
           else np.full(len(y), C.RISK_THRESHOLD if threshold is None else float(threshold)))
    actual = y < thr
    rows = []
    for k, p in d.preds.items():
        cls = M.classification(actual, p < thr)
        rk = M.ranking(actual, thr - p)
        rows.append({"model": label(k), "key": k, "threshold": "60" if threshold is None else str(threshold),
                     "risk_score": "threshold - predicted grade (ranking, not a probability)", **cls, **rk,
                     "recall_at_budget": M.recall_at_budget(actual, thr - p, int(actual.sum())),
                     "probability_output": NOT_AVAILABLE if k != "ngboost" else "see NGBoost P(Y<threshold) table"})
    return pd.DataFrame(rows)


def ngboost_probability_table(d: AlignedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probabilistic risk from the NGBoost Normal distribution: P(Y < threshold)."""
    y, rows, rel_all = d.y, [], []
    for name, thr in (("60", np.full(len(y), C.RISK_THRESHOLD)),
                      ("official", d.frame[C.OFFICIAL_THRESHOLD_COL].to_numpy(float))):
        actual = y < thr
        for version, scale in (("normal_raw", d.ngb_scale), ("normal_rescaled", d.ngb_scale * d.ngb_scale_mult)):
            prob = U.prob_below(thr, d.ngb_loc, scale)
            # reference forecast: prevalence of the TRAINING folds (no test label used)
            base = np.empty(len(y))
            for f in range(C.EXPECTED_N_FOLDS):
                base[d.fold == f] = actual[d.fold != f].mean()
            rel, ece = M.reliability(actual, prob)
            rel_all.append(rel.assign(threshold=name, version=version))
            bs, bs0 = M.brier(actual, prob), M.brier(actual, base)
            cls = M.classification(actual, prob >= 0.5)
            rows.append({"threshold": name, "version": version, "n": len(y), "prevalence": float(actual.mean()),
                         "mean_predicted_probability": float(prob.mean()), **M.ranking(actual, prob),
                         "brier": bs, "brier_reference_train_prevalence": bs0, "brier_skill_score": 1 - bs / bs0,
                         "log_loss": M.log_loss(actual, prob), "ece_10_bins": ece,
                         "precision_at_p0.5": cls["precision"], "recall_at_p0.5": cls["recall"],
                         "f1_at_p0.5": cls["f1"],
                         "p0.5_rule_equals_point_rule": bool(np.array_equal(prob >= 0.5, d.preds["ngboost"] < thr))})
    return pd.DataFrame(rows), pd.concat(rel_all, ignore_index=True)


def pr_curves(d: AlignedData, max_points=400) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_curve
    actual = d.y < C.RISK_THRESHOLD
    rows = []
    scores = {label(k): -p for k, p in d.preds.items()}
    scores["NGBoost P(Y<60)"] = U.prob_below(C.RISK_THRESHOLD, d.ngb_loc, d.ngb_scale)
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(actual, s)
        idx = np.unique(np.linspace(0, len(prec) - 1, min(max_points, len(prec))).astype(int))
        rows.append(pd.DataFrame({"model": name, "recall": rec[idx], "precision": prec[idx]}))
    return pd.concat(rows, ignore_index=True)


# ---- Table 4 --------------------------------------------------------------------------------------------
def bias_table(d: AlignedData) -> pd.DataFrame:
    out = []
    for k, p in d.preds.items():
        out.append(M.band_bias(d.y, p, C.GRADE_BANDS).assign(model=label(k), key=k))
    t = pd.concat(out, ignore_index=True)
    return t[["model", "key"] + [c for c in t.columns if c not in ("model", "key")]]


def low_grade_detail(d: AlignedData) -> pd.DataFrame:
    """Within actual < 60: share predicted >= 60 (missed) and share predicted >= 70 (strong overestimation);
    within actual >= 90: share predicted < 80 (strong underestimation)."""
    y, rows = d.y, []
    lo, hi = y < 60, y >= 90
    for k, p in d.preds.items():
        rows.append({"model": label(k), "n_actual_below_60": int(lo.sum()),
                     "share_low_predicted_ge_60": float((p[lo] >= 60).mean()),
                     "share_low_predicted_ge_70": float((p[lo] >= 70).mean()),
                     "mean_overestimate_low": float((p[lo] - y[lo]).mean()),
                     "n_actual_ge_90": int(hi.sum()), "share_high_predicted_lt_80": float((p[hi] < 80).mean()),
                     "mean_underestimate_high": float((p[hi] - y[hi]).mean()),
                     "min_prediction": float(p.min()), "share_predictions_below_60": float((p < 60).mean())})
    return pd.DataFrame(rows)


# ---- Table 5 --------------------------------------------------------------------------------------------
def uncertainty_tables(d: AlignedData) -> dict[str, pd.DataFrame]:
    cat = d.frame["course_category"].astype(str).to_numpy()
    iv = U.interval_table(d.y, d.ngb_loc, d.ngb_scale, d.ngb_scale_mult, d.fold, cat, C.GRADE_BANDS,
                          C.INTERVAL_LEVELS, C.MONDRIAN_MIN_CALIB)
    iv["group_en"] = iv["group"].map(lambda g: C.COURSE_CATEGORY_EN.get(g, g))
    sc = U.scoring_table(d.y, d.ngb_loc, d.ngb_scale, d.ngb_scale_mult)
    pit = U.pit(d.y, d.ngb_loc, d.ngb_scale)
    hist = np.histogram(pit, bins=20, range=(0, 1))[0]
    pit_t = pd.DataFrame({"bin_lower": np.linspace(0, 0.95, 20), "count": hist, "expected_uniform": len(pit) / 20})
    return {"intervals": iv, "scores": sc, "pit": pit_t}


def check_against_ir(unc: dict) -> pd.DataFrame:
    """Compare the recomputed raw Normal intervals with the IR study's saved T3 (student) row and record
    the source of the family-split figures (91.7% / 46.662) quoted in the IR reports."""
    ir = pd.read_csv(C.FINAL_NGBOOST_UNCERTAINTY)
    iv, sc = unc["intervals"], unc["scores"]
    rows = []
    for unit, version, mine in (("T3", "raw", "normal_raw"), ("T3", "rescaled_in_fold", "normal_rescaled")):
        r = ir[(ir.unit == unit) & (ir.version == version) & (ir.population == "all")].iloc[0]
        rows.append({"source": f"final/ngboost_uncertainty.csv {unit} {version}", "quantity": "NLL",
                     "ir_value": r["NLL"], "recomputed": float(sc.loc[sc.version == mine, "NLL"].iat[0])})
        for lv in C.INTERVAL_LEVELS:
            q = iv[(iv.method == mine) & (iv.nominal == lv) & (iv.group_by == "all")].iloc[0]
            rows.append({"source": f"final/ngboost_uncertainty.csv {unit} {version}", "quantity": f"coverage_{int(lv * 100)}",
                         "ir_value": r[f"coverage_{int(lv * 100)}"], "recomputed": q["coverage"]})
            rows.append({"source": f"final/ngboost_uncertainty.csv {unit} {version}", "quantity": f"width_{int(lv * 100)}",
                         "ir_value": r[f"width_{int(lv * 100)}"], "recomputed": q["mean_width"]})
    t = pd.DataFrame(rows)
    t["abs_diff"] = (t["ir_value"] - t["recomputed"]).abs()
    return t


def verify_family_t1(d: AlignedData) -> pd.DataFrame:
    """Source check of the course-FAMILY split figures quoted in the IR reports (90% coverage ~91.7%,
    mean width 46.662): recomputed from the saved T1 per-fold NGBoost files (same primary frame, a
    different fold partition). Context only; never pooled with the student-split results."""
    n = len(d.y)
    loc, scale = np.full(n, np.nan), np.full(n, np.nan)
    for f in range(C.EXPECTED_N_FOLDS):
        z = np.load(C.AMENDMENT_JOBS / f"M3_ngboost__T1__f{f}.npz")
        loc[z["test_idx"]], scale[z["test_idx"]] = z["loc"], z["scale"]
    if not (np.isfinite(loc).all() and np.isfinite(scale).all()):
        raise ValueError("T1 files do not cover every row")
    ir = pd.read_csv(C.FINAL_NGBOOST_UNCERTAINTY)
    r = ir[(ir.unit == "T1") & (ir.version == "raw") & (ir.population == "all")].iloc[0]
    rows = [{"source": "jobs/M3_ngboost__T1__f*.npz vs final/ngboost_uncertainty.csv (T1, raw, all)",
             "quantity": "NLL", "ir_value": r["NLL"], "recomputed": U.nll_normal(d.y, loc, scale)}]
    from scipy.stats import norm
    for lv in C.INTERVAL_LEVELS:
        z = norm.ppf(0.5 + lv / 2)
        lo, hi = np.clip(loc - z * scale, 0, 100), np.clip(loc + z * scale, 0, 100)
        rows.append({"source": rows[0]["source"], "quantity": f"coverage_{int(lv * 100)}",
                     "ir_value": r[f"coverage_{int(lv * 100)}"], "recomputed": float(np.mean((d.y >= lo) & (d.y <= hi)))})
        rows.append({"source": rows[0]["source"], "quantity": f"width_{int(lv * 100)}",
                     "ir_value": r[f"width_{int(lv * 100)}"], "recomputed": float(np.mean(hi - lo))})
    t = pd.DataFrame(rows)
    t["abs_diff"] = (t["ir_value"] - t["recomputed"]).abs()
    return t
