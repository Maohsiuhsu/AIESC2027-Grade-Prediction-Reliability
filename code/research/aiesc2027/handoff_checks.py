"""Read-only evidence checks for the ChatGPT hand-off: independent recomputation and cross-version consistency.

The recomputation reloads the analysis frame and the saved predictions in memory and recomputes headline numbers
with scikit-learn / numpy instead of the AIESC metric code. Nothing is written by this module.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from research.aiesc2027 import config as C
from research.aiesc2027 import io_data
from research.aiesc2027.handoff_values import MODEL_KEYS, MODEL_LABEL, OUT, Registry, table

CAL_JOBS = OUT / "calibration_split" / "jobs"
RISK_JOBS = OUT / "risk_methods" / "jobs"


def _row(quantity, recorded_key, reg: Registry, recomputed, tol, method):
    rec = reg[recorded_key]
    diff = abs(float(recomputed) - rec.value)
    return {"quantity": quantity, "recorded_value": rec.value, "recorded_source": rec.source,
            "recorded_selector": f"{rec.selector}; column {rec.column}", "recomputed_value": float(recomputed),
            "abs_difference": diff, "tolerance": tol, "status": "MATCH" if diff <= tol else "MISMATCH",
            "recomputation_method": method}


def recompute(reg: Registry) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (recomputation table, leakage / partition checks)."""
    d = io_data.load_all()
    y, fold = d.y, d.fold
    students = d.frame["student_id"].to_numpy()
    pos = y < C.RISK_THRESHOLD
    rows = []
    for m in MODEL_KEYS:
        p = d.preds[m]
        lab = MODEL_LABEL[m]
        rows.append(_row(f"{lab} MAE (original OOF)", f"{m}_MAE", reg, np.mean(np.abs(y - p)), 5e-4, "numpy mean |y - p|"))
        rows.append(_row(f"{lab} RMSE (original OOF)", f"{m}_RMSE", reg, np.sqrt(np.mean((y - p) ** 2)), 5e-4, "numpy"))
        flag = p < C.RISK_THRESHOLD
        tp = int((flag & pos).sum())
        rows.append(_row(f"{lab} recall, predicted < 60", f"{m}_recall", reg, tp / pos.sum(), 5e-4, "count TP / positives"))
        rows.append(_row(f"{lab} precision, predicted < 60", f"{m}_precision", reg, tp / flag.sum(), 5e-4, "count TP / flagged"))
        rows.append(_row(f"{lab} ROC-AUC (score = -prediction)", f"{m}_roc_auc", reg, roc_auc_score(pos, -p), 5e-4,
                         "sklearn roc_auc_score"))
        rows.append(_row(f"{lab} PR-AUC (score = -prediction)", f"{m}_pr_auc", reg, average_precision_score(pos, -p), 5e-3,
                         "sklearn average_precision_score (step interpolation; AIESC uses a tie-aware PR-AUC, "
                         "so a small difference is expected where predictions tie)"))
    d_mae = np.mean(np.abs(y - d.preds["lightgbm_l1"])) - np.mean(np.abs(y - d.preds["lightgbm"]))
    rows.append(_row("LightGBM-L1 minus LightGBM, MAE", "d_lightgbm_l1_MAE_difference", reg, d_mae, 5e-4, "numpy"))
    d_rec = ((d.preds["lightgbm_l1"] < 60) & pos).sum() / pos.sum() - ((d.preds["lightgbm"] < 60) & pos).sum() / pos.sum()
    rows.append(_row("LightGBM-L1 minus LightGBM, recall", "d_lightgbm_l1_recall_difference", reg, d_rec, 5e-4, "numpy"))
    z90 = norm.ppf(0.95)
    cov = np.mean(np.abs(y - d.ngb_loc) <= z90 * d.ngb_scale)
    rows.append(_row("NGBoost raw 90% coverage (original OOF)", "ni_normal_raw_cov90", reg, cov, 5e-4, "|y - loc| <= z * scale"))
    pr = norm.cdf((C.RISK_THRESHOLD - d.ngb_loc) / d.ngb_scale)
    rows.append(_row("NGBoost mean P(Y < 60) (original OOF)", "ngbprob_mean_predicted_probability", reg, pr.mean(), 5e-4,
                     "Phi((60 - loc) / scale)"))
    rows.append(_row("NGBoost Brier score (original OOF)", "ngbprob_brier", reg, np.mean((pr - pos) ** 2), 5e-4, "numpy"))

    # ---- follow-up 1: calibration split ---------------------------------------------------------------
    leak = []
    cov_abs, brier_iso, cov_low = [], [], []
    for f in range(C.EXPECTED_N_FOLDS):
        j = dict(np.load(CAL_JOBS / f"fold{f}.npz"))
        te, cal, fit = j["test_idx"], j["cal_idx"], j["fit_idx"]
        sf, sc, st = set(students[fit]), set(students[cal]), set(students[te])
        leak += [
            {"check": f"fold {f}: calibration-split test rows = saved student-split test fold",
             "status": np.array_equal(np.sort(te), np.flatnonzero(fold == f)), "detail": f"{len(te)} rows"},
            {"check": f"fold {f}: no row in both fit and calibration parts",
             "status": len(np.intersect1d(fit, cal)) == 0, "detail": f"fit {len(fit)}, calibration {len(cal)}"},
            {"check": f"fold {f}: fit + calibration = complement of the test fold",
             "status": np.array_equal(np.sort(np.concatenate([fit, cal])), np.flatnonzero(fold != f)), "detail": ""},
            {"check": f"fold {f}: no student shared by fit / calibration / test",
             "status": not (sf & sc or sc & st or sf & st), "detail": f"{len(sc)} calibration students"},
        ]
        q = np.sort(np.abs(y[cal] - j["ngb_loc_cal"]))[int(np.ceil((len(cal) + 1) * 0.9)) - 1]
        hit = np.abs(y[te] - j["ngb_loc_te"]) <= q
        cov_abs.append(hit)
        cov_low.append(hit[y[te] < 60])
        p_cal = norm.cdf((60 - j["ngb_loc_cal"]) / j["ngb_scale_cal"])
        p_te = norm.cdf((60 - j["ngb_loc_te"]) / j["ngb_scale_te"])
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_cal, (y[cal] < 60).astype(float))
        brier_iso.append((iso.predict(p_te) - (y[te] < 60)) ** 2)
    rows.append(_row("Split conformal (absolute) 90% coverage, refit", "cs_split_abs_cov90", reg,
                     np.concatenate(cov_abs).mean(), 5e-4, "quantile ceil((n+1)0.9) of calibration |y - loc|"))
    rows.append(_row("Split conformal (absolute) 90% coverage, actual 0-59", "cs_split_abs_cov90_low", reg,
                     np.concatenate(cov_low).mean(), 5e-4, "same, restricted to actual < 60"))
    rows.append(_row("Isotonic-recalibrated NGBoost Brier score, refit", "pr_refit_isotonic_brier", reg,
                     np.concatenate(brier_iso).mean(), 5e-4, "sklearn IsotonicRegression fitted per fold on calibration rows"))

    # ---- follow-up 2: risk methods, population A -------------------------------------------------------
    sc_clf, lab_clf, rec_op1, rec_l1 = [], [], [], []
    for f in range(C.EXPECTED_N_FOLDS):
        j = dict(np.load(RISK_JOBS / f"A_fold{f}.npz"))
        te, cal = j["test_idx"], j["cal_idx"]
        pos_cal = y[cal] < 60
        k = max(1, int(round(pos_cal.mean() * len(cal))))
        cut = np.sort(-j["lgb_cal"])[::-1][k - 1]
        rec_op1.append(((-j["lgb_te"] >= cut), y[te] < 60))
        rec_l1.append(((j["l1_te"] < 60), y[te] < 60))
        sc_clf.append(j["clf_te"])
        lab_clf.append(y[te] < 60)
        leak.append({"check": f"fold {f}: risk-method partition identical to the calibration-split partition",
                     "status": np.array_equal(np.sort(cal), np.sort(np.load(CAL_JOBS / f"fold{f}.npz")["cal_idx"])),
                     "detail": f"{len(cal)} calibration rows"})
    s, l = np.concatenate(sc_clf), np.concatenate(lab_clf)
    rows.append(_row("LightGBM classifier PR-AUC, refit", "ra_classifier_pr_auc", reg, average_precision_score(l, s), 5e-3,
                     "sklearn average_precision_score"))
    rows.append(_row("LightGBM classifier ROC-AUC, refit", "ra_classifier_roc_auc", reg, roc_auc_score(l, s), 5e-4,
                     "sklearn roc_auc_score"))
    fl = np.concatenate([a for a, _ in rec_op1]); po = np.concatenate([b for _, b in rec_op1])
    rows.append(_row("LightGBM-L2 recall at OP1 (threshold from calibration prevalence)", "ra_lgb_OP1_recall", reg,
                     (fl & po).sum() / po.sum(), 5e-4, "re-derived threshold from calibration rows"))
    fl = np.concatenate([a for a, _ in rec_l1]); po = np.concatenate([b for _, b in rec_l1])
    rows.append(_row("LightGBM-L1 recall, native rule (refit)", "ra_l1_native_recall", reg, (fl & po).sum() / po.sum(),
                     5e-4, "count"))
    n_b = sum(len(np.load(RISK_JOBS / f"B_fold{f}.npz")["test_idx"]) for f in range(C.EXPECTED_N_FOLDS))
    leak.append({"check": "population B test folds cover 85,567 rows", "status": n_b == C.EXPECTED_TERM_ROWS, "detail": n_b})
    leak.append({"check": "risk label = actual grade < 60 (strict); zero grades excluded from population A",
                 "status": bool((y > 0).all() and pos.sum() == int(reg["n_low"].value)), "detail": int(pos.sum())})
    lk = pd.DataFrame(leak)
    lk["status"] = lk["status"].map({True: "PASS", False: "FAIL"})
    return pd.DataFrame(rows), lk


def text_checks(results_text: str, reg: Registry) -> pd.DataFrame:
    """Each phrase must occur in the Results reference text after the {{key}} values are filled in."""
    t = reg.text()
    phrases = [
        ("4.1", "lowest MAE ({{lightgbm_l1_MAE}}; 95% CI [{{lightgbm_l1_MAE_ci_low}}, {{lightgbm_l1_MAE_ci_high}}])"),
        ("4.1", "lowest RMSE ({{lightgbm_RMSE}}; 95% CI [{{lightgbm_RMSE_ci_low}}, {{lightgbm_RMSE_ci_high}}])"),
        ("4.1", "(difference {{d_lightgbm_l1_MAE_difference}}, 95% CI [{{d_lightgbm_l1_MAE_ci_low_95}}, {{d_lightgbm_l1_MAE_ci_high_95}}])"),
        ("4.1", "CatBoost ({{d_catboost_MAE_difference}}, 95% CI [{{d_catboost_MAE_ci_low_95}}, {{d_catboost_MAE_ci_high_95}}])"),
        ("4.1", "NGBoost ({{d_ngboost_MAE_difference}}, 95% CI [{{d_ngboost_MAE_ci_low_95}}, {{d_ngboost_MAE_ci_high_95}}])"),
        ("4.1", "{{sd_ratio_min}} to {{sd_ratio_max}} times"),
        ("4.1", "Holm-adjusted p-values cannot fall below about {{holm_floor}}"),
        ("4.2", "recall ({{lightgbm_recall}}; 95% CI [{{lightgbm_recall_ci_low}}, {{lightgbm_recall_ci_high}}])"),
        ("4.2", "lowest recall ({{lightgbm_l1_recall}}; 95% CI [{{lightgbm_l1_recall_ci_low}}, {{lightgbm_l1_recall_ci_high}}])"),
        ("4.2", "it did not flag {{lightgbm_l1_FN}} of the {{n_low}} low-grade enrolments"),
        ("4.2", "({{flag_rate_min}} to {{flag_rate_max}})"),
        ("4.2", "PR-AUC ({{lightgbm_pr_auc}}; 95% CI [{{lightgbm_pr_auc_ci_low}}, {{lightgbm_pr_auc_ci_high}}]) and ROC-AUC ({{lightgbm_roc_auc}})"),
        ("4.2", "recall was {{lightgbm_recall_at_budget}} for LightGBM and {{lightgbm_l1_recall_at_budget}} for LightGBM-L1"),
        ("4.2", "(adjusted p = {{d_ft_transformer_precision_p_holm}})"),
        ("4.2", "The mean predicted probability was {{ngbprob_mean_predicted_probability}}"),
        ("4.2", "Brier skill score {{ngbprob_brier_skill_score}}"),
        ("4.2", "(recall {{official_recall_min}} to {{official_recall_max}}; PR-AUC {{official_pr_auc_min}} to {{official_pr_auc_max}})"),
        ("4.2.1", "flagged very different shares of enrolments ({{ra_native_flag_min}} to {{ra_native_flag_max}})"),
        ("4.2.1", "the OP1 cutoff was a predicted grade of {{rt_lgb_grade_cutoff_OP1_min}} to {{rt_lgb_grade_cutoff_OP1_max}}"),
        ("4.2.1", "recall rose from {{ra_l1_native_recall}} to {{ra_l1_OP1_recall}}"),
        ("4.2.1", "recall ranged from {{ra_op1_recall_min}} to {{ra_op1_recall_max}} across methods"),
        ("4.2.1", "(PR-AUC {{ra_classifier_pr_auc}} vs {{ra_lgb_pr_auc}} for L2 regression, {{c_classifier_vs_lgb_pr_auc_difference}}"),
        ("4.2.1", "Class weighting harmed both ranking (PR-AUC {{c_weighted_vs_lgb_pr_auc_difference}}"),
        ("4.2.1", "required flagging {{ra_op2_flag_min}} to {{ra_op2_flag_max}} of all enrolments"),
        ("4.2.1", "Brier skill score {{rp_classifier_brier_skill_score}}"),
        ("4.2.2", "(PR-AUC {{rb_classifier_pr_auc}})"),
        ("4.2.2", "At OP1, {{rb_zero_flag_op1_min}} to {{rb_zero_flag_op1_max}} of the zero-grade enrolments were flagged"),
        ("4.2.2", "PR-AUC changed by {{rb_nonzero_prdiff_min}} to {{rb_nonzero_prdiff_max}}"),
        ("4.3", "mean signed error was {{mse_low_min}} to {{mse_low_max}} points"),
        ("4.3", "{{share_low_pred_ge60_min}} to {{share_low_pred_ge60_max}}"),
        ("4.4", "(empirical coverage {{ni_normal_raw_cov50}}, {{ni_normal_raw_cov80}} and {{ni_normal_raw_cov90}})"),
        ("4.4", "mean width of {{ni_normal_raw_width90}} grade points"),
        ("4.4", "covered only {{ni_normal_raw_cov90_low}} (raw Normal)"),
        ("4.4", "ranged from {{ni_raw_cat_cov90_min}} to {{ni_raw_cat_cov90_max}}"),
        ("4.4.1", "over-covered ({{cs_ngb_raw_cov90}} [95% CI {{cs_ngb_raw_cov90_lo}}, {{cs_ngb_raw_cov90_hi}}]; mean width {{cs_ngb_raw_width90}} points)"),
        ("4.4.1", "nominal marginal coverage ({{cs_conf_cov90_min}} to {{cs_conf_cov90_max}})"),
        ("4.4.1", "coverage was {{cs_conf_cov90_low_min}} to {{cs_conf_cov90_low_max}} across the conformal methods"),
        ("4.4.1", "CQR gave the highest low-grade coverage ({{cs_cqr_lightgbm_cov90_low}} [95% CI {{cs_cqr_lightgbm_cov90_low_lo}}, {{cs_cqr_lightgbm_cov90_low_hi}}])"),
        ("4.4.1", "from {{cs_split_norm_cat_cov90_min}} (pooled normalised conformal) to {{cs_mondrian_category_cat_cov90_min}}"),
        ("4.4.1", "reduced ECE to {{pr_refit_isotonic_ece_10_bins}}"),
    ]
    norm_text = re.sub(r"\s+", " ", results_text)
    out = []
    for sec, ph in phrases:
        keys = re.findall(r"\{\{(\w+)\}\}", ph)
        filled = re.sub(r"\{\{(\w+)\}\}", lambda mt: t[mt.group(1)], ph)
        out.append({"section": sec, "expected_phrase": filled, "value_keys": " ".join(keys),
                    "found_in_results_text": filled in norm_text})
    return pd.DataFrame(out)


def per_fold_claims() -> pd.DataFrame:
    """Claims of the Results text that rest on per-fold or per-bin tables rather than single values."""
    pf = table("metrics/per_fold.csv")
    l1 = pf[pf["model"] == "LightGBM-L1"].set_index("fold")
    lg = pf[pf["model"] == "LightGBM"].set_index("fold")
    rel = table("metrics/ngboost_reliability.csv")
    rel = rel[(rel["version"] == "normal_raw") & (rel["threshold"].astype(str) == "60") & (rel["n"] >= 100)]
    rows = [
        {"claim": "LightGBM-L1 had a lower MAE than LightGBM in every fold (4.1)",
         "evidence": f"per-fold MAE difference {', '.join(f'{v:+.3f}' for v in (l1['MAE'] - lg['MAE']))}",
         "holds": bool((l1["MAE"] < lg["MAE"]).all()), "source": "metrics/per_fold.csv"},
        {"claim": "LightGBM-L1 recall lower than LightGBM in every fold (4.2)",
         "evidence": f"L1 {l1['recall'].min():.3f}-{l1['recall'].max():.3f}; LightGBM {lg['recall'].min():.3f}-{lg['recall'].max():.3f}",
         "holds": bool((l1["recall"] < lg["recall"]).all()), "source": "metrics/per_fold.csv"},
        {"claim": "In every reliability bin with >= 100 records the observed share was below the NGBoost probability (4.2)",
         "evidence": f"{len(rel)} bins; max(observed - predicted) = {(rel['observed_rate'] - rel['mean_predicted']).max():+.3f}",
         "holds": bool(len(rel) > 0 and (rel["observed_rate"] < rel["mean_predicted"]).all()),
         "source": "metrics/ngboost_reliability.csv"},
    ]
    return pd.DataFrame(rows)
