"""English Results draft (Sections 4.1-4.5) generated from the computed tables.

Every number is inserted from the result DataFrames, and every directional statement is guarded by an
assertion: if a future rerun changes a direction, the run fails instead of printing a false sentence.
Output: outputs/aiesc2027/results/results_en.md and key_numbers.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C


def write(out: Path, d, overall, risk60, risk_off, ngb_prob, bias, low, unc, per_model, diff, fam_check):
    L = {k: C.MODELS[k]["label"] for k in C.MODEL_ORDER}
    ov = overall.set_index("key")
    rk = risk60.set_index("key")
    ro = risk_off.set_index("key")
    lo = low.set_index("model")
    iv = unc["intervals"]
    sc = unc["scores"].set_index("version")
    y = d.y
    n, n_st = len(y), int(d.frame["student_id"].nunique())
    n_pos = int((y < C.RISK_THRESHOLD).sum())
    prev = n_pos / n

    def ci(k, m):
        r = per_model[(per_model.model == k) & (per_model.metric == m)].iloc[0]
        return f"95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"

    def dd(k, m):
        return diff[(diff.model == k) & (diff.metric == m)].iloc[0]

    def dci(k, m):
        r = dd(k, m)
        return f"{r['difference']:+.3f}, 95% CI [{r['ci_low_95']:+.3f}, {r['ci_high_95']:+.3f}]"

    def cov(method, lv, group="all"):
        return iv[(iv.method == method) & (iv.nominal == lv) & (iv.group == group)].iloc[0]

    def rng(vals, fmt="{:.3f}"):
        a, b = fmt.format(min(vals)), fmt.format(max(vals))
        return a if a == b else f"{a} to {b}"

    others = [k for k in C.MODEL_ORDER if k != "lightgbm"]
    # ---- guards for every directional claim ---------------------------------------------------------
    best_mae = ov["MAE"].idxmin()
    best_rmse = ov["RMSE"].idxmin()
    assert best_mae == "lightgbm_l1" and best_rmse == "lightgbm"
    assert ov.loc["lightgbm_l1", "median_ae"] == ov["median_ae"].min()
    assert all(dd(k, "MAE")["supported_holm_0.05"] and dd(k, "RMSE")["supported_holm_0.05"] for k in others)
    assert not any(dd(k, m)["practically_important"] for k in others for m in ("MAE", "RMSE"))
    assert rk["recall"].idxmax() == "lightgbm" and rk["recall"].idxmin() == "lightgbm_l1"
    assert rk["precision"].idxmax() == "lightgbm_l1"
    assert rk["pr_auc"].idxmax() == "lightgbm" and rk["roc_auc"].idxmax() == "lightgbm"
    assert rk["f1"].idxmax() == "lightgbm" and rk["balanced_accuracy"].idxmax() == "lightgbm"
    assert rk.loc["lightgbm_l1", "recall_at_budget"] < rk.loc["lightgbm", "recall_at_budget"]
    assert rk["recall_at_budget"].idxmax() == "lightgbm"
    assert (rk["flag_rate"] < prev).all()
    for m in ("recall", "f1", "pr_auc", "roc_auc", "balanced_accuracy"):
        assert dd("lightgbm_l1", m)["direction"] == "worse" and dd("lightgbm_l1", m)["supported_holm_0.05"]
        assert dd("lightgbm_l1", m)["bonf_ci_excludes_0"]
    for m in ("recall", "f1", "pr_auc"):          # the ordering under the official thresholds is unchanged
        assert list(ro[m].sort_values().index) == list(rk[m].sort_values().index)
    pf = d.frame.assign(e_l1=np.abs(d.preds["lightgbm_l1"] - y), e_lgb=np.abs(d.preds["lightgbm"] - y))
    fold_mae = pf.groupby("fold")[["e_l1", "e_lgb"]].mean()
    assert (fold_mae["e_l1"] < fold_mae["e_lgb"]).all()
    fold_rec = {}
    for k in ("lightgbm", "lightgbm_l1"):
        fold_rec[k] = [float(((d.preds[k] < 60) & (y < 60))[d.fold == f].sum() / (y < 60)[d.fold == f].sum())
                       for f in range(C.EXPECTED_N_FOLDS)]
    assert max(fold_rec["lightgbm_l1"]) < min(fold_rec["lightgbm"])
    b = bias.set_index(["key", "grade_range"])
    mse_low = [b.loc[(k, "0-59"), "mean_signed_error"] for k in C.MODEL_ORDER]
    mse_high = [b.loc[(k, "90-100"), "mean_signed_error"] for k in C.MODEL_ORDER]
    mse_mid = [b.loc[(k, "70-79"), "mean_signed_error"] for k in C.MODEL_ORDER]
    assert min(mse_low) > 0 and max(mse_high) < 0 and max(abs(v) for v in mse_mid) < 1.5
    assert max(mse_low) == b.loc[("lightgbm_l1", "0-59"), "mean_signed_error"]
    assert max(mse_high) == b.loc[("lightgbm_l1", "90-100"), "mean_signed_error"]   # least underestimation
    assert np.median(y) > np.mean(y)                                            # left-skewed grades
    p = ngb_prob[(ngb_prob.threshold == "60") & (ngb_prob.version == "normal_raw")].iloc[0]
    assert p["brier_skill_score"] < 0 and p["mean_predicted_probability"] > prev
    assert p["pr_auc"] > rk.loc["ngboost", "pr_auc"] and p["roc_auc"] > rk.loc["ngboost", "roc_auc"]
    ft_p = dd("ft_transformer", "precision")
    assert ft_p["supported_holm_0.05"] and not ft_p["bonf_ci_excludes_0"]
    assert 28 < np.mean(mse_low) < 34
    import pandas as pd
    rel = pd.read_csv(C.OUT / "metrics" / "ngboost_reliability.csv")
    rr = rel[(rel.threshold == "60") & (rel.version == "normal_raw") & (rel.n >= 100)]
    assert (rr["observed_rate"] < rr["mean_predicted"]).all()
    raw = {lv: cov("normal_raw", lv) for lv in C.INTERVAL_LEVELS}
    assert 42 < raw[0.9]["mean_width"] < 46
    resc = {lv: cov("normal_rescaled", lv) for lv in C.INTERVAL_LEVELS}
    assert all(raw[lv]["coverage"] > lv for lv in (0.5, 0.8, 0.9))
    assert resc[0.9]["mean_width"] > raw[0.9]["mean_width"] and resc[0.9]["coverage"] > raw[0.9]["coverage"]
    cf = {m: cov(m, 0.9) for m in ("cf_absolute", "cf_normalised", "cf_mondrian")}
    assert all(abs(v["coverage"] - 0.9) < 0.005 for v in cf.values())
    assert all(v["mean_width"] < raw[0.9]["mean_width"] for v in cf.values())
    low_cov = {m: cov(m, 0.9, "0-59")["coverage"] for m in ("normal_raw", "normal_rescaled", "cf_absolute", "cf_normalised", "cf_mondrian")}
    assert max(low_cov.values()) < 0.6
    mid_cov = [cov(m, 0.9, g)["coverage"] for m in low_cov for g in ("70-79", "80-89")]
    hi_cov = [cov(m, 0.9, "90-100")["coverage"] for m in low_cov]
    cat_raw = iv[(iv.method == "normal_raw") & (iv.nominal == 0.9) & (iv.group_by == "course_category")]
    fam = fam_check.set_index("quantity")

    l1, lg = "lightgbm_l1", "lightgbm"
    rel_mae = -dd(l1, "MAE")["relative_difference"]
    rel_rec = -dd(l1, "recall")["relative_difference"]
    missed_l1 = int(rk.loc[l1, "FN"])
    fam_size = int(diff["family_size"].iat[0])

    text = f"""# 4. Results (draft for AIESC 2027)

> Generated by `research/aiesc2027/report_text.py` from `outputs/aiesc2027/`. All numbers come from the existing
> out-of-fold predictions of the IR study, re-verified for alignment. **Exploratory analysis:** the same semester of
> data was used repeatedly for model selection in the IR study, so none of the results is an independent validation.

All results refer to the unseen-student setting: five-fold cross-validation grouped by student (seed 42), applied to
{n:,d} enrolments of {n_st:,d} students in one semester (113-2). Inputs are registration-time course and student
attributes only. The five models are LightGBM, CatBoost, FT-Transformer, NGBoost and LightGBM with an absolute-error
objective (LightGBM-L1). All models are evaluated on identical rows and folds. Confidence intervals come from a paired
student-cluster bootstrap with 5,000 replicates. Comparisons against LightGBM (4 models × 9 metrics = {fam_size}
comparisons) are adjusted with the Holm procedure and with Bonferroni-adjusted intervals. With 5,000 replicates, the
smallest attainable two-sided bootstrap p-value is about 0.0004, so Holm-adjusted p-values cannot fall below about
0.014.

## 4.1 Overall Prediction Performance

Table 2 and Figure 2 summarise point-prediction accuracy. LightGBM-L1 obtained the lowest MAE
({ov.loc[l1, 'MAE']:.3f}; {ci(l1, 'MAE')}), {abs(dd(l1, 'MAE')['difference']):.3f} grade points lower than LightGBM
(difference {dci(l1, 'MAE')}), and the lowest median absolute error ({ov.loc[l1, 'median_ae']:.3f}). LightGBM obtained the
lowest RMSE ({ov.loc[lg, 'RMSE']:.3f}; {ci(lg, 'RMSE')}). LightGBM-L1's RMSE was higher than LightGBM's (difference
{dci(l1, 'RMSE')}), and its mean signed error of {ov.loc[l1, 'mean_signed_error']:+.3f} indicates systematically higher
predictions. LightGBM-L1 had a lower MAE than LightGBM in every one of the five folds. Relative to LightGBM, MAE was
higher for CatBoost ({dci('catboost', 'MAE')}), FT-Transformer ({dci('ft_transformer', 'MAE')}) and NGBoost
({dci('ngboost', 'MAE')}). All MAE and RMSE differences remained statistically supported after Holm adjustment.
However, none reached the pre-specified practical threshold of 0.5 grade points. All models compressed their
predictions toward the centre of the grade distribution: the standard deviation of the predictions was
{rng(ov['sd_ratio'])} times that of the actual grades. The MAE and RMSE values reproduce those reported by the IR study
to four decimals; this analysis does not present them as new results.

## 4.2 Academic Risk Identification

Of the {n:,d} enrolments, {n_pos:,d} ({prev:.1%}) had an actual term grade below 60. Table 3 and Figure 3 report how
well each model identified these enrolments. With the fixed rule "predicted grade < 60", LightGBM achieved the highest
recall ({rk.loc[lg, 'recall']:.3f}; {ci(lg, 'recall')}), F1 ({rk.loc[lg, 'f1']:.3f}) and balanced accuracy
({rk.loc[lg, 'balanced_accuracy']:.3f}). LightGBM-L1 achieved the highest precision ({rk.loc[l1, 'precision']:.3f}) but the
lowest recall ({rk.loc[l1, 'recall']:.3f}; {ci(l1, 'recall')}): it did not flag {missed_l1:,d} of the {n_pos:,d} low-grade
enrolments. Compared with LightGBM, LightGBM-L1 reduced recall by {abs(dd(l1, 'recall')['difference']):.3f}
({rel_rec:.1%} in relative terms; {dci(l1, 'recall')}), even though it reduced MAE by {rel_mae:.1%}. Its F1
({dci(l1, 'f1')}) and balanced accuracy ({dci(l1, 'balanced_accuracy')}) were also lower. LightGBM-L1's recall was lower
than LightGBM's in every fold (fold range {rng(fold_rec[l1])} versus {rng(fold_rec[lg])}). Every model flagged a
smaller share of enrolments ({rng(rk['flag_rate'], '{:.1%}')}) than the observed prevalence ({prev:.1%}).

Threshold-free ranking metrics led to the same conclusion. Records were ranked by the negative predicted grade; this
ranking score is not a calibrated probability. LightGBM had the highest PR-AUC ({rk.loc[lg, 'pr_auc']:.3f};
{ci(lg, 'pr_auc')}) and ROC-AUC ({rk.loc[lg, 'roc_auc']:.3f}), whereas LightGBM-L1 had a PR-AUC of
{rk.loc[l1, 'pr_auc']:.3f} ({dci(l1, 'pr_auc')}) and the lowest ROC-AUC ({rk.loc[l1, 'roc_auc']:.3f}). For all
LightGBM-L1 comparisons on recall, F1, balanced accuracy, PR-AUC and ROC-AUC, the Bonferroni-adjusted intervals exclude
zero. We also matched the alert budget to the observed prevalence by flagging the {n_pos:,d} highest-risk records. Under
this budget, recall was {rk.loc[lg, 'recall_at_budget']:.3f} for LightGBM and {rk.loc[l1, 'recall_at_budget']:.3f} for
LightGBM-L1, so LightGBM-L1's deficit is not only a matter of where the fixed threshold falls. FT-Transformer's precision
was higher than LightGBM's ({dci('ft_transformer', 'precision')}); this difference was supported by the Holm procedure
(adjusted p = {dd('ft_transformer', 'precision')['p_holm']:.3f}), but its Bonferroni-adjusted interval included zero.

NGBoost is the only model with a predictive distribution, so it is the only one for which a risk probability
P(Y < 60) can be evaluated. This probability ranked records slightly better than NGBoost's own point prediction (PR-AUC
{p['pr_auc']:.3f}, ROC-AUC {p['roc_auc']:.3f}), but it was poorly calibrated. The mean predicted probability was
{p['mean_predicted_probability']:.3f}, against a prevalence of {prev:.3f}. In every reliability bin with at least 100
records, the observed share of low grades was below the predicted probability (Figure 5d). The Brier score was
{p['brier']:.3f}, versus {p['brier_reference_train_prevalence']:.3f} for a constant forecast equal to the training-fold
prevalence (Brier skill score {p['brier_skill_score']:+.3f}), and the expected calibration error was
{p['ece_10_bins']:.3f}. In a sensitivity analysis using the official, programme-specific pass thresholds (60, or 70 for
master's and doctoral programmes; {int((y < d.frame[C.OFFICIAL_THRESHOLD_COL]).sum()):,d} positives), the model ordering
for recall, F1 and PR-AUC was unchanged (recall {rng(ro['recall'])}; PR-AUC {rng(ro['pr_auc'])}). Thresholds chosen on
held-out students, risk-targeted training objectives and the inclusion of zero grades are examined in Sections 4.2.1
and 4.2.2.

## 4.3 Grade-Dependent Prediction Bias

Table 4 and Figure 4 stratify errors by the actual grade range. These strata are defined after the fact and are
never model inputs. Because the strata condition on the actual grade, they show regression toward the mean by
construction, so range-wise errors describe where errors fall and do not measure how difficult each range is. All
five models overestimated grades in the 0–59 range: the mean signed error was {rng(mse_low, '{:+.3f}')} points, and the
mean prediction was {rng([b.loc[(k, '0-59'), 'mean_predicted'] for k in C.MODEL_ORDER])}, against a mean actual grade of
{b.loc[(lg, '0-59'), 'mean_actual']:.3f}. Depending on the model, {rng(lo['share_low_predicted_ge_60'], '{:.1%}')} of
low-grade enrolments received a prediction of 60 or above. All models underestimated grades in the 90–100 range (mean signed error
{rng(mse_high, '{:+.3f}')}), and errors were close to zero in the 70–79 range ({rng(mse_mid, '{:+.3f}')}). LightGBM-L1
showed the largest overestimation of low grades ({b.loc[(l1, '0-59'), 'mean_signed_error']:+.3f}) and the smallest
underestimation of high grades ({b.loc[(l1, '90-100'), 'mean_signed_error']:+.3f}). This is consistent with its
absolute-error objective, which targets the conditional median. Because the grade distribution is left-skewed (median
{np.median(y):.0f}, mean {np.mean(y):.3f}), the conditional median lies above the conditional mean, so median-targeting
moves predictions upward. This lowers errors for the majority of enrolments, which pass, and raises them for the
minority with low grades.

## 4.4 Predictive Uncertainty and Calibration

Table 5 and Figure 5 evaluate NGBoost's Normal predictive distribution on the student split (NLL
{sc.loc['normal_raw', 'NLL']:.3f}; CRPS {sc.loc['normal_raw', 'CRPS']:.3f}). The raw central intervals over-covered at
the 50%, 80% and 90% levels (empirical coverage {raw[0.5]['coverage']:.3f}, {raw[0.8]['coverage']:.3f} and
{raw[0.9]['coverage']:.3f}) and were close to nominal at 95% ({raw[0.95]['coverage']:.3f}). The 90% intervals had a
mean width of {raw[0.9]['mean_width']:.3f} grade points (median {raw[0.9]['median_width']:.3f}) on a 0–100 scale.
Rescaling the predicted standard deviation with factors estimated inside each training fold widened the 90% intervals
to {resc[0.9]['mean_width']:.3f} points and increased over-coverage ({resc[0.9]['coverage']:.3f}), so it did not improve
calibration. As an exploratory post-hoc check, cross-fold conformal intervals took their quantiles from the other four
(student-disjoint) folds. These intervals reached nominal marginal coverage (90% level:
{rng([v['coverage'] for v in cf.values()])} for all three variants) with narrower mean widths ({rng([v['mean_width'] for v in cf.values()])}
points). They had no separate calibration set, so the finite-sample split-conformal guarantee does not strictly apply.

Marginal coverage concealed strong conditional miscoverage. For enrolments with an actual grade below 60, the 90%
intervals covered only {low_cov['normal_raw']:.3f} (raw Normal), {low_cov['normal_rescaled']:.3f} (rescaled) and
{rng([low_cov[m] for m in ('cf_absolute', 'cf_normalised', 'cf_mondrian')])} (conformal variants) of outcomes. In the
70–89 range, coverage was at least {min(mid_cov):.3f}, and in the 90–100 range it was {rng(hi_cov)}. Across course
categories, coverage of the raw intervals ranged from {rng(cat_raw['coverage'])}; Mondrian conformal equalised category
coverage by construction but did not correct the under-coverage of low grades. Earlier IR reports quoted 91.7%
coverage and a 46.662-point width for NGBoost's 90% intervals. Recomputing from the saved per-fold files confirmed these
values ({fam.loc['coverage_90', 'recomputed']:.3f} and {fam.loc['width_90', 'recomputed']:.3f}), but they come from the
course-family split, a different partition, and are therefore not pooled with the student-split results above.
Split conformal, Mondrian conformal and CQR with an independent in-fold calibration set, and recalibrated risk
probabilities, are reported in Section 4.4.1 (follow-up with refitted models).

## 4.5 Discussion of Accuracy and Reliability Trade-offs

Three observations follow. First, a lower average error did not translate into better identification of low grades.
LightGBM-L1 reduced MAE by {rel_mae:.1%} relative to LightGBM but reduced recall for grades below 60 by {rel_rec:.1%},
and its ranking quality was also lower. These differences were statistically supported, consistent across folds and,
for recall and F1, larger than the practical threshold. The MAE differences between models, by contrast, stayed below
the practical threshold of 0.5 grade points. Most of LightGBM-L1's deficit came from where its upward-shifted
predictions met the fixed cutoff (Section 4.2.1): choosing a loss function for MAE also silently chooses an operating
point for any grade-threshold rule. Second, all five models shared the same failure mode: predictions were
compressed toward the centre, low grades were overestimated by about 30 points on average, and the fixed rule
"predicted grade < 60" missed most low-grade enrolments (recall at most {rk['recall'].max():.3f}). The best PR-AUC
({rk['pr_auc'].max():.3f}) was about {rk['pr_auc'].max() / prev:.1f} times the prevalence, so registration-time
information carries some ranking signal but limited discrimination; thresholds tuned on held-out students, a direct
classifier, quantile regression and class weighting changed this picture only marginally (Section 4.2.1). Third, NGBoost's uncertainty estimates were
conservative overall but wide (about 44 points at the 90% level), under-covered precisely the low-grade enrolments that
matter for academic risk, and produced risk probabilities that overestimated the observed risk unless they were
recalibrated on held-out students (Section 4.4.1). Together, these results
argue for reporting risk-oriented and conditional metrics alongside MAE when selecting models for academic-risk
applications.

These findings are exploratory. They come from a single semester that has been reused for model selection; they exclude
{d.counts['zero_rows']:,d} zero-grade records whose meaning is unknown (all of which would fall below 60); and they use
a fixed research threshold of 60, which is not identical to every programme's official pass threshold. No early-warning
system was deployed, no advising intervention was evaluated, and no claim is made about future semesters or about
changes in student achievement. A zero-inclusive sensitivity analysis with refitted models is reported in Section
4.2.2. [TO BE COMPLETED: validation on a later semester, which requires new data.]
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_en.md").write_text(text, encoding="utf-8")
    key = {"n": n, "students": n_st, "n_below_60": n_pos, "prevalence": prev,
           "mae": ov["MAE"].to_dict(), "rmse": ov["RMSE"].to_dict(), "recall": rk["recall"].to_dict(),
           "precision": rk["precision"].to_dict(), "f1": rk["f1"].to_dict(), "pr_auc": rk["pr_auc"].to_dict(),
           "roc_auc": rk["roc_auc"].to_dict(), "recall_at_budget": rk["recall_at_budget"].to_dict(),
           "fold_recall": fold_rec, "ngboost_prob60": p.to_dict(), "coverage90_low": low_cov,
           "coverage90_all": {m: float(cov(m, 0.9)["coverage"]) for m in low_cov},
           "width90_all": {m: float(cov(m, 0.9)["mean_width"]) for m in low_cov}}
    (out / "key_numbers.json").write_text(json.dumps(key, indent=2, default=float), encoding="utf-8")
