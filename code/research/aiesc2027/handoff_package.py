"""Build the AIESC 2027 data hand-off package for ChatGPT (no new experiments, no edits to existing results).

Reads outputs/aiesc2027/ and the IR inputs (read-only); writes only
  outputs/aiesc2027/handoff_to_chatgpt/
  outputs/aiesc2027/AIESC2027_data_handoff_v1.zip (+ .sha256 and _verification.json beside it)

Usage (pinned environment, see README section 14):
  $PY -B research/aiesc2027/handoff_package.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.aiesc2027 import config as C  # noqa: E402
from research.aiesc2027 import handoff_checks as K  # noqa: E402
from research.aiesc2027 import handoff_values as H  # noqa: E402
from research.aiesc2027 import integrity  # noqa: E402
from research.aiesc2027 import paper_package as P  # noqa: E402

OUT = C.OUT
HANDOFF = OUT / "handoff_to_chatgpt"
ZIP = OUT / "AIESC2027_data_handoff_v1.zip"
ZIP_VERIFY = OUT / "AIESC2027_data_handoff_v1_verification.json"
TEMPLATES = Path(__file__).resolve().parent / "handoff_templates"
CODE_DIR = Path(__file__).resolve().parent
PY = sys.executable
log = logging.getLogger("aiesc2027.handoff")

SHA_IR = C.EXPECTED_SHA256["IR.csv"][:12]
SHA_CLEAN = C.EXPECTED_SHA256["cleaned_grades.csv"][:12]
DV_A = f"113-2, zeros excluded (n = 83,494); IR.csv sha256 {SHA_IR}..., cleaned_grades.csv sha256 {SHA_CLEAN}..."
DV_B = f"113-2, zero grades included (n = 85,567); IR.csv sha256 {SHA_IR}..., cleaned_grades.csv sha256 {SHA_CLEAN}..."
TC_OOF = "Original IR out-of-fold prediction: model fitted on the full training fold (other 4 folds); frozen hyperparameters"
TC_80 = ("AIESC refit on the proper-training part (80% of the training-fold students); the other 20% of students "
         "form the calibration set (thresholds / conformal quantiles / recalibration only)")
TC_80B = TC_80 + "; zero grades included in fit and calibration"
SPLIT = "Student-grouped 5-fold CV (saved IR T3 test indices), pooled out-of-fold over 5 test folds"
SPLIT_B = SPLIT + "; zero-grade rows follow their student (50 zero-only students assigned with rng(42))"

MASTER_COLUMNS = [
    "Experiment ID", "Experiment Type", "Population", "Model", "Operating Point", "Training Configuration",
    "Evaluation Split", "Dataset Version", "Sample Size", "MAE", "RMSE", "Mean Signed Error", "Precision", "Recall", "F1",
    "PR-AUC", "ROC-AUC", "Flag Rate", "Calibration Metric", "Calibration Value", "Brier", "Brier Skill Score", "NLL",
    "Interval Nominal Level", "Interval Subgroup", "Prediction Interval Coverage", "Prediction Interval Width",
    "Baseline Model", "Compared Metric", "Difference", "Confidence Interval", "Holm p", "Practical Threshold",
    "Statistical Status", "Completion Status", "Results Section", "Paper Item", "Source File", "Notes"]


# ---- helpers ----------------------------------------------------------------------------------------------
def r3(x):
    if x is None:
        return "NA"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(v):
        return "NA"
    return f"{v:.3f}".replace("-0.000", "0.000")


def s3(x):
    s = f"{float(x):+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s


def pval(x):
    v = float(x)
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def ci(lo, hi):
    return f"[{r3(lo)}, {r3(hi)}]"


def sci(lo, hi):
    return f"[{s3(lo)}, {s3(hi)}]"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tab(rel: str) -> pd.DataFrame:
    return H.table(rel)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def row(**kw) -> dict:
    out = {c: "NA" for c in MASTER_COLUMNS}
    for k, v in kw.items():
        key = k.replace("_", " ")
        match = next((c for c in MASTER_COLUMNS if c.lower().replace("-", " ") == key.lower().replace("-", " ")), None)
        if match is None:
            raise KeyError(k)
        out[match] = v if isinstance(v, str) else r3(v)
    return out


# ---- MASTER_RESULTS -----------------------------------------------------------------------------------------
def master_results() -> pd.DataFrame:
    rows = []
    ov, rk, ro = tab("metrics/overall.csv"), tab("metrics/risk_threshold60.csv"), tab("metrics/risk_official_threshold.csv")
    for _, o in ov.iterrows():
        k = rk[rk["key"] == o["key"]].iloc[0]
        rows.append(row(experiment_id="E1-E2", experiment_type="Overall error and risk identification (re-analysis of saved OOF)",
                        population="A (n = 83,494)", model=o["model"], operating_point="predicted grade < 60",
                        training_configuration=TC_OOF, evaluation_split=SPLIT, dataset_version=DV_A,
                        sample_size=str(int(o["n"])), mae=o["MAE"], rmse=o["RMSE"], mean_signed_error=o["mean_signed_error"],
                        precision=k["precision"], recall=k["recall"], f1=k["f1"], pr_auc=k["pr_auc"], roc_auc=k["roc_auc"],
                        flag_rate=k["flag_rate"], statistical_status="Descriptive; 95% CIs in stats/per_model_bootstrap_ci.csv",
                        completion_status="Complete", results_section="4.1, 4.2", paper_item="Table 2, Table 3, Figure 2, Figure 3",
                        source_file="metrics/overall.csv; metrics/risk_threshold60.csv",
                        notes="Ranking metrics use score = -predicted grade (not a probability)"))
        rows.append(row(experiment_id="E2b", experiment_type="Risk identification at an equal flag budget (descriptive, test-fold prevalence)",
                        population="A (n = 83,494)", model=o["model"], operating_point="top 7,685 risk scores (budget = observed positives)",
                        training_configuration=TC_OOF, evaluation_split=SPLIT, dataset_version=DV_A, sample_size=str(int(o["n"])),
                        recall=k["recall_at_budget"], statistical_status="Descriptive (no CI); budget uses the observed test prevalence",
                        completion_status="Complete", results_section="4.2", paper_item="Text only",
                        source_file="metrics/risk_threshold60.csv (recall_at_budget)",
                        notes="Oracle budget: not a deployable rule; the deployable analogue is OP1 in E8"))
        k2 = ro[ro["key"] == o["key"]].iloc[0]
        rows.append(row(experiment_id="E3", experiment_type="Sensitivity: official programme pass threshold (60 or 70)",
                        population="A (n = 83,494)", model=o["model"], operating_point="predicted grade < official threshold",
                        training_configuration=TC_OOF, evaluation_split=SPLIT, dataset_version=DV_A, sample_size=str(int(o["n"])),
                        precision=k2["precision"], recall=k2["recall"], f1=k2["f1"], pr_auc=k2["pr_auc"], roc_auc=k2["roc_auc"],
                        flag_rate=k2["flag_rate"], statistical_status="Descriptive (no CI)", completion_status="Complete",
                        results_section="4.2", paper_item="Text only", source_file="metrics/risk_official_threshold.csv",
                        notes=f"{int(k2['TP'] + k2['FN']):,d} positives (official threshold 70 for 2,700 graduate records)"))
    pv = tab("stats/paired_vs_lightgbm.csv")
    for _, p in pv.iterrows():
        sup, prac = bool(p["supported_holm_0.05"]), bool(p["practically_important"])
        status = ("Holm-supported" if sup else "Not supported after Holm") + ("; |difference| >= practical threshold" if prac
                                                                              else "; below practical threshold")
        rows.append(row(experiment_id="E4", experiment_type="Paired comparison vs LightGBM (original OOF; family of 36, Holm)",
                        population="A (n = 83,494)", model=H.MODEL_LABEL[p["model"]], training_configuration=TC_OOF,
                        evaluation_split=SPLIT, dataset_version=DV_A, sample_size="83494", baseline_model="LightGBM",
                        compared_metric=p["metric"], difference=p["difference"], confidence_interval=ci(p["ci_low_95"], p["ci_high_95"]),
                        holm_p=pval(p["p_holm"]), practical_threshold=p["practical_threshold"],
                        statistical_status=f"{status}; direction: {p['direction']}", completion_status="Complete",
                        results_section="4.1, 4.2", paper_item="Table S2", source_file="stats/paired_vs_lightgbm.csv",
                        notes=f"Bonferroni 95% CI {ci(p['ci_low_bonf'], p['ci_high_bonf'])}; difference = model - LightGBM"))
    gb = tab("metrics/grade_range_bias.csv")
    for _, g in gb.iterrows():
        rows.append(row(experiment_id="E5", experiment_type="Grade-range bias (strata of the actual grade; post hoc)",
                        population=f"A, actual grade {g['grade_range']}", model=g["model"], training_configuration=TC_OOF,
                        evaluation_split=SPLIT, dataset_version=DV_A, sample_size=str(int(g["n"])), mae=g["MAE"], rmse=g["RMSE"],
                        mean_signed_error=g["mean_signed_error"], statistical_status="Descriptive (no CI)",
                        completion_status="Complete", results_section="4.3", paper_item="Table 6, Figure 5",
                        source_file="metrics/grade_range_bias.csv",
                        notes="Signed error = predicted - actual; regression toward the mean by construction"))
    ni = tab("metrics/ngboost_intervals.csv")
    for _, i in ni[(ni["group_by"] == "all") | ((ni["group_by"] == "grade_range") & (ni["nominal"] == 0.9))].iterrows():
        rows.append(row(experiment_id="E6", experiment_type="NGBoost predictive intervals (original OOF)",
                        population="A (n = 83,494)" if i["group_by"] == "all" else f"A, actual grade {i['group']}",
                        model=f"NGBoost ({i['method']})", training_configuration=TC_OOF, evaluation_split=SPLIT, dataset_version=DV_A,
                        sample_size=str(int(i["n"])), interval_nominal_level=i["nominal"],
                        interval_subgroup="all" if i["group_by"] == "all" else f"actual {i['group']}",
                        prediction_interval_coverage=i["coverage"], prediction_interval_width=i["mean_width"],
                        statistical_status="Descriptive (no CI)", completion_status="Complete", results_section="4.4",
                        paper_item="Table 7, Figure 6", source_file="metrics/ngboost_intervals.csv",
                        notes="cf_* = post-hoc cross-fold conformal without a separate calibration set"))
    npb, ns = tab("metrics/ngboost_probability.csv"), tab("metrics/ngboost_scores.csv")
    for _, q in npb[npb["threshold"].astype(str) == "60"].iterrows():
        s = ns[ns["version"] == q["version"]].iloc[0]
        rows.append(row(experiment_id="E6", experiment_type="NGBoost P(Y < 60) calibration (original OOF)",
                        population="A (n = 83,494)", model=f"NGBoost P(Y<60) ({q['version']})", operating_point="probability >= 0.5",
                        training_configuration=TC_OOF, evaluation_split=SPLIT, dataset_version=DV_A, sample_size=str(int(q["n"])),
                        precision=q["precision_at_p0.5"], recall=q["recall_at_p0.5"], f1=q["f1_at_p0.5"], pr_auc=q["pr_auc"],
                        roc_auc=q["roc_auc"], calibration_metric="ECE (10 equal-width bins)", calibration_value=q["ece_10_bins"],
                        brier=q["brier"], brier_skill_score=q["brier_skill_score"], nll=s["NLL"],
                        statistical_status="Descriptive (no CI)", completion_status="Complete", results_section="4.2, 4.4",
                        paper_item="Table 7, Figure 6d", source_file="metrics/ngboost_probability.csv; metrics/ngboost_scores.csv",
                        notes=f"mean predicted probability {r3(q['mean_predicted_probability'])} vs prevalence {r3(q['prevalence'])}; "
                              "BSS reference = training-fold prevalence; NLL is of the predictive Normal (grade scale)"))
    pa = tab("calibration_split/metrics/point_accuracy.csv")
    for _, a in pa.iterrows():
        rows.append(row(experiment_id="F1-R", experiment_type="Reproduction gate and 80% refit point accuracy",
                        population="A (n = 83,494)", model=a["model"], operating_point="predicted grade < 60",
                        training_configuration=TC_OOF if "original" in a["model"] else (
                            "AIESC refit on the full saved training fold (reproduction check)" if "reproduction" in a["model"] else TC_80),
                        evaluation_split=SPLIT, dataset_version=DV_A, sample_size="83494", mae=a["MAE"], rmse=a["RMSE"],
                        precision=a["precision"], recall=a["recall"], f1=a["f1"], statistical_status="Descriptive (no CI)",
                        completion_status="Complete", results_section="4.4.1", paper_item="Table S11, Table S13",
                        source_file="calibration_split/metrics/point_accuracy.csv",
                        notes="LightGBM reproduction exact; NGBoost reproduction approximate (inner GroupKFold tie order)"))
    ci_ = tab("calibration_split/metrics/intervals.csv")
    sel = ci_[(ci_["group_by"] == "all") | ((ci_["group_by"] == "grade_range") & (ci_["nominal"] == 0.9))]
    for _, i in sel.iterrows():
        rows.append(row(experiment_id="F1-I", experiment_type="Intervals with an in-fold calibration set (refit 80%)",
                        population="A (n = 83,494)" if i["group_by"] == "all" else f"A, actual grade {i['group']}",
                        model=i["method_label"], training_configuration=TC_80, evaluation_split=SPLIT, dataset_version=DV_A,
                        sample_size=str(int(i["n"])), interval_nominal_level=i["nominal"],
                        interval_subgroup="all" if i["group_by"] == "all" else f"actual {i['group']}",
                        prediction_interval_coverage=i["coverage"], prediction_interval_width=i["mean_width"],
                        statistical_status="Descriptive; 95% CIs (90% level) in calibration_split/stats/bootstrap_ci.csv",
                        completion_status="Complete", results_section="4.4.1", paper_item="Table 8, Figure 7, Tables S7-S10",
                        source_file="calibration_split/metrics/intervals.csv",
                        notes=f"share of actual below interval {r3(i['share_actual_below_interval'])}"))
    pr = tab("calibration_split/metrics/probability_recalibration.csv")
    for _, q in pr.iterrows():
        rows.append(row(experiment_id="F1-P", experiment_type="P(Y < 60) recalibration on the calibration set",
                        population="A (n = 83,494)", model=q["label"], training_configuration=TC_OOF if q["version"] == "original_full_fold" else TC_80,
                        evaluation_split=SPLIT, dataset_version=DV_A, sample_size=str(int(q["n"])), pr_auc=q["pr_auc"],
                        roc_auc=q["roc_auc"], calibration_metric="ECE (10 equal-width bins)", calibration_value=q["ece_10_bins"],
                        brier=q["brier"], brier_skill_score=q["brier_skill_score"], statistical_status="Descriptive; Brier CIs in bootstrap_ci.csv",
                        completion_status="Complete", results_section="4.4.1", paper_item="Table 9, Figure 7c",
                        source_file="calibration_split/metrics/probability_recalibration.csv",
                        notes=f"mean predicted probability {r3(q['mean_predicted_probability'])} vs prevalence {r3(q['prevalence'])}"))
    bc = tab("calibration_split/stats/bootstrap_ci.csv")
    for _, b in bc[bc["statistic"] == "brier_difference"].iterrows():
        rows.append(row(experiment_id="F1-P", experiment_type="P(Y < 60) recalibration: paired Brier difference",
                        population="A (n = 83,494)", model=b["method"].split(" - ")[0], training_configuration=TC_80,
                        evaluation_split=SPLIT, dataset_version=DV_A, sample_size="83494", baseline_model="refit_raw",
                        compared_metric="Brier score", difference=b["estimate"], confidence_interval=ci(b["ci_low"], b["ci_high"]),
                        statistical_status="Descriptive paired student-cluster bootstrap CI (pre-specified: no hypothesis family)",
                        completion_status="Complete", results_section="4.4.1", paper_item="Table 9",
                        source_file="calibration_split/stats/bootstrap_ci.csv", notes="negative = lower (better) Brier score"))
    for pop, rel, dv, tc, split, sec, item in (
            ("A (n = 83,494)", "risk_methods/metrics/A_metrics.csv", DV_A, TC_80, SPLIT, "4.2.1", "Table 4, Figure 4a-b"),
            ("B, all rows (n = 85,567)", "risk_methods/metrics/B_metrics_all_rows.csv", DV_B, TC_80B, SPLIT_B, "4.2.2", "Table 5, Figure 4c"),
            ("B models, non-zero rows (n = 83,494)", "risk_methods/metrics/B_metrics_nonzero_rows.csv", DV_B, TC_80B, SPLIT_B, "4.2.2", "Table 5")):
        for _, m in tab(rel).iterrows():
            rows.append(row(experiment_id="F2-A" if pop.startswith("A") else "F2-B",
                            experiment_type="Risk-oriented rules and objectives" + ("" if pop.startswith("A") else " (zero-grade sensitivity)"),
                            population=pop, model=m["label"], operating_point=m["operating_point"], training_configuration=tc,
                            evaluation_split=split, dataset_version=dv, sample_size=str(int(m["n"])), mae=m["MAE"], rmse=m["RMSE"],
                            precision=m["precision"], recall=m["recall"], f1=m["f1"], pr_auc=m["pr_auc"], roc_auc=m["roc_auc"],
                            flag_rate=m["flag_rate"],
                            statistical_status=f"Descriptive; 95% CIs in risk_methods/stats/{pop[0]}_per_method_ci.csv",
                            completion_status="Complete", results_section=sec, paper_item=item, source_file=rel,
                            notes="OP1: flag share = calibration prevalence; OP2: calibration recall >= 0.50; thresholds from calibration students"))
    zf = tab("risk_methods/metrics/B_zero_rows_flagged.csv")
    for _, z in zf.iterrows():
        rows.append(row(experiment_id="F2-B", experiment_type="Zero-grade rows flagged (sensitivity)", population="B, zero-grade rows only",
                        model=z["method"], operating_point=z["operating_point"], training_configuration=TC_80B, evaluation_split=SPLIT_B,
                        dataset_version=DV_B, sample_size=str(int(z["n_zero_rows"])), flag_rate=z["share_flagged"],
                        statistical_status="Descriptive (no CI)", completion_status="Complete", results_section="4.2.2",
                        paper_item="Table 5, Figure 4c", source_file="risk_methods/metrics/B_zero_rows_flagged.csv",
                        notes="All zero-grade rows are positives (grade < 60), so the share flagged equals recall on these rows"))
    for pop, rel, dv in (("A (n = 83,494)", "risk_methods/metrics/A_probabilities.csv", DV_A),
                         ("B (n = 85,567)", "risk_methods/metrics/B_probabilities.csv", DV_B)):
        for _, q in tab(rel).iterrows():
            rows.append(row(experiment_id="F2-A" if pop.startswith("A") else "F2-B", experiment_type="Recalibrated risk probabilities",
                            population=pop, model=q["label"], training_configuration=TC_80 if pop.startswith("A") else TC_80B,
                            evaluation_split=SPLIT, dataset_version=dv, sample_size=pop.split("= ")[1].rstrip(")").replace(",", ""),
                            calibration_metric="ECE (10 equal-width bins)", calibration_value=q["ece_10_bins"], brier=q["brier"],
                            brier_skill_score=q["brier_skill_score"], statistical_status="Descriptive (no CI)",
                            completion_status="Complete", results_section="4.2.1" if pop.startswith("A") else "4.2.2",
                            paper_item="Table S6", source_file=rel,
                            notes=f"mean probability {r3(q['mean_probability'])} vs prevalence {r3(q['prevalence'])}"))
    ac = tab("risk_methods/stats/A_contrasts.csv")
    for _, c in ac.iterrows():
        sup, imp, prac = bool(c["supported_holm_0.05"]), bool(c["improves"]), bool(c["practically_important"])
        status = (("Holm-supported" if sup else "Not supported after Holm") + ("; |difference| >= 0.05" if prac else "; below 0.05")
                  + ("; meets 'improves' criterion" if imp else ""))
        rows.append(row(experiment_id="F2-C", experiment_type=f"Paired contrast ({c['family']} family, Holm within family of {int(c['family_size'])})",
                        population="A (n = 83,494)", model=c["candidate"], training_configuration=TC_80, evaluation_split=SPLIT,
                        dataset_version=DV_A, sample_size="83494", baseline_model=c["reference"], compared_metric=c["contrast"],
                        difference=c["difference"], confidence_interval=ci(c["ci_low_95"], c["ci_high_95"]), holm_p=pval(c["p_holm"]),
                        practical_threshold="0.050", statistical_status=status, completion_status="Complete",
                        results_section="4.2.1", paper_item="Table S3", source_file="risk_methods/stats/A_contrasts.csv",
                        notes=f"Bonferroni 95% CI {ci(c['ci_low_bonf'], c['ci_high_bonf'])}; difference = candidate - reference"))
    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    df.insert(0, "Row ID", [f"R{i + 1:04d}" for i in range(len(df))])
    return df


def metric_definitions() -> pd.DataFrame:
    d = [
        ("MAE", "mean |actual - predicted| (grade points, 0-100 scale)", "lower is better", "E1, E5, F1-R, F2"),
        ("RMSE", "sqrt(mean (actual - predicted)^2)", "lower is better", "E1, E5, F1-R, F2"),
        ("Mean Signed Error", "mean (predicted - actual); > 0 = over-estimation", "closer to 0 is better", "E1, E5"),
        ("Precision", "TP / flagged; positive = actual term grade < 60", "higher is better", "E2, E3, F2"),
        ("Recall", "TP / actual positives (grade < 60)", "higher is better", "E2, E3, F2"),
        ("F1", "harmonic mean of precision and recall", "higher is better", "E2, E3, F2"),
        ("PR-AUC", "area under the precision-recall curve of the risk score (tie-aware); baseline = prevalence (0.092 in A, 0.114 in B)",
         "higher is better", "E2, E3, F1-P, F2"),
        ("ROC-AUC", "area under the ROC curve of the risk score; 0.5 = random", "higher is better", "E2, E3, F1-P, F2"),
        ("Flag Rate", "share of enrolments flagged as at risk", "context-dependent (alert budget)", "E2, F2"),
        ("Risk score", "regressions: -predicted grade (ranking only, not a probability); classifier / NGBoost: probability",
         "-", "E2, F2"),
        ("Brier", "mean (p - 1[grade < 60])^2", "lower is better", "E6, F1-P, F2"),
        ("Brier Skill Score", "1 - Brier / Brier of a constant forecast equal to the training-fold prevalence", "higher is better; < 0 = worse than constant",
         "E6, F1-P, F2"),
        ("ECE (10 equal-width bins)", "sum over bins of share x |mean predicted - observed rate|", "lower is better", "E6, F1-P, F2"),
        ("NLL", "mean negative log-likelihood of the NGBoost Normal predictive distribution (grade scale)", "lower is better", "E6"),
        ("CRPS", "continuous ranked probability score of the NGBoost Normal (grade points)", "lower is better", "E6"),
        ("Prediction Interval Coverage", "share of actual grades inside the central interval", "closest to the nominal level is best",
         "E6, F1-I"),
        ("Prediction Interval Width", "mean (upper - lower), intervals clipped to [0, 100]", "narrower is better at equal coverage", "E6, F1-I"),
        ("Difference", "candidate (or model) minus baseline, same rows, paired student-cluster bootstrap", "sign interpreted with the metric direction",
         "E4, F1-P, F2-C"),
        ("Holm p", "two-sided bootstrap p-value, Holm-adjusted within the stated family", "< 0.05 = supported", "E4, F2-C"),
        ("Practical Threshold", "pre-specified minimum important difference: 0.5 grade points (MAE / RMSE), 0.05 (classification / ranking metrics)",
         "-", "E4, F2-C"),
        ("Improves (F2-C)", "Holm p < 0.05 AND |difference| >= 0.05 AND difference in the favourable direction", "-", "F2-C"),
    ]
    return pd.DataFrame(d, columns=["Metric", "Definition", "Direction", "Experiments"])


# ---- inventory ------------------------------------------------------------------------------------------------
def experiment_inventory() -> pd.DataFrame:
    status_main = json.loads((OUT / "logs/run_status.json").read_text())["status"]
    status_cal = json.loads((OUT / "calibration_split/logs/run_status.json").read_text())["status"]
    status_risk = json.loads((OUT / "risk_methods/logs/run_status.json").read_text())["status"]
    ok = lambda s: "Completed" if s == "complete" else f"Status {s}"  # noqa: E731
    models5 = ("LightGBM (frozen, L2), CatBoost (frozen, RMSE), FT-Transformer (Amendment 1 M2), NGBoost (Amendment 1 M3, Normal), "
               "LightGBM-L1 (C5); parameters in EXPERIMENT_METHODS.md section 3")
    rows = [
        ("E0", "Data and alignment audit", ok(status_main), "validation/alignment_checks.csv; validation/data_counts.json",
         "run_all.py (io_data.py)", "Input checks", DV_A, models5, "Saved T3 student-split folds", "Yes (78 checks PASS)",
         "Yes (deterministic; rerun 2026-11-09)", "Methods (data)", "None"),
        ("E1", "Overall prediction error", ok(status_main), "metrics/overall.csv; stats/per_model_bootstrap_ci.csv",
         "run_all.py", "RQ1", DV_A, models5, SPLIT, "Yes", "Yes (MAE / RMSE reproduce the IR registry to 4 decimals)",
         "4.1; Table 2; Figure 2", "Values are not new: they reproduce the IR study"),
        ("E2", "Academic risk identification (grade < 60)", ok(status_main), "metrics/risk_threshold60.csv; metrics/pr_curves.csv",
         "run_all.py", "RQ1", DV_A, models5, SPLIT, "Yes", "Yes", "4.2; Table 3; Figure 3",
         "Budget-matched recall uses the observed test prevalence (descriptive)"),
        ("E3", "Official pass-threshold sensitivity", ok(status_main), "metrics/risk_official_threshold.csv", "run_all.py",
         "RQ1 sensitivity", DV_A, models5, SPLIT, "Yes", "Yes", "4.2 (text)", "No CI computed"),
        ("E4", "Paired bootstrap comparisons vs LightGBM", ok(status_main), "stats/paired_vs_lightgbm.csv; stats/bootstrap_replicates.npz",
         "run_all.py (bootstrap.py)", "RQ1", DV_A, models5, SPLIT + "; 5,000 student-cluster replicates, seed 42", "Yes",
         "Yes (seeded)", "4.1-4.2; Table S2", "Holm floor 0.014 (36 contrasts x 0.0004)"),
        ("E5", "Grade-range bias", ok(status_main), "metrics/grade_range_bias.csv; metrics/low_high_grade_detail.csv",
         "run_all.py", "RQ2", DV_A, models5, SPLIT, "Yes", "Yes", "4.3; Table 6; Figure 5", "No CI for range-wise errors"),
        ("E6", "NGBoost uncertainty and probability calibration (original OOF)", ok(status_main),
         "metrics/ngboost_*.csv; metrics/crosscheck_ir_*.csv", "run_all.py (uncertainty.py)", "RQ3", DV_A,
         "NGBoost (Amendment 1 M3): raw Normal, rescaled Normal, cross-fold conformal (absolute / normalised / Mondrian)", SPLIT,
         "Yes", "Yes (NLL / coverage reproduce IR to 1e-4)", "4.4; Table 7; Figure 6; Figures S1-S2",
         "Cross-fold conformal is post hoc (no separate calibration set)"),
        ("F1-R", "Reproduction gate (refit on full saved training folds)", ok(status_cal),
         "calibration_split/metrics/reproduction_gate.csv; calibration_split/jobs/fold*.json", "calibration_fit.py",
         "Follow-up 1 prerequisite", DV_A, "Frozen LightGBM; NGBoost (Amendment 1 settings)", SPLIT, "Yes",
         "LightGBM exact; NGBoost approximate", "4.4.1; Tables S11, S13",
         "NGBoost inner GroupKFold tie order is CPU-dependent: MAE 9.856 vs saved 9.843"),
        ("F1-I", "Split / Mondrian conformal and CQR with in-fold calibration set", ok(status_cal),
         "calibration_split/metrics/intervals.csv; calibration_split/stats/bootstrap_ci.csv",
         "calibration_fit.py + run_calibration.py", "RQ3 follow-up", DV_A,
         "NGBoost, LightGBM-L2 and 8 LightGBM quantile models refitted on 80% of training-fold students", SPLIT,
         "Yes (36 checks PASS)", "Yes from saved fold jobs; refits need the pinned venv and sequential runs",
         "4.4.1; Table 8; Figure 7; Tables S7-S10, S12", "Refit models differ from the original OOF models"),
        ("F1-P", "Probability recalibration (isotonic, Platt)", ok(status_cal),
         "calibration_split/metrics/probability_recalibration.csv", "run_calibration.py", "RQ3 follow-up", DV_A,
         "Refitted NGBoost P(Y < 60)", SPLIT, "Yes", "Yes", "4.4.1; Table 9; Figure 7c", "Monotone maps: ranking unchanged by design"),
        ("F2-A", "Risk-oriented decision rules and training objectives", ok(status_risk),
         "risk_methods/metrics/A_*.csv; risk_methods/stats/A_*.csv", "risk_fit.py + run_risk.py", "RQ1/RQ4 follow-up", DV_A,
         "LightGBM L2 / L1 / binary classifier / q0.25 quantile / class-weighted L2; NGBoost P(Y<60); all refitted on 80%",
         SPLIT, "Yes (31 checks PASS)", "Yes from saved fold jobs", "4.2.1; Table 4; Figure 4; Tables S3-S6",
         "CatBoost and FT-Transformer not refitted"),
        ("F2-B", "Zero-grade sensitivity (zero-inclusive frame)", ok(status_risk),
         "risk_methods/metrics/B_*.csv; risk_methods/stats/B_per_method_ci.csv", "risk_fit.py + run_risk.py", "Sensitivity", DV_B,
         "Same six methods, refitted with zero grades", SPLIT_B, "Yes (23 checks PASS)", "Yes from saved fold jobs",
         "4.2.2; Table 5; Figure 4c", "CatBoost / FT-Transformer not refitted; meaning of zero grades unknown"),
        ("P1", "Results reference text and unified numbering", "Completed", "paper/results_en_full.md; paper/numbering_map.csv",
         "paper_package.py", "Packaging", "-", "-", "-", "Yes", "Yes", "Whole chapter",
         "Reference text only; not the submission manuscript"),
        ("X1", "Risk probabilities for LightGBM, CatBoost, FT-Transformer, LightGBM-L1 (original OOF)", "Not Available",
         "metrics/not_available.csv", "-", "RQ3", DV_A, "-", "-", "No", "-", "Stated in 4.2",
         "Point regressors have no predictive distribution"),
        ("X2", "CatBoost and FT-Transformer in the follow-ups", "Skipped (pre-specified)", "risk_methods_spec.json (not_included)",
         "-", "Follow-ups", "-", "-", "-", "No", "-", "Stated in 4.2.1 / README", "Compute cost on the shared host"),
        ("X3", "TabM on the student split", "Not Available (IR study incomplete)", "metrics/not_available.csv", "-", "-", "-", "TabM",
         "-", "No", "-", "Not reported", "IR budget exhausted before T3"),
        ("X4", "Cross-semester / prospective validation", "Not Available", "metrics/not_available.csv", "-", "External validity",
         "-", "-", "-", "No", "-", "Placeholder in 4.5", "Only 447 enrolments outside 113-2 in IR.csv"),
        ("X5", "CQR around NGBoost / FT-Transformer quantiles", "Not run (not pre-specified)",
         "calibration_split/metrics/not_available.csv", "-", "RQ3", "-", "-", "-", "No", "-", "Not reported", "-"),
        ("X6", "Conditional (per actual grade) coverage guarantee", "Not Available (not defined)",
         "calibration_split/metrics/not_available.csv", "-", "RQ3", "-", "-", "-", "No", "-", "Explained in 4.4.1",
         "Actual grade unknown at prediction time"),
        ("X7", "Family-split (unseen course family) conformal results", "Not comparable; excluded (EAIT scope)",
         "metrics/crosscheck_ir_family_T1.csv", "IR research/family", "Context only", "-", "-", "Course-family split", "-", "-",
         "Quoted once in 4.4 as context", "Different split and population of folds; not pooled"),
        ("D1", "Recomputing the GroupKFold partition on this host", "Failed (deviation recorded)",
         "calibration_split_spec.json (outer_folds); calibration_split/jobs/fold*.json", "calibration_fit.py", "-", "-", "-", "-",
         "-", "No: tie order of numpy argsort is CPU-dependent", "Methods (deviation)", "Saved test indices used instead"),
        ("D2", "Parallel per-fold fitting of follow-up 1", "Aborted (stalled; OpenMP oversubscription)", "not retained",
         "calibration_fit.py", "-", "-", "-", "-", "No", "-", "No", "Rerun sequentially; only sequential outputs are used"),
        ("D3", "Main analysis in conda env 'zhe' (scikit-learn 1.8.0)", "Superseded by pinned-venv rerun", "logs/environment.json",
         "run_all.py", "-", DV_A, "-", "-", "Yes", "Identical results reported in README", "-", "-"),
    ]
    cols = ["Experiment ID", "Experiment", "Status", "Files (relative to outputs/aiesc2027)", "Created By",
            "Research Question", "Dataset Version", "Models and Parameters", "Validation Scenario", "Completed / Checks",
            "Reproducible", "In Integrated Results", "Open Issues"]
    df = pd.DataFrame(rows, columns=cols)
    df["Evidence Exists"] = [all(any(OUT.glob(p)) for p in (s.strip() for s in f.split(";")) if p.split("/")[0] in OUTPUT_DIRS)
                             if any(s.strip().split("/")[0] in OUTPUT_DIRS for s in f.split(";")) else "NA (no output file)"
                             for f in df[cols[3]]]
    return df


OUTPUT_DIRS = {"metrics", "stats", "validation", "calibration_split", "risk_methods", "paper", "logs", "tables", "figures"}


# ---- consistency between versions ---------------------------------------------------------------------------
def consistency_table(reg: H.Registry) -> pd.DataFrame:
    v = reg.text()
    irz = pd.read_csv(ROOT / "outputs/research/final/zero_grade_sensitivity.csv", encoding="utf-8-sig")
    lz = irz[(irz["split"] == "student") & (irz["system"] == "frozen:lightgbm:cat")].iloc[0]
    cs = tab("metrics/crosscheck_ir_student_T3.csv")
    cf = tab("metrics/crosscheck_ir_family_T1.csv")
    rows = [
        ("Five-model MAE / RMSE (student split)", "IR registry outputs/research/final/comparison_student.csv",
         "AIESC recomputation from saved OOF (validation/alignment_checks.csv)", "identical to 4 decimals",
         "same predictions", "No", "AIESC values (= IR values)"),
        ("NGBoost NLL / coverage / width (student split, T3)", "IR outputs/research/final/ngboost_uncertainty.csv",
         f"AIESC recomputation (max |diff| {cs['abs_diff'].max():.1e}, metrics/crosscheck_ir_student_T3.csv)",
         "rounding of the IR table only", "same predictions", "No", "AIESC values"),
        ("NGBoost 90% coverage / width quoted in earlier IR reports (91.7%, 46.662)",
         "IR family-split result (T1)", f"student split: {v['ni_normal_raw_cov90']} / {v['ni_normal_raw_width90']}",
         f"different split (course family vs student); family recomputation max |diff| {cf['abs_diff'].max():.1e}",
         "different evaluation", "No", "Student split for AIESC; family split only as context (EAIT scope)"),
        ("Recomputed GroupKFold partition vs saved T3 partition", "saved test_idx (Amendment 1 jobs)",
         "recomputed on this host: different", "numpy unstable argsort tie order depends on CPU SIMD kernel",
         "-", "No", "Saved partition (authoritative, used everywhere)"),
        ("LightGBM reproduction refit vs saved OOF", f"MAE {v['pa_lgb_orig_MAE']}", f"MAE {v['pa_lgb_repro_MAE']} (max |diff| 0)",
         "none", "same", "No", "Original OOF (main); refit only inside follow-ups"),
        ("NGBoost reproduction refit vs saved OOF", f"MAE {v['pa_ngb_orig_MAE']}", f"MAE {v['pa_ngb_repro_MAE']}",
         "inner GroupKFold of the target encoder (CPU-dependent tie order)", "approximate", "No",
         "Original OOF for 4.1-4.4; refit only inside 4.4.1"),
        ("Original OOF vs 80% refit (point accuracy)", f"NGBoost {v['pa_ngb_orig_MAE']}, LightGBM {v['pa_lgb_orig_MAE']}",
         f"NGBoost {v['pa_ngb_80_MAE']}, LightGBM {v['pa_lgb_80_MAE']}", "refits use 80% of each training fold",
         "different training data", "No", "Report both; never pool or rank across the two configurations"),
        ("NGBoost raw 90% interval: original vs refit", f"coverage {v['ni_normal_raw_cov90']}, width {v['ni_normal_raw_width90']}",
         f"coverage {v['cs_ngb_raw_cov90']}, width {v['cs_ngb_raw_width90']}", "different fitted models", "different training data",
         "No", "4.4 uses original; 4.4.1 uses refit (labelled)"),
        ("NGBoost P(Y<60) Brier skill score: original vs refit", v["ngbprob_brier_skill_score"], v["pr_refit_raw_brier_skill_score"],
         "different fitted models", "different training data", "No", "As labelled in 4.2 / 4.4.1"),
        ("LightGBM-L2 recall at an equal flag budget: main (oracle budget) vs follow-up OP1",
         f"{v['lightgbm_recall_at_budget']} (budget = observed test positives)", f"{v['ra_lgb_OP1_recall']} (budget from calibration prevalence)",
         "different budget definition and different model fit", "different", "No",
         "OP1 is the deployable definition; the main-analysis value is descriptive"),
        ("Zero-inclusive LightGBM MAE (all rows / non-zero rows)", f"IR study (full training folds): {lz['MAE_all']:.3f} / {lz['MAE_nonzero']:.3f}",
         f"AIESC refit 80%: {v['rb_lgb_MAE_all']} / {v['rbn_lgb_MAE']}", "AIESC uses 80% of each training fold",
         "consistent direction", "No", "AIESC values for 4.2.2 (risk metrics exist only there)"),
        ("Main analysis environment: conda 'zhe' (sklearn 1.8.0) vs pinned venv (1.7.2)", "first run", "rerun 2026-11-09",
         "identical (no model is trained in the main analysis)", "same", "No", "Pinned-venv outputs (current files)"),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Original / recorded value (version)", "Latest value (version)",
                                       "Difference reason", "Same training conditions?", "Needs recomputation",
                                       "Value suitable for the paper"])


# ---- tables and figures -----------------------------------------------------------------------------------------
EXP_OF_SOURCE = {
    "table1_dataset": ("E0", "Data audit", "counts", "-"),
    "table2_overall": ("E1", SPLIT + "; original OOF", "MAE, RMSE, mean signed error, SD ratio", "95% student-cluster bootstrap CIs (5,000)"),
    "table3_risk": ("E2", SPLIT + "; original OOF", "precision, recall, F1, balanced accuracy, PR-AUC, ROC-AUC at grade < 60", "95% CIs"),
    "table8_risk_methods": ("F2-A", SPLIT + "; refit 80%", "PR-AUC, ROC-AUC, recall / precision / F1 / flag rate at native, OP1, OP2", "95% CIs; Holm contrasts in Table S3"),
    "table9_zero_grade_sensitivity": ("F2-B", SPLIT_B + "; refit 80%", "PR-AUC, ROC-AUC, recall at OP1, zero rows flagged", "95% CIs"),
    "table4_bias": ("E5", SPLIT + "; original OOF", "MAE, mean signed error by actual grade range", "descriptive"),
    "table5_uncertainty": ("E6", SPLIT + "; original OOF", "NLL, CRPS, coverage, width, P(Y<60) calibration", "descriptive"),
    "table6_calibration_split_intervals": ("F1-I", SPLIT + "; refit 80%", "90% coverage overall / actual 0-59, width", "95% CIs"),
    "table7_probability_recalibration": ("F1-P", SPLIT + "; refit 80%", "Brier, BSS, ECE, PR-AUC, ROC-AUC", "95% CI of Brier difference"),
    "table1b_folds": ("E0", "Saved T3 folds", "records, students, prevalence per fold", "-"),
    "table_s1_paired_comparisons": ("E4", SPLIT + "; original OOF", "differences vs LightGBM (36)", "Holm and Bonferroni"),
    "table_s7a_risk_contrasts": ("F2-C", SPLIT + "; refit 80%", "differences vs tuned LightGBM-L2 and threshold contrasts", "Holm within families of 20 and 8"),
    "table_s7b_thresholds": ("F2-A", "calibration students", "OP1 / OP2 thresholds per fold", "-"),
    "table_s7c_regression_accuracy": ("F2-A", SPLIT + "; refit 80%", "MAE, RMSE of regression variants", "-"),
    "table_s7d_probabilities": ("F2-A/B", SPLIT + "; refit 80%", "Brier, BSS, ECE of recalibrated probabilities", "-"),
    "table_s_calibration_split_all_levels": ("F1-I", SPLIT + "; refit 80%", "coverage and width at 50/80/90/95%", "-"),
    "table_s_calibration_split_by_actual_range": ("F1-I", SPLIT + "; refit 80%", "90% coverage by actual grade range", "-"),
    "table_s_calibration_split_by_predicted_range": ("F1-I", SPLIT + "; refit 80%", "90% coverage by predicted range", "-"),
    "table_s_calibration_split_by_category": ("F1-I", SPLIT + "; refit 80%", "90% coverage by course category", "-"),
    "table_s_point_accuracy": ("F1-R", SPLIT, "MAE, RMSE, risk metrics of original / reproduced / 80% models", "-"),
    "table_s_ngboost_scores": ("F1-R", SPLIT, "NLL, CRPS original vs refit", "-"),
    "table_s_reproduction_gate": ("F1-R", SPLIT, "reproduction differences and partition sizes", "-"),
    "figure1_workflow": ("All", "-", "-", "-"),
    "figure2_mae_rmse": ("E1", SPLIT + "; original OOF", "MAE, RMSE", "95% CIs"),
    "figure3_risk_identification": ("E2", SPLIT + "; original OOF", "precision, recall, F1; PR curves", "95% CIs"),
    "figure7_risk_methods": ("F2-A/B", SPLIT + "; refit 80%", "recall vs flag budget; recall by rule; zero rows flagged", "-"),
    "figure4_grade_range_bias": ("E5", SPLIT + "; original OOF", "mean signed error, mean prediction by range", "-"),
    "figure5_ngboost_calibration": ("E6", SPLIT + "; original OOF", "coverage error, width, coverage by range, reliability", "-"),
    "figure6_calibration_split": ("F1-I/P", SPLIT + "; refit 80%", "coverage by range, width vs coverage, reliability", "-"),
    "figureS1_ngboost_pit": ("E6", SPLIT + "; original OOF", "PIT histogram", "-"),
    "figureS2_coverage_by_course_category": ("E6", SPLIT + "; original OOF", "90% coverage by course category", "-"),
}
SCRIPT_OF_DIR = {"tables": "research/aiesc2027/run_all.py -> tables.py", "figures": "research/aiesc2027/run_all.py -> figures.py",
                 "calibration_split": "research/aiesc2027/run_calibration.py", "risk_methods": "research/aiesc2027/run_risk.py"}
SUITABILITY = {"Table 1": "Methods (main)", "Figure 1": "Methods (main); may be replaced by text if space is short",
               "Figure 2": "Optional (values are in Table 2)", "Table 5": "Main or supplementary (space)",
               "Table 8": "Main (90% level may suffice)"}


def catalogue_and_copy(dst: Path) -> pd.DataFrame:
    rows = []
    (dst / "tables").mkdir(parents=True)
    (dst / "figures").mkdir(parents=True)
    for kind, new, old, place, section, src, short in P.ITEMS:
        name = P.stem(kind, new, short)
        sub = "tables" if kind == "Table" else "figures"
        files = sorted((P.PAPER / sub).glob(f"{name}.*")) + sorted((P.PAPER / sub).glob(f"{name}_full_precision.csv"))
        for f in files:
            shutil.copyfile(f, dst / sub / f.name)
        if kind == "Table":
            title = (P.PAPER / "tables" / f"{name}.md").read_text(encoding="utf-8").splitlines()[0].strip("*")
        else:
            title = f"Figure {new}. " + P.FIGURE_CAPTIONS[new]
        exp, setting, metrics, stats = EXP_OF_SOURCE[src.name]
        top = src.relative_to(OUT).parts[0]
        label = f"{kind} {new}"
        rows.append({"Paper Label": label, "Previous Label": f"{kind} {old}", "Section": section, "Title / Caption": title,
                     "Experiment ID": exp, "Evaluation Setting": setting, "Metrics": metrics, "Statistics": stats,
                     "Data Source (CSV)": f"{sub}/{name}.csv",
                     "Original Output": str(src.relative_to(ROOT)), "Generating Script": SCRIPT_OF_DIR[top],
                     "Formats": ", ".join(sorted({f.suffix.lstrip('.') if not f.name.endswith('_full_precision.csv') else 'full-precision csv' for f in files})),
                     "Recommended Placement": SUITABILITY.get(label, "Main text" if place == "main" else "Supplementary"),
                     "Superseded Versions": "Figure 1 redrawn 2026-11-09 to show both branches (previous version: main analysis only, overwritten)"
                     if label == "Figure 1" else "none"})
    return pd.DataFrame(rows)


# ---- row-level and input manifests --------------------------------------------------------------------------------
def npz_structure(p: Path) -> str:
    z = np.load(p, allow_pickle=False)
    return "; ".join(f"{k}:{z[k].dtype.str}{list(z[k].shape)}" for k in z.files[:12]) + (" ..." if len(z.files) > 12 else "")


def row_level_manifest() -> pd.DataFrame:
    rows = []
    for p in sorted((OUT / "calibration_split/jobs").glob("*.npz")) + sorted((OUT / "risk_methods/jobs").glob("*.npz")):
        rows.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p),
                     "content": npz_structure(p), "identifiers": "none (row positions into the private analysis frame)",
                     "packaged": "No", "reason": "row-level predictions linkable to private records through row positions",
                     "regenerate": "calibration_fit.py --fold k / risk_fit.py --population A|B --fold k (pinned venv, sequential)"})
    src = [C.EXPLORATORY_OOF] + [C.AMENDMENT_JOBS / f"{m}__T3__f{f}.npz" for m in ("M2_ft_transformer", "M3_ngboost") for f in range(5)]
    for p in src:
        rows.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p),
                     "content": npz_structure(p), "identifiers": "none (row positions)", "packaged": "No",
                     "reason": "original IR out-of-fold predictions (protected, read-only, row-level)",
                     "regenerate": "IR study (outputs/research/...); not regenerated by AIESC"})
    return pd.DataFrame(rows)


def input_manifest() -> pd.DataFrame:
    items = [(C.RAW_CSV, "raw registrar export (personal data)"), (C.CLEANED_CSV, "cleaned enrolment table (personal data)"),
             (C.AMENDMENT_FIT_STATUS, "Amendment 1 fit status"), (C.FINAL_COMPARISON_STUDENT, "IR registry (MAE / RMSE)"),
             (C.FINAL_NGBOOST_UNCERTAINTY, "IR NGBoost uncertainty table"), (C.FINAL_MANIFEST, "IR reproducibility manifest"),
             (C.FAMILY_INTERVALS, "IR family-split intervals (context only)"),
             (ROOT / "outputs/research/final/zero_grade_sensitivity.csv", "IR zero-grade sensitivity (MAE only)")]
    rows = []
    for p, role in items:
        rows.append({"path": str(p.relative_to(ROOT)), "role": role, "bytes": p.stat().st_size, "sha256": sha256(p),
                     "expected_sha256": C.EXPECTED_SHA256.get(p.name, ""),
                     "packaged": "No (personal data)" if p.name in ("IR.csv", "cleaned_grades.csv") else "No (IR artefact; path and hash only)"})
    ir_code = ["research/config.py", "research/data.py", "research/models.py", "research/exploratory/candidates.py",
               "research/exploratory/encoders.py", "research/exploratory/nn.py", "research/exploratory/amendment1/worker.py",
               "research/exploratory/amendment1/amendment.json", "research/advanced/common.py", "research/advanced/course_repr.py",
               "src/ir_pipeline/privacy.py"]
    for rel in ir_code:
        p = ROOT / rel
        rows.append({"path": rel, "role": "IR code used by the AIESC refits / checks", "bytes": p.stat().st_size,
                     "sha256": sha256(p), "expected_sha256": "", "packaged": "No (path and hash only)"})
    return pd.DataFrame(rows)


# ---- documents ---------------------------------------------------------------------------------------------------
def fill(template: str, values: dict[str, str]) -> tuple[str, set[str]]:
    used = set()

    def sub(mt):
        k = mt.group(1)
        if k not in values:
            raise KeyError(f"template placeholder without value: {k}")
        used.add(k)
        return values[k]
    out = re.sub(r"\{\{(\w+)\}\}", sub, template)
    return out, used


def doc_blocks(reg, master, inventory, recompute_df, leak_df, text_df, claims_df, consistency, catalogue, tests) -> dict[str, str]:
    v = reg.text()
    ov = tab("metrics/overall.csv").set_index("key")
    rk = tab("metrics/risk_threshold60.csv").set_index("key")
    main = pd.DataFrame([{"Model": H.MODEL_LABEL[m], "MAE": r3(ov.at[m, "MAE"]), "RMSE": r3(ov.at[m, "RMSE"]),
                          "Precision": r3(rk.at[m, "precision"]), "Recall": r3(rk.at[m, "recall"]), "F1": r3(rk.at[m, "f1"]),
                          "PR-AUC": r3(rk.at[m, "pr_auc"]), "ROC-AUC": r3(rk.at[m, "roc_auc"]),
                          "Flag rate": r3(rk.at[m, "flag_rate"])} for m in H.MODEL_KEYS])
    am = tab("risk_methods/metrics/A_metrics.csv")
    risk = []
    for m in H.RISK_METHODS:
        d = am[am["method"] == m].set_index("operating_point")
        risk.append({"Method": d["label"].iat[0], "PR-AUC": r3(d.at["native", "pr_auc"]), "ROC-AUC": r3(d.at["native", "roc_auc"]),
                     "Recall native / OP1 / OP2": " / ".join(r3(d.at[o, "recall"]) for o in ("native", "OP1", "OP2")),
                     "Flag rate native / OP1 / OP2": " / ".join(r3(d.at[o, "flag_rate"]) for o in ("native", "OP1", "OP2")),
                     "Precision OP2": r3(d.at["OP2", "precision"])})
    iv = tab("calibration_split/stats/bootstrap_ci.csv")
    lab = tab("calibration_split/metrics/intervals.csv").drop_duplicates("method").set_index("method")["method_label"]
    intervals = []
    for m in H.INTERVAL_METHODS:
        g = iv[iv["method"] == m].set_index("statistic")
        intervals.append({"Method": lab[m], "Coverage 90% (95% CI)": f"{r3(g.at['coverage90_all', 'estimate'])} {ci(g.at['coverage90_all', 'ci_low'], g.at['coverage90_all', 'ci_high'])}",
                          "Coverage, actual 0-59": f"{r3(g.at['coverage90_actual_0_59', 'estimate'])} {ci(g.at['coverage90_actual_0_59', 'ci_low'], g.at['coverage90_actual_0_59', 'ci_high'])}",
                          "Mean width": r3(g.at["mean_width90_all", "estimate"])})
    pv = tab("stats/paired_vs_lightgbm.csv")
    paired = pd.DataFrame([{"Model": H.MODEL_LABEL[p["model"]], "Metric": p["metric"], "Difference": s3(p["difference"]),
                            "95% CI": sci(p["ci_low_95"], p["ci_high_95"]), "Bonferroni CI": sci(p["ci_low_bonf"], p["ci_high_bonf"]),
                            "Holm p": pval(p["p_holm"]), "Holm": "yes" if p["supported_holm_0.05"] else "no",
                            "Practical": "yes" if p["practically_important"] else "no", "Direction": p["direction"]}
                           for _, p in pv.iterrows()])
    ac = tab("risk_methods/stats/A_contrasts.csv")
    contr = pd.DataFrame([{"Family": c["family"], "Contrast": c["contrast"], "Difference": s3(c["difference"]),
                           "95% CI": sci(c["ci_low_95"], c["ci_high_95"]), "Holm p": pval(c["p_holm"]),
                           "Holm": "yes" if c["supported_holm_0.05"] else "no", "|diff| >= 0.05": "yes" if c["practically_important"] else "no",
                           "Improves": "yes" if c["improves"] else "no"} for _, c in ac.iterrows()])
    def listing(df, fmt):
        return "\n".join(f"- {fmt(x)}" for _, x in df.iterrows()) or "- (none)"

    def pv_line(p):
        return (f"{H.MODEL_LABEL[p['model']]} vs LightGBM, {p['metric']}: {s3(p['difference'])} "
                f"(95% CI {sci(p['ci_low_95'], p['ci_high_95'])}, Holm p {pval(p['p_holm'])}; {p['direction']} than LightGBM)")

    def ac_line(c):
        side = "favourable" if c["difference"] > 0 else "unfavourable"
        return (f"{c['contrast']}: {s3(c['difference'])} (95% CI {sci(c['ci_low_95'], c['ci_high_95'])}, "
                f"Holm p {pval(c['p_holm'])}; {side})")
    sup, prac = pv["supported_holm_0.05"].astype(bool), pv["practically_important"].astype(bool)
    asup, aimp = ac["supported_holm_0.05"].astype(bool), ac["improves"].astype(bool)
    lists = {
        "L_paired_not_supported": listing(pv[~sup], pv_line),
        "L_paired_supported_small": listing(pv[sup & ~prac], pv_line),
        "L_paired_supported_large": listing(pv[sup & prac], pv_line),
        "L_contrasts_improves": listing(ac[aimp], ac_line),
        "L_contrasts_not_supported": listing(ac[~asup], ac_line),
        "L_contrasts_supported_no_improve": listing(ac[asup & ~aimp], ac_line),
    }
    n_sup_main = int(pv["supported_holm_0.05"].sum())
    n_prac_main = int((pv["supported_holm_0.05"] & pv["practically_important"]).sum())
    blocks = {
        "T_main_models": md_table(main), "T_risk_methods": md_table(pd.DataFrame(risk)),
        "T_intervals": md_table(pd.DataFrame(intervals)), "T_paired": md_table(paired), "T_contrasts": md_table(contr),
        "T_recompute": md_table(recompute_df.assign(recorded_value=recompute_df["recorded_value"].map(r3),
                                                    recomputed_value=recompute_df["recomputed_value"].map(r3),
                                                    abs_difference=recompute_df["abs_difference"].map(lambda x: f"{x:.1e}"))
                                [["quantity", "recorded_value", "recomputed_value", "abs_difference", "status"]]),
        "T_leak": md_table(leak_df), "T_claims": md_table(claims_df[["claim", "evidence", "holds"]]),
        "T_consistency": md_table(consistency), "T_inventory": md_table(inventory[["Experiment ID", "Experiment", "Status", "In Integrated Results"]]),
        "T_catalogue": md_table(catalogue[["Paper Label", "Section", "Experiment ID", "Recommended Placement", "Data Source (CSV)"]]),
        "n_master_rows": f"{len(master):,d}", "n_experiments_done": str(int(inventory["Status"].str.startswith("Completed").sum())),
        "n_inventory": str(len(inventory)), "n_recompute": str(len(recompute_df)),
        "n_recompute_match": str(int((recompute_df["status"] == "MATCH").sum())), "n_leak": str(len(leak_df)),
        "n_leak_pass": str(int((leak_df["status"] == "PASS").sum())), "n_text_checks": str(len(text_df)),
        "n_text_found": str(int(text_df["found_in_results_text"].sum())), "n_claims": str(len(claims_df)),
        "n_claims_hold": str(int(claims_df["holds"].sum())), "n_paired": str(len(pv)), "n_paired_holm": str(n_sup_main),
        "n_paired_practical": str(n_prac_main), "n_contrasts": str(len(ac)),
        "n_contrasts_holm": str(int(ac["supported_holm_0.05"].sum())), "n_contrasts_improves": str(int(ac["improves"].sum())),
        "n_tables_main": str(int(((catalogue["Paper Label"].str.startswith("Table ")) & ~catalogue["Paper Label"].str.contains("S")).sum())),
        "n_tables_supp": str(int(catalogue["Paper Label"].str.startswith("Table S").sum())),
        "n_figures_main": str(int(((catalogue["Paper Label"].str.startswith("Figure ")) & ~catalogue["Paper Label"].str.contains("S")).sum())),
        "n_figures_supp": str(int(catalogue["Paper Label"].str.startswith("Figure S").sum())),
        "tests_summary": tests["summary"], "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "zip_name": ZIP.name,
        **lists,
    }
    clash = set(blocks) & set(v)
    if clash:
        raise KeyError(f"document block names collide with value keys: {sorted(clash)}")
    blocks.update(v)
    return blocks


# ---- tests, privacy, integrity ---------------------------------------------------------------------------------------
def run_tests() -> dict:
    cmd = [PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "research/aiesc2027/tests"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={"PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": "/tmp/mpl_aiesc",
                                                                         "PATH": "/usr/bin:/bin"})
    last = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1] if p.stdout.strip() else "no output"
    return {"command": " ".join(cmd), "returncode": p.returncode, "summary": last, "stdout": p.stdout, "stderr": p.stderr}


def extra_privacy_checks(root: Path) -> list[dict]:
    bad_cols = {"student_id", "instructor_id", "student_name", "name", "姓名", "學號", "teacher", "instructor"}
    out = []
    for f in sorted(root.rglob("*.csv")):
        try:
            d = pd.read_csv(f, nrows=5, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            out.append({"file": str(f.relative_to(root)), "issue": f"unreadable: {exc}"})
            continue
        cols = {str(c).strip().lower() for c in d.columns}
        if cols & bad_cols:
            out.append({"file": str(f.relative_to(root)), "issue": f"identifier-like column {sorted(cols & bad_cols)}"})
        n = sum(1 for _ in open(f, encoding="utf-8-sig")) - 1
        if n > 5000:
            out.append({"file": str(f.relative_to(root)), "issue": f"{n} rows (possible row-level data)"})
    for f in root.rglob("*.npz"):
        if "bootstrap_replicates" not in f.name:
            out.append({"file": str(f.relative_to(root)), "issue": "npz other than aggregated bootstrap replicates"})
    return out


# ---- packaging ----------------------------------------------------------------------------------------------------
def copy_experiment_outputs(dst: Path):
    skip_dirs = {"paper", "handoff_to_chatgpt", "__pycache__"}
    for f in sorted(OUT.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(OUT)
        if rel.parts[0] in skip_dirs or f == ZIP or f.name.startswith(ZIP.stem):
            continue
        if f.suffix == ".npz" and "jobs" in rel.parts:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)


def copy_code(dst: Path):
    for f in sorted(CODE_DIR.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix in (".py", ".json", ".md"):
            target = dst / f.relative_to(CODE_DIR.parents[1])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)


def pip_freeze() -> str:
    p = subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True)
    return p.stdout


def write_manifest(root: Path) -> pd.DataFrame:
    rows = [{"path": str(f.relative_to(root)), "bytes": f.stat().st_size, "sha256": sha256(f)}
            for f in sorted(root.rglob("*")) if f.is_file() and f.name != "MANIFEST_SHA256.csv"]
    df = pd.DataFrame(rows)
    df.to_csv(root / "MANIFEST_SHA256.csv", index=False)
    return df


def make_zip(root: Path, manifest: pd.DataFrame):
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in list(manifest["path"]) + ["MANIFEST_SHA256.csv"]:
            z.write(root / rel, f"{root.name}/{rel}")


def verify_zip(manifest: pd.DataFrame) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    res = {"zip": str(ZIP.relative_to(ROOT)), "bytes": ZIP.stat().st_size, "sha256": sha256(ZIP)}
    with zipfile.ZipFile(ZIP) as z:
        res["testzip_first_bad_member"] = z.testzip()
        names = {n.split("/", 1)[1] for n in z.namelist()}
        expected = set(manifest["path"]) | {"MANIFEST_SHA256.csv"}
        res["n_members"] = len(names)
        res["missing_members"] = sorted(expected - names)
        res["unexpected_members"] = sorted(names - expected)
        mism, png_bad, pdf_bad, csv_bad = [], [], [], []
        man = manifest.set_index("path")["sha256"]
        prefix = HANDOFF.name + "/"
        for rel in man.index:
            data = z.read(prefix + rel)
            if hashlib.sha256(data).hexdigest() != man[rel]:
                mism.append(rel)
            if rel.endswith(".png"):
                try:
                    img = mpimg.imread(io.BytesIO(data), format="png")
                    if img.size == 0:
                        png_bad.append(rel)
                except Exception:  # noqa: BLE001
                    png_bad.append(rel)
            elif rel.endswith(".pdf"):
                if not (data.startswith(b"%PDF") and b"%%EOF" in data[-2048:]):
                    pdf_bad.append(rel)
            elif rel.endswith(".csv"):
                try:
                    pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
                except Exception:  # noqa: BLE001
                    csv_bad.append(rel)
        res.update({"sha256_mismatches": mism, "png_unreadable": png_bad, "pdf_invalid": pdf_bad, "csv_unreadable": csv_bad,
                    "n_png": sum(r.endswith(".png") for r in man.index), "n_pdf": sum(r.endswith(".pdf") for r in man.index),
                    "n_csv": sum(r.endswith(".csv") for r in man.index)})
    res["status"] = "PASS" if (res["testzip_first_bad_member"] is None and not res["missing_members"] and not res["unexpected_members"]
                               and not mism and not png_bad and not pdf_bad and not csv_bad) else "FAIL"
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("aiesc2027").setLevel(logging.WARNING)
    t0 = time.time()
    before = integrity.snapshot()
    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    for f in (ZIP, ZIP_VERIFY, ZIP.with_suffix(".zip.sha256")):
        if f.exists():
            f.unlink()
    HANDOFF.mkdir(parents=True)
    rep, exo = HANDOFF / "reproducibility", HANDOFF / "experiment_outputs"
    rep.mkdir()

    reg = H.build()
    log.info("value registry: %d values", len(reg.values))
    recompute_df, leak_df = K.recompute(reg)
    results_text = (P.PAPER / "results_en_full.md").read_text(encoding="utf-8")
    text_df = K.text_checks(results_text, reg)
    claims_df = K.per_fold_claims()
    consistency = consistency_table(reg)
    master = master_results()
    inventory = experiment_inventory()
    catalogue = catalogue_and_copy(HANDOFF)
    tests = run_tests()
    log.info("unit tests: %s", tests["summary"])

    (HANDOFF / "paper_reference").mkdir()
    for name in ("results_en_full.md", "tables_and_figures.md", "numbering_map.csv"):
        shutil.copyfile(P.PAPER / name, HANDOFF / "paper_reference" / name)
    copy_experiment_outputs(exo)
    copy_code(rep / "code")
    master.to_csv(HANDOFF / "MASTER_RESULTS.csv", index=False)
    metric_definitions().to_csv(HANDOFF / "METRIC_DEFINITIONS.csv", index=False)
    inventory.to_csv(HANDOFF / "EXPERIMENT_INVENTORY.csv", index=False)
    catalogue.to_csv(HANDOFF / "TABLE_FIGURE_CATALOGUE.csv", index=False)
    recompute_df.to_csv(rep / "recomputation_check.csv", index=False)
    leak_df.to_csv(rep / "leakage_and_partition_checks.csv", index=False)
    text_df.to_csv(rep / "results_text_numeric_checks.csv", index=False)
    claims_df.to_csv(rep / "results_text_claim_checks.csv", index=False)
    consistency.to_csv(rep / "version_consistency.csv", index=False)
    row_level_manifest().to_csv(rep / "row_level_files_not_packaged.csv", index=False)
    input_manifest().to_csv(rep / "input_and_ir_source_manifest.csv", index=False)
    (rep / "unit_tests.txt").write_text(f"$ {tests['command']}\n\n{tests['stdout']}\n{tests['stderr']}", encoding="utf-8")
    (rep / "pip_freeze_aiesc2027_venv.txt").write_text(pip_freeze(), encoding="utf-8")

    values = doc_blocks(reg, master, inventory, recompute_df, leak_df, text_df, claims_df, consistency, catalogue, tests)
    used_keys: set[str] = set()
    for tpl in sorted(TEMPLATES.glob("*.md")):
        text, used = fill(tpl.read_text(encoding="utf-8"), values)
        used_keys |= used
        (HANDOFF / tpl.name).write_text(text, encoding="utf-8")
    trace = pd.DataFrame([{"Key": k, "Value (as quoted)": x.text(), "Full Precision": repr(x.value), "Description": x.description,
                           "Results Section": x.section, "Source File": f"outputs/aiesc2027/{x.source}",
                           "Handoff Copy": f"experiment_outputs/{x.source}" if " - " not in x.source else "experiment_outputs/risk_methods/metrics/",
                           "Row Selector": x.selector, "Column": x.column,
                           "Quoted In Handoff Documents": k in used_keys,
                           "Phrase Checked In Results Text": any(k in s.split() for s in text_df["value_keys"])}
                          for k, x in reg.values.items()])
    trace.to_csv(HANDOFF / "DATA_TRACEABILITY.csv", index=False)

    problems = []
    if tests["returncode"] != 0:
        problems.append(f"unit tests failed: {tests['summary']}")
    if (recompute_df["status"] != "MATCH").any():
        problems.append("recomputation mismatch")
    if (leak_df["status"] != "PASS").any():
        problems.append("leakage / partition check failed")
    if not text_df["found_in_results_text"].all():
        problems.append("Results-text numeric phrase not found (see results_text_numeric_checks.csv)")
    if not claims_df["holds"].all():
        problems.append("Results-text claim not supported (see results_text_claim_checks.csv)")
    if (master == "").any().any():
        problems.append("empty cell in MASTER_RESULTS (must be NA)")

    from ir_pipeline.privacy import privacy_gate, raw_identifiers
    ids, names = raw_identifiers([pd.read_csv(C.RAW_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")])
    gate = privacy_gate([HANDOFF], ids, names)
    extra = extra_privacy_checks(HANDOFF)
    (rep / "privacy_scan_handoff.json").write_text(json.dumps({"privacy_gate": gate, "additional_checks": extra}, ensure_ascii=False,
                                                              indent=2), encoding="utf-8")
    if gate["status"] != "PASS" or extra:
        problems.append(f"privacy: gate {gate['status']}, additional findings {len(extra)}")
    cmp_ = integrity.compare(before, integrity.snapshot())
    (rep / "integrity_check_handoff.json").write_text(json.dumps(cmp_, indent=2), encoding="utf-8")
    if cmp_["status"] != "PASS":
        problems.append(f"protected files changed: {cmp_}")
    status = {"status": "PASS" if not problems else "FAIL", "problems": problems, "generated_at": values["generated_at"],
              "seconds": round(time.time() - t0, 1), "unit_tests": tests["summary"], "privacy_gate": gate["status"],
              "privacy_gate_findings": gate["n_findings"], "privacy_additional_findings": len(extra),
              "integrity": cmp_["status"], "integrity_files_checked": cmp_["files_checked"],
              "recomputation": f"{values['n_recompute_match']}/{values['n_recompute']} MATCH",
              "leakage_checks": f"{values['n_leak_pass']}/{values['n_leak']} PASS",
              "results_text_numeric_phrases": f"{values['n_text_found']}/{values['n_text_checks']} found",
              "master_rows": len(master)}
    (rep / "handoff_build_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if problems:
        log.error("hand-off checks failed: %s", problems)
        return 1

    manifest = write_manifest(HANDOFF)
    make_zip(HANDOFF, manifest)
    ver = verify_zip(manifest)
    after = integrity.compare(before, integrity.snapshot())
    ver["integrity_after_zip"] = after["status"]
    ver["n_manifest_files"] = len(manifest)
    ZIP_VERIFY.write_text(json.dumps(ver, indent=2), encoding="utf-8")
    ZIP.with_suffix(".zip.sha256").write_text(f"{ver['sha256']}  {ZIP.name}\n", encoding="utf-8")
    log.info("zip %s: %.1f MB, %d members, verification %s", ZIP.name, ver["bytes"] / 1e6, ver["n_members"], ver["status"])
    return 0 if ver["status"] == "PASS" and after["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
