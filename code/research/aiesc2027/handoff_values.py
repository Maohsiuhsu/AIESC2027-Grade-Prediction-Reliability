"""Registry of every number quoted in the ChatGPT hand-off documents, read from the saved AIESC outputs.

Each value records its source file, row selector and column so that DATA_TRACEABILITY.csv can be produced from the
same objects that fill the document templates. Nothing here writes to disk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "aiesc2027"

MODEL_KEYS = ["lightgbm", "catboost", "ft_transformer", "ngboost", "lightgbm_l1"]
MODEL_LABEL = {"lightgbm": "LightGBM", "catboost": "CatBoost", "ft_transformer": "FT-Transformer",
               "ngboost": "NGBoost", "lightgbm_l1": "LightGBM-L1"}
RISK_METHODS = ["lgb", "l1", "classifier", "quantile_q25", "weighted", "ngboost_prob"]
INTERVAL_METHODS = ["ngb_raw", "ngb_rescaled", "split_abs", "split_norm", "mondrian_category",
                    "mondrian_predicted_range", "cqr_lightgbm", "split_abs_lightgbm"]
CONFORMAL = INTERVAL_METHODS[2:]


@dataclass
class Value:
    key: str
    value: float
    fmt: str
    source: str
    selector: str
    column: str
    description: str
    section: str = ""

    def text(self) -> str:
        v = self.value
        if self.fmt == "int":
            return f"{int(round(v)):,d}"
        if self.fmt == "f3":
            return f"{v:.3f}".replace("-0.000", "0.000")
        if self.fmt == "s3":
            s = f"{v:+.3f}"
            return "0.000" if s in ("+0.000", "-0.000") else s
        if self.fmt == "pct1":
            return f"{100 * v:.1f}%"
        if self.fmt == "p":
            return "< 0.001" if v < 0.001 else f"{v:.3f}"
        if self.fmt == "f1":
            return f"{v:.1f}"
        raise ValueError(self.fmt)


@lru_cache(maxsize=None)
def table(rel: str) -> pd.DataFrame:
    return pd.read_csv(OUT / rel, encoding="utf-8-sig")


def _select(rel: str, where: dict) -> pd.DataFrame:
    d = table(rel)
    for k, v in where.items():
        d = d[d[k].astype(str) == str(v)]
    return d


def _sel_text(where: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in where.items()) or "all rows"


class Registry:
    def __init__(self):
        self.values: dict[str, Value] = {}

    def add(self, key, value, fmt, source, selector, column, description, section=""):
        if key in self.values:
            raise KeyError(f"duplicate value key {key}")
        self.values[key] = Value(key, float(value), fmt, source, selector, column, description, section)

    def one(self, key, rel, where, col, fmt, description, section=""):
        d = _select(rel, where)
        if len(d) != 1:
            raise ValueError(f"{key}: {rel} {where} matched {len(d)} rows")
        self.add(key, d[col].iat[0], fmt, rel, _sel_text(where), col, description, section)

    def span(self, key, rel, where, col, fmt, description, section="", rows=None):
        """Adds key_min and key_max over the selected rows (optionally restricted to a column filter)."""
        d = _select(rel, where)
        if rows is not None:
            d = d[d[rows[0]].isin(rows[1])]
        if d.empty:
            raise ValueError(f"{key}: {rel} {where} matched no rows")
        sel = _sel_text(where) + (f"; {rows[0]} in {rows[1]}" if rows else "")
        self.add(f"{key}_min", d[col].min(), fmt, rel, sel, f"min({col})", f"{description} (minimum)", section)
        self.add(f"{key}_max", d[col].max(), fmt, rel, sel, f"max({col})", f"{description} (maximum)", section)

    def json_value(self, key, rel, path, fmt, description, section=""):
        obj = json.loads((OUT / rel).read_text(encoding="utf-8"))
        for p in path:
            obj = obj[p]
        self.add(key, obj, fmt, rel, "/".join(path), path[-1], description, section)

    def __getitem__(self, key) -> Value:
        return self.values[key]

    def text(self) -> dict[str, str]:
        return {k: v.text() for k, v in self.values.items()}


def build() -> Registry:
    r = Registry()
    # ---- data -----------------------------------------------------------------------------------------
    dc = "validation/data_counts.json"
    for k in ("raw_rows", "term_rows", "zero_rows", "analysis_rows", "students", "sections", "courses"):
        r.json_value(f"n_{k}", dc, [k], "int", f"data count: {k}", "Data")
    counts = json.loads((OUT / dc).read_text(encoding="utf-8"))
    r.add("n_other_terms", counts["raw_rows"] - counts["term_rows"], "int", dc, "raw_rows - term_rows", "derived",
          "enrolments in IR.csv outside the study term 113-2 (excluded)", "Data")
    ds = "metrics/dataset.csv"
    r.one("n_low", ds, {"item": "Actual grade < 60 (fixed research threshold)"}, "n", "int", "enrolments with grade < 60", "Data")
    r.one("n_low_official", ds, {"item": "Actual grade < official programme pass threshold (60 or 70)"}, "n", "int",
          "enrolments below the official programme threshold", "Data")
    r.one("n_official70", ds, {"item": "Records with official threshold 70 (master's / doctoral)"}, "n", "int",
          "enrolments whose official pass threshold is 70", "Data")
    r.one("prevalence", "metrics/risk_threshold60.csv", {"key": "lightgbm"}, "prevalence", "f3", "share of grade < 60", "Data")
    folds = table("metrics/folds.csv")
    r.add("fold_records_min", folds["records"].min(), "int", "metrics/folds.csv", "all folds", "min(records)", "test-fold size (minimum)", "Data")
    r.add("fold_records_max", folds["records"].max(), "int", "metrics/folds.csv", "all folds", "max(records)", "test-fold size (maximum)", "Data")
    r.add("fold_prev_min", folds["prevalence_below_60"].min(), "f3", "metrics/folds.csv", "all folds", "min(prevalence_below_60)", "test-fold prevalence (minimum)", "Data")
    r.add("fold_prev_max", folds["prevalence_below_60"].max(), "f3", "metrics/folds.csv", "all folds", "max(prevalence_below_60)", "test-fold prevalence (maximum)", "Data")

    # ---- 4.1 overall error (original OOF) --------------------------------------------------------------------
    ov = "metrics/overall.csv"
    for m in MODEL_KEYS:
        for col in ("MAE", "RMSE", "mean_signed_error", "sd_ratio", "median_ae"):
            r.one(f"{m}_{col}", ov, {"key": m}, col, "f3", f"{MODEL_LABEL[m]} {col}, original OOF, n = 83,494", "4.1")
    r.span("sd_ratio", ov, {}, "sd_ratio", "f3", "SD(prediction) / SD(actual) across the five models", "4.1")
    ci = "stats/per_model_bootstrap_ci.csv"
    for m in MODEL_KEYS:
        for met in ("MAE", "RMSE", "recall", "pr_auc"):
            for col in ("ci_low", "ci_high"):
                r.one(f"{m}_{met}_{col}", ci, {"model": m, "metric": met}, col, "f3",
                      f"{MODEL_LABEL[m]} {met} 95% student-cluster bootstrap CI ({col})", "4.1-4.2")
    pv = "stats/paired_vs_lightgbm.csv"
    for m in MODEL_KEYS[1:]:
        for met in ("MAE", "RMSE", "precision", "recall", "f1", "balanced_accuracy", "pr_auc", "roc_auc"):
            for col, fmt in (("difference", "s3"), ("ci_low_95", "s3"), ("ci_high_95", "s3"), ("p_holm", "p"),
                             ("ci_low_bonf", "s3"), ("ci_high_bonf", "s3"), ("relative_difference", "pct1")):
                r.one(f"d_{m}_{met}_{col}", pv, {"model": m, "metric": met}, col, fmt,
                      f"{MODEL_LABEL[m]} minus LightGBM, {met}: {col}", "4.1-4.2")
    r.add("holm_floor", table(pv)["p_holm"].min(), "p", pv, "all 36 contrasts", "min(p_holm)",
          "smallest attainable Holm-adjusted p (5,000 replicates, 36 contrasts)", "4.1")

    # ---- 4.2 risk identification (original OOF) -------------------------------------------------------------
    rk = "metrics/risk_threshold60.csv"
    for m in MODEL_KEYS:
        for col in ("precision", "recall", "f1", "balanced_accuracy", "pr_auc", "roc_auc", "flag_rate", "recall_at_budget",
                    "FN", "TP"):
            r.one(f"{m}_{col}", rk, {"key": m}, col, "int" if col in ("FN", "TP") else "f3",
                  f"{MODEL_LABEL[m]} {col}, rule predicted grade < 60 (ranking metrics: score = -prediction)", "4.2")
    r.span("flag_rate", rk, {}, "flag_rate", "pct1", "share of enrolments flagged by predicted grade < 60", "4.2")
    ro = "metrics/risk_official_threshold.csv"
    r.span("official_recall", ro, {}, "recall", "f3", "recall with the official programme threshold", "4.2")
    r.span("official_pr_auc", ro, {}, "pr_auc", "f3", "PR-AUC with the official programme threshold", "4.2")
    lh = "metrics/low_high_grade_detail.csv"
    r.span("share_low_pred_ge60", lh, {}, "share_low_predicted_ge_60", "pct1",
           "share of grade < 60 enrolments predicted >= 60", "4.3")
    np_ = "metrics/ngboost_probability.csv"
    for col in ("mean_predicted_probability", "brier", "brier_reference_train_prevalence", "brier_skill_score",
                "ece_10_bins", "pr_auc", "roc_auc"):
        r.one(f"ngbprob_{col}", np_, {"threshold": 60, "version": "normal_raw"}, col, "f3",
              f"NGBoost raw P(Y < 60): {col} (original OOF)", "4.2")
    # ---- 4.3 grade-range bias ----------------------------------------------------------------------------------
    gb = "metrics/grade_range_bias.csv"
    for band, tag in (("0-59", "low"), ("90-100", "high"), ("70-79", "mid")):
        r.span(f"mse_{tag}", gb, {"grade_range": band}, "mean_signed_error", "s3",
               f"mean signed error (predicted - actual) in actual range {band}", "4.3")
    r.span("meanpred_low", gb, {"grade_range": "0-59"}, "mean_predicted", "f3", "mean prediction for actual 0-59", "4.3")
    r.one("meanactual_low", gb, {"key": "lightgbm", "grade_range": "0-59"}, "mean_actual", "f3", "mean actual grade in 0-59", "4.3")
    for m in ("lightgbm", "lightgbm_l1"):
        for band, tag in (("0-59", "low"), ("90-100", "high")):
            r.one(f"{m}_mse_{tag}", gb, {"key": m, "grade_range": band}, "mean_signed_error", "s3",
                  f"{MODEL_LABEL[m]} mean signed error in {band}", "4.3")

    # ---- 4.4 NGBoost uncertainty (original OOF) ---------------------------------------------------------------
    ni = "metrics/ngboost_intervals.csv"
    for meth in ("normal_raw", "normal_rescaled", "cf_absolute", "cf_normalised", "cf_mondrian"):
        for lvl in (0.5, 0.8, 0.9, 0.95):
            r.one(f"ni_{meth}_cov{int(lvl * 100)}", ni, {"method": meth, "nominal": lvl, "group_by": "all"}, "coverage",
                  "f3", f"{meth} empirical coverage at nominal {lvl} (original NGBoost OOF)", "4.4")
        r.one(f"ni_{meth}_width90", ni, {"method": meth, "nominal": 0.9, "group_by": "all"}, "mean_width", "f3",
              f"{meth} mean width at nominal 0.9", "4.4")
        r.one(f"ni_{meth}_cov90_low", ni, {"method": meth, "nominal": 0.9, "group_by": "grade_range", "group": "0-59"},
              "coverage", "f3", f"{meth} coverage at 0.9 for actual grade 0-59", "4.4")
        r.one(f"ni_{meth}_cov90_high", ni, {"method": meth, "nominal": 0.9, "group_by": "grade_range", "group": "90-100"},
              "coverage", "f3", f"{meth} coverage at 0.9 for actual grade 90-100", "4.4")
    r.span("ni_raw_cat_cov90", ni, {"method": "normal_raw", "nominal": 0.9, "group_by": "course_category"}, "coverage",
           "f3", "raw NGBoost 90% coverage across course categories", "4.4")
    r.span("ni_cfm_cat_cov90", ni, {"method": "cf_mondrian", "nominal": 0.9, "group_by": "course_category"}, "coverage",
           "f3", "cross-fold Mondrian 90% coverage across course categories", "4.4")
    ns = "metrics/ngboost_scores.csv"
    r.one("ngb_nll", ns, {"version": "normal_raw"}, "NLL", "f3", "NGBoost NLL (original OOF)", "4.4")
    r.one("ngb_crps", ns, {"version": "normal_raw"}, "CRPS", "f3", "NGBoost CRPS (original OOF)", "4.4")

    # ---- 4.4.1 calibration split (refitted, 80% of each training fold) -----------------------------------------
    cb = "calibration_split/stats/bootstrap_ci.csv"
    for meth in INTERVAL_METHODS:
        for stat, tag in (("coverage90_all", "cov90"), ("coverage90_actual_0_59", "cov90_low"), ("mean_width90_all", "width90")):
            for col, suf in (("estimate", ""), ("ci_low", "_lo"), ("ci_high", "_hi")):
                r.one(f"cs_{meth}_{tag}{suf}", cb, {"method": meth, "statistic": stat}, col, "f3",
                      f"calibration split, {meth}: {stat} ({col})", "4.4.1")
    for meth in ("refit_isotonic - refit_raw", "refit_platt - refit_raw"):
        tag = meth.split()[0]
        for col, suf in (("estimate", ""), ("ci_low", "_lo"), ("ci_high", "_hi")):
            r.one(f"cs_brierdiff_{tag}{suf}", cb, {"method": meth, "statistic": "brier_difference"}, col, "s3",
                  f"Brier difference {meth} ({col})", "4.4.1")
    ci_ = "calibration_split/metrics/intervals.csv"
    r.span("cs_conf_cov90", ci_, {"nominal": 0.9, "group_by": "all"}, "coverage", "f3",
           "90% coverage of the six split-conformal methods", "4.4.1", rows=("method", CONFORMAL))
    r.span("cs_conf_width90", ci_, {"nominal": 0.9, "group_by": "all"}, "mean_width", "f3",
           "90% mean width of the six split-conformal methods", "4.4.1", rows=("method", CONFORMAL))
    r.span("cs_conf_cov90_low", ci_, {"nominal": 0.9, "group_by": "grade_range", "group": "0-59"}, "coverage", "f3",
           "90% coverage for actual 0-59, six split-conformal methods", "4.4.1", rows=("method", CONFORMAL))
    r.one("cs_cqr_cov90_high", ci_, {"method": "cqr_lightgbm", "nominal": 0.9, "group_by": "grade_range", "group": "90-100"},
          "coverage", "f3", "CQR 90% coverage for actual 90-100", "4.4.1")
    r.span("cs_conf_below_low", ci_, {"nominal": 0.9, "group_by": "grade_range", "group": "0-59"},
           "share_actual_below_interval", "f3", "share of actual 0-59 rows below the 90% interval (conformal methods)",
           "4.4.1", rows=("method", CONFORMAL))
    for meth in ("split_norm", "mondrian_category"):
        r.span(f"cs_{meth}_cat_cov90", ci_, {"method": meth, "nominal": 0.9, "group_by": "course_category"}, "coverage",
               "f3", f"{meth} 90% coverage across course categories", "4.4.1")
    pa = "calibration_split/metrics/point_accuracy.csv"
    for lab, tag in (("NGBoost, original (full training fold)", "ngb_orig"), ("NGBoost, reproduction refit (full training fold)", "ngb_repro"),
                     ("NGBoost, proper-training refit (80%)", "ngb_80"), ("LightGBM, original (full training fold)", "lgb_orig"),
                     ("LightGBM, reproduction refit (full training fold)", "lgb_repro"), ("LightGBM, proper-training refit (80%)", "lgb_80")):
        for col in ("MAE", "recall"):
            r.one(f"pa_{tag}_{col}", pa, {"model": lab}, col, "f3", f"{lab}: {col}", "4.4.1")
    pr = "calibration_split/metrics/probability_recalibration.csv"
    for ver in ("original_full_fold", "refit_raw", "refit_isotonic", "refit_platt"):
        for col in ("mean_predicted_probability", "brier", "brier_skill_score", "ece_10_bins", "pr_auc", "roc_auc"):
            r.one(f"pr_{ver}_{col}", pr, {"version": ver}, col, "f3", f"P(Y < 60) {ver}: {col}", "4.4.1")
    rg = "calibration_split/metrics/reproduction_gate.csv"
    r.span("rg_lgb_maxdiff", rg, {}, "lightgbm_max_abs_diff", "f3", "LightGBM reproduction max |difference| per fold", "4.4.1")
    r.span("rg_cal_students", rg, {}, "partition_cal_students", "int", "calibration students per fold", "4.4.1")

    # ---- 4.2.1 / 4.2.2 risk methods ----------------------------------------------------------------------------
    am = "risk_methods/metrics/A_metrics.csv"
    for m in RISK_METHODS:
        for op in ("native", "OP1", "OP2"):
            for col in ("precision", "recall", "f1", "flag_rate"):
                r.one(f"ra_{m}_{op}_{col}", am, {"method": m, "operating_point": op}, col, "f3",
                      f"population A, {m}, {op}: {col}", "4.2.1")
        for col in ("pr_auc", "roc_auc"):
            r.one(f"ra_{m}_{col}", am, {"method": m, "operating_point": "native"}, col, "f3", f"population A, {m}: {col}", "4.2.1")
    for m in ("lgb", "weighted"):
        r.one(f"ra_{m}_MAE", am, {"method": m, "operating_point": "native"}, "MAE", "f3", f"population A, {m}: MAE", "4.2.1")
    r.span("ra_native_flag", am, {"operating_point": "native"}, "flag_rate", "f3", "native-rule flag rate across methods", "4.2.1")
    r.span("ra_op1_recall", am, {"operating_point": "OP1"}, "recall", "f3", "OP1 recall across methods", "4.2.1")
    r.span("ra_op2_flag", am, {"operating_point": "OP2"}, "flag_rate", "f3", "OP2 flag rate across methods", "4.2.1")
    r.span("ra_op2_precision", am, {"operating_point": "OP2"}, "precision", "f3", "OP2 precision across methods", "4.2.1")
    r.span("ra_pr_auc", am, {"operating_point": "native"}, "pr_auc", "f3", "PR-AUC across methods (population A)", "4.2.1")
    ac = "risk_methods/stats/A_contrasts.csv"
    for _, row in table(ac).iterrows():
        tag = (row["contrast"].replace(" vs lgb (tuned): ", "_vs_lgb_").replace(": OP", "_OP").replace(" vs native rule: ", "_vs_native_")
               .replace(" ", "_"))
        for col, fmt in (("difference", "s3"), ("ci_low_95", "s3"), ("ci_high_95", "s3"), ("p_holm", "p")):
            r.one(f"c_{tag}_{col}", ac, {"contrast": row["contrast"]}, col, fmt, f"{row['contrast']}: {col}", "4.2.1")
    ap = "risk_methods/metrics/A_probabilities.csv"
    for m in ("classifier", "ngboost_prob"):
        for col in ("mean_probability", "brier_skill_score", "ece_10_bins"):
            r.one(f"rp_{m}_{col}", ap, {"method": m}, col, "f3", f"population A, isotonic {m}: {col}", "4.2.1")
    at = "risk_methods/metrics/A_thresholds.csv"
    for m in ("lgb", "l1"):
        for col in ("grade_cutoff_OP1", "grade_cutoff_OP2"):
            r.span(f"rt_{m}_{col}", at, {"method": m}, col, "f1", f"{m} {col} across folds", "4.2.1")
    bm = "risk_methods/metrics/B_metrics_all_rows.csv"
    r.one("rb_prevalence", bm, {"method": "lgb", "operating_point": "native"}, "prevalence", "f3", "population B prevalence", "4.2.2")
    r.span("rb_pr_auc", bm, {"operating_point": "native"}, "pr_auc", "f3", "population B PR-AUC across methods", "4.2.2")
    for m in RISK_METHODS:
        r.one(f"rb_{m}_pr_auc", bm, {"method": m, "operating_point": "native"}, "pr_auc", "f3", f"population B, {m}: PR-AUC", "4.2.2")
        r.one(f"rbn_{m}_pr_auc", "risk_methods/metrics/B_metrics_nonzero_rows.csv", {"method": m, "operating_point": "native"},
              "pr_auc", "f3", f"population B models evaluated on the 83,494 non-zero rows, {m}: PR-AUC", "4.2.2")
    r.span("rb_zero_flag_op1", "risk_methods/metrics/B_zero_rows_flagged.csv", {"operating_point": "OP1"}, "share_flagged",
           "f3", "share of the 2,073 zero-grade rows flagged at OP1", "4.2.2")
    r.one("rb_lgb_MAE_all", bm, {"method": "lgb", "operating_point": "native"}, "MAE", "f3", "population B LightGBM-L2 MAE (all rows)", "4.2.2")
    r.one("rbn_lgb_MAE", "risk_methods/metrics/B_metrics_nonzero_rows.csv", {"method": "lgb", "operating_point": "native"}, "MAE",
          "f3", "population B LightGBM-L2 MAE on non-zero rows", "4.2.2")
    a = table(am).query("operating_point == 'native'").set_index("method")["pr_auc"]
    bn = table("risk_methods/metrics/B_metrics_nonzero_rows.csv").query("operating_point == 'native'").set_index("method")["pr_auc"]
    diff = (bn - a).reindex(RISK_METHODS)
    r.add("rb_nonzero_prdiff_min", diff.min(), "s3", "risk_methods/metrics/B_metrics_nonzero_rows.csv - A_metrics.csv",
          "operating_point=native", "min(pr_auc_B_nonzero - pr_auc_A)", "PR-AUC change on non-zero rows from training with zeros (min)", "4.2.2")
    r.add("rb_nonzero_prdiff_max", diff.max(), "s3", "risk_methods/metrics/B_metrics_nonzero_rows.csv - A_metrics.csv",
          "operating_point=native", "max(pr_auc_B_nonzero - pr_auc_A)", "PR-AUC change on non-zero rows from training with zeros (max)", "4.2.2")
    return r
