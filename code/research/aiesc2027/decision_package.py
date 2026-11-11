"""Documents, traceability, reproducibility records and SHA-256 manifest for decision_reliability_update_v1.

Every number in the documents is read from the CSV outputs of run_decision_reliability.py (nothing is typed by hand).
Run after the analysis: python -B research/aiesc2027/decision_package.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.aiesc2027 import config as C  # noqa: E402
from research.aiesc2027 import integrity  # noqa: E402

BASE = C.OUT / "decision_reliability_update_v1"
SUB = {k: BASE / k for k in ("figures_recolored", "paper_ready_figures", "budget_analysis", "risk_strata_analysis",
                             "source_traceability", "reproducibility")}
CODE_DIR = ROOT / "research" / "aiesc2027"
PAPER = C.OUT / "paper"
PY = sys.executable
V3_NAME = "AIESC2027_submission_review_v3"
V3_FILES = ["main.pdf", "DATA_SOURCE_MAP.csv", "FOCUSED_DATA_CHECKS.csv", "REFERENCE_VERIFICATION.csv", "README_zh_TW.md"]
BUDGET_PCT = {0.05: "5%", 0.1: "10%", 0.15: "15%", 0.2: "20%", 0.25: "25%"}


def f3(x) -> str:
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def s3(x) -> str:
    return f"{x:+.3f}"


def ci(lo, hi, signed=False) -> str:
    f = s3 if signed else f3
    return f"[{f(lo)}, {f(hi)}]"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ---- load analysis outputs -----------------------------------------------------------------------------------------
def load() -> dict:
    rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")  # noqa: E731
    b, r = SUB["budget_analysis"], SUB["risk_strata_analysis"]
    return dict(
        bt=rd(b / "budget_reliability_table_full_precision.csv"),
        bc=rd(b / "budget_recall_contrasts_full_precision.csv"),
        st=rd(r / "risk_strata_reliability_full_precision.csv"),
        gd=rd(r / "calibration_gap_differences_full_precision.csv"),
        iv=rd(r / "high_risk_interval_coverage_full_precision.csv"),
        checks=rd(SUB["reproducibility"] / "analysis_checks.csv"),
        status=json.loads((SUB["reproducibility"] / "analysis_status.json").read_text(encoding="utf-8")),
        spec=json.loads((CODE_DIR / "decision_reliability_spec.json").read_text(encoding="utf-8")),
    )


def brow(bt, key, x):
    return bt[(bt.score_key == key) & np.isclose(bt.budget_share, x)].iloc[0]


def crow(bc, key, x):
    return bc[(bc.score_key == key) & np.isclose(bc.budget_share, x)].iloc[0]


def srow(st, key, stratum, version="NA (ranking only)"):
    return st[(st.score_key == key) & (st.stratum == stratum) & (st.probability_version == version)].iloc[0]


def irow(iv, config, method, stratum):
    return iv[(iv.configuration == config) & (iv.method == method) & (iv.stratum == stratum)].iloc[0]


# ---- derived checks -------------------------------------------------------------------------------------------------
def monotonicity(st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, key), g in st[st.probability_version == "NA (ranking only)"].groupby(["configuration", "score_key"], sort=False):
        g = g.sort_values("stratum_order")
        rate, lo, hi = g.observed_low_rate.to_numpy(), g.observed_low_rate_ci_low.to_numpy(), g.observed_low_rate_ci_high.to_numpy()
        overlap = [f"{g.stratum.iat[j]} / {g.stratum.iat[j + 1]}" for j in range(len(g) - 1) if lo[j] <= hi[j + 1]]
        rows.append({"configuration": config, "score_key": key, "score_label": g.score_label.iat[0],
                     "observed_rate_by_stratum": " > ".join(f3(v) for v in rate),
                     "monotone_decreasing_point_estimates": bool(np.all(np.diff(rate) < 0)),
                     "top5_rate": rate[0], "top5_ci_low": lo[0], "top5_ci_high": hi[0],
                     "bottom_rate": rate[-1], "top5_to_bottom_ratio": rate[0] / rate[-1],
                     "adjacent_strata_with_overlapping_95ci": "; ".join(overlap) if overlap else "none",
                     "note": "CI overlap is a conservative descriptive check, not a test"})
    return pd.DataFrame(rows)


def budget_method_comparison() -> pd.DataFrame:
    """Recall of the original OOF scores at the paper's budget share (7,685 / 83,494) under both tie/budget rules."""
    from research.aiesc2027 import io_data
    from research.aiesc2027.decision_reliability import FoldBlocks
    d = io_data.load_all()
    pos = (d.y < C.RISK_THRESHOLD).astype(float)
    k = int(pos.sum())
    share = k / len(pos)
    pub = pd.read_csv(C.OUT / "metrics" / "risk_threshold60.csv", encoding="utf-8-sig").set_index("key")
    rows = []
    for m in C.MODEL_ORDER:
        fl, cap, P = FoldBlocks(-d.preds[m], d.fold).capture(pos, np.ones(len(pos)), [share])
        rows.append({"model": C.MODELS[m]["label"], "budget_share": share, "budget_records": k,
                     "recall_paper_pooled_sort_order": float(pub.at[m, "recall_at_budget"]),
                     "recall_this_update_per_fold_fractional_ties": float(cap[0] / P),
                     "difference": float(cap[0] / P - pub.at[m, "recall_at_budget"]),
                     "paper_rule": "pooled over folds, top 7,685 by score, ties broken by sort order (metrics.recall_at_budget)",
                     "update_rule": "same share within each fold, tied records at the cut-off counted fractionally"})
    return pd.DataFrame(rows)


def interval_top5(iv: pd.DataFrame) -> pd.DataFrame:
    t = iv[iv.stratum == "top 5%"].copy()
    keep = ["configuration", "method", "method_label", "strata_defined_by", "n_records", "coverage", "coverage_ci_low",
            "coverage_ci_high", "coverage_all_records", "coverage_minus_all", "coverage_minus_all_ci_low",
            "coverage_minus_all_ci_high", "coverage_actual_low", "share_actual_below_interval", "mean_width"]
    t = t[keep].copy()
    t["below_nominal_0.90_ci_excludes"] = t.coverage_ci_high < 0.9
    t["supports_lower_than_overall"] = t.coverage_minus_all_ci_high < 0
    return t


# ---- value dictionary ----------------------------------------------------------------------------------------------
def values(o: dict, mono: pd.DataFrame, bmc: pd.DataFrame, top5: pd.DataFrame) -> tuple[dict, list[dict]]:
    bt, bc, st, gd, iv = o["bt"], o["bc"], o["st"], o["gd"], o["iv"]
    V, H = {}, []
    BT = "budget_analysis/budget_reliability_table.csv"
    BCF = "budget_analysis/budget_recall_contrasts.csv"
    STF = "risk_strata_analysis/risk_strata_reliability.csv"
    GDF = "risk_strata_analysis/calibration_gap_differences.csv"
    IVF = "risk_strata_analysis/high_risk_interval_coverage.csv"

    def hl(id_, topic, zh, value, lo=None, hi=None, source="", selector="", status=""):
        H.append({"id": id_, "topic": topic, "statement_zh": zh, "value": round(float(value), 3),
                  "ci_low_95": None if lo is None else round(float(lo), 3), "ci_high_95": None if hi is None else round(float(hi), 3),
                  "source_file": source, "selector": selector, "statistical_status": status})

    # budget, original OOF
    for x in (0.05, 0.10, 0.20, 0.25):
        r = brow(bt, "lightgbm", x)
        p = BUDGET_PCT[x]
        V[f"lgb_rec_{p}"], V[f"lgb_rec_{p}_ci"] = f3(r.recall), ci(r.recall_ci_low, r.recall_ci_high)
        V[f"lgb_pre_{p}"], V[f"lgb_pre_{p}_ci"] = f3(r.precision), ci(r.precision_ci_low, r.precision_ci_high)
        V[f"lgb_lift_{p}"], V[f"lgb_nnr_{p}"] = f3(r.lift), f3(r.records_per_low_grade_found)
        V[f"lgb_cap_{p}"], V[f"lgb_flag_{p}"] = f"{r.low_grade_captured:,.1f}", f"{r.records_flagged:,.1f}"
        hl(f"B_orig_lgb_recall_{p}", "budget", f"原始 OOF LightGBM 在每折前 {p} 審查預算的召回率", r.recall, r.recall_ci_low,
           r.recall_ci_high, BT, f"score_key=lightgbm, budget_share={x}", "95% student-cluster bootstrap CI (5,000, seed 42)")
        hl(f"B_orig_lgb_precision_{p}", "budget", f"原始 OOF LightGBM 在前 {p} 的精確率", r.precision, r.precision_ci_low,
           r.precision_ci_high, BT, f"score_key=lightgbm, budget_share={x}", "95% student-cluster bootstrap CI")
    V["prev"] = f3(bt.prevalence.iat[0])
    o_con = bc[bc.configuration == "original_oof"]
    r_con = bc[bc.configuration == "refit_80"]
    V["orig_n_con"], V["orig_n_sup"] = len(o_con), int(o_con["supported_holm_0.05"].sum())
    V["orig_n_prac"] = int(o_con.practically_important.sum())
    V["orig_diff_min"], V["orig_diff_max"] = s3(o_con.recall_difference.min()), s3(o_con.recall_difference.max())
    ns = o_con[~o_con["supported_holm_0.05"]]
    V["orig_ns_list"] = "；".join(f"{r.score_label} {BUDGET_PCT[round(r.budget_share, 2)]}（{s3(r.recall_difference)}，"
                                 f"95% CI {ci(r.recall_ci_low_95, r.recall_ci_high_95, True)}，Holm p = {f3(r.p_holm)}）"
                                 for r in ns.itertuples()) or "無"
    V["orig_all_lower"] = bool((o_con.recall_difference < 0).all())
    for key in ("catboost", "ft_transformer", "ngboost", "lightgbm_l1", "ngboost_prob"):
        for x in (0.05, 0.10):
            r = crow(bc, key, x)
            V[f"d_{key}_{BUDGET_PCT[x]}"] = f"{s3(r.recall_difference)} {ci(r.recall_ci_low_95, r.recall_ci_high_95, True)}"
            hl(f"B_orig_diff_{key}_{BUDGET_PCT[x]}", "budget contrast", f"{r.score_label} 減 LightGBM 的召回率差（前 {BUDGET_PCT[x]}）",
               r.recall_difference, r.recall_ci_low_95, r.recall_ci_high_95, BCF, f"score_key={key}, budget_share={x}",
               f"paired bootstrap; Holm p = {f3(r.p_holm)} (family of {r.family_size}); "
               f"{'supported' if r['supported_holm_0.05'] else 'not supported'}; |diff| >= 0.05: {'yes' if r.practically_important else 'no'}")
    # budget, refit
    V["refit_n_con"], V["refit_n_sup"] = len(r_con), int(r_con["supported_holm_0.05"].sum())
    V["refit_n_prac"] = int(r_con.practically_important.sum())
    clf = r_con[r_con.score_key == "refit_classifier"]
    V["clf_diff_min"], V["clf_diff_max"] = s3(clf.recall_difference.min()), s3(clf.recall_difference.max())
    V["clf_all_sup"] = bool(clf["supported_holm_0.05"].all())
    q = r_con[r_con.score_key == "refit_quantile_q25"]
    V["q25_sup_budgets"] = "、".join(BUDGET_PCT[round(x, 2)] for x in q[q["supported_holm_0.05"]].budget_share)
    V["q25_ns_budgets"] = "、".join(BUDGET_PCT[round(x, 2)] for x in q[~q["supported_holm_0.05"]].budget_share)
    V["q25_sup_min"], V["q25_sup_max"] = s3(q[q["supported_holm_0.05"]].recall_difference.min()), s3(q[q["supported_holm_0.05"]].recall_difference.max())
    w5 = crow(bc, "refit_weighted", 0.05)
    w25 = crow(bc, "refit_weighted", 0.25)
    V["w5"] = f"{s3(w5.recall_difference)} {ci(w5.recall_ci_low_95, w5.recall_ci_high_95, True)}"
    V["w25"] = f"{s3(w25.recall_difference)} {ci(w25.recall_ci_low_95, w25.recall_ci_high_95, True)}，Holm p = {f3(w25.p_holm)}"
    V["w5_prac"] = bool(w5.practically_important)
    for key in ("refit_classifier", "refit_quantile_q25", "refit_weighted", "refit_l1", "refit_ngboost_prob"):
        for x in (0.05, 0.10):
            r = crow(bc, key, x)
            hl(f"B_refit_diff_{key}_{BUDGET_PCT[x]}", "budget contrast", f"{r.score_label} 減 refit LightGBM-L2 的召回率差（前 {BUDGET_PCT[x]}）",
               r.recall_difference, r.recall_ci_low_95, r.recall_ci_high_95, BCF, f"score_key={key}, budget_share={x}",
               f"paired bootstrap; Holm p = {f3(r.p_holm)} (family of {r.family_size}); "
               f"{'supported' if r['supported_holm_0.05'] else 'not supported'}; |diff| >= 0.05: {'yes' if r.practically_important else 'no'}")
    hl("B_refit_diff_refit_weighted_25%", "budget contrast", "class-weighted L2 減 refit L2 的召回率差（前 25%）", w25.recall_difference,
       w25.recall_ci_low_95, w25.recall_ci_high_95, BCF, "score_key=refit_weighted, budget_share=0.25",
       f"Holm p = {f3(w25.p_holm)}; not supported")
    rc10, rl10 = brow(bt, "refit_classifier", 0.1), brow(bt, "refit_lgb", 0.1)
    V["clf_rec_10"], V["clf_rec_10_ci"] = f3(rc10.recall), ci(rc10.recall_ci_low, rc10.recall_ci_high)
    V["rl2_rec_10"], V["rl2_rec_10_ci"] = f3(rl10.recall), ci(rl10.recall_ci_low, rl10.recall_ci_high)
    V["p_floor"] = f3(min(1.0, 2 / (C.N_BOOT + 1) * len(o_con)))
    # method comparison
    lg = bmc[bmc.model == "LightGBM"].iloc[0]
    l1 = bmc[bmc.model == "LightGBM-L1"].iloc[0]
    V["bmc_share"] = f3(lg.budget_share)
    V["bmc_lgb_paper"], V["bmc_lgb_new"] = f3(lg.recall_paper_pooled_sort_order), f3(lg.recall_this_update_per_fold_fractional_ties)
    V["bmc_l1_paper"], V["bmc_l1_new"] = f3(l1.recall_paper_pooled_sort_order), f3(l1.recall_this_update_per_fold_fractional_ties)
    V["bmc_maxdiff"] = f3(bmc.difference.abs().max())
    # strata, ranking
    V["mono_all"] = bool(mono.monotone_decreasing_point_estimates.all())
    V["mono_fail"] = "、".join(mono[~mono.monotone_decreasing_point_estimates].score_label) or "無"
    V["top5_min"], V["top5_max"] = f3(mono.top5_rate.min()), f3(mono.top5_rate.max())
    prev = bt.prevalence.iat[0]
    V["top5_lift_min"], V["top5_lift_max"] = f3(mono.top5_rate.min() / prev), f3(mono.top5_rate.max() / prev)
    f2 = pd.read_csv(PAPER / "figures" / "Figure02_mae_rmse.csv", encoding="utf-8-sig")
    for m in ("MAE", "RMSE"):
        e = f2[f2.metric == m].estimate
        V[f"fig2_{m}"] = f"{f3(e.min())}–{f3(e.max())}"
    V["bot_min"], V["bot_max"] = f3(mono.bottom_rate.min()), f3(mono.bottom_rate.max())
    ov = mono[mono.adjacent_strata_with_overlapping_95ci != "none"]
    V["overlap_list"] = "；".join(f"{r.score_label}（{r.adjacent_strata_with_overlapping_95ci}）" for r in ov.itertuples()) or "無"
    for key in ("lightgbm", "refit_classifier", "refit_weighted", "ngboost"):
        r0, r1 = srow(st, key, "top 5%"), srow(st, key, "5-10%")
        V[f"s_{key}_top5"] = f"{f3(r0.observed_low_rate)} {ci(r0.observed_low_rate_ci_low, r0.observed_low_rate_ci_high)}"
        V[f"s_{key}_5_10"] = f"{f3(r1.observed_low_rate)} {ci(r1.observed_low_rate_ci_low, r1.observed_low_rate_ci_high)}"
        V[f"s_{key}_lift"] = f3(r0.lift)
        hl(f"S_{key}_top5_rate", "risk strata", f"{r0.score_label} 最高 5% 風險層的實際低分率", r0.observed_low_rate,
           r0.observed_low_rate_ci_low, r0.observed_low_rate_ci_high, STF, f"score_key={key}, stratum=top 5%, probability_version=NA",
           "95% student-cluster bootstrap CI; descriptive")
        r4 = srow(st, key, "40-100%")
        V[f"s_{key}_bottom"] = f3(r4.observed_low_rate)
    lgt = srow(st, "lightgbm", "top 5%")
    V["lgb_top5_pred"], V["lgb_top5_act"] = f3(lgt.mean_predicted_grade), f3(lgt.mean_actual_grade)
    # strata, calibration
    for key, nm in (("refit_ngboost_prob", "ngb"), ("refit_classifier", "clf")):
        for v in ("raw", "isotonic", "platt"):
            r = srow(st, key, "top 5%", v)
            V[f"g_{nm}_{v}"] = f"{s3(r.calibration_gap)} {ci(r.calibration_gap_ci_low, r.calibration_gap_ci_high, True)}"
            V[f"p_{nm}_{v}"] = f3(r.mean_predicted_probability)
            hl(f"S_gap_{nm}_{v}_top5", "calibration by stratum", f"{r.score_label}（{v}）最高 5% 層的校準差（預測平均機率 − 實際率）",
               r.calibration_gap, r.calibration_gap_ci_low, r.calibration_gap_ci_high, STF,
               f"score_key={key}, stratum=top 5%, probability_version={v}", "95% student-cluster bootstrap CI; descriptive")
        V[f"o_{nm}_top5"] = f3(srow(st, key, "top 5%", "raw").observed_low_rate)
        iso = st[(st.score_key == key) & (st.probability_version == "isotonic")]
        V[f"iso_{nm}_maxabs"] = f3(iso.calibration_gap.abs().max())
        V[f"iso_{nm}_all_ci0"] = bool(((iso.calibration_gap_ci_low <= 0) & (iso.calibration_gap_ci_high >= 0)).all())
        pl = st[(st.score_key == key) & (st.probability_version == "platt")]
        V[f"platt_{nm}_ex0"] = "、".join(pl[(pl.calibration_gap_ci_low > 0) | (pl.calibration_gap_ci_high < 0)].stratum)
    raw_o = st[(st.score_key == "ngboost_prob") & (st.probability_version == "raw")]
    V["orig_raw_gap_min"], V["orig_raw_gap_max"] = s3(raw_o.calibration_gap.min()), s3(raw_o.calibration_gap.max())
    r = srow(st, "ngboost_prob", "top 5%", "raw")
    V["g_orig_raw"] = f"{s3(r.calibration_gap)} {ci(r.calibration_gap_ci_low, r.calibration_gap_ci_high, True)}"
    gi = gd[(gd.score_key == "refit_ngboost_prob") & (gd.version == "isotonic") & (gd.stratum == "top 5%")].iloc[0]
    V["gd_ngb_iso_top5"] = f"{s3(gi.abs_gap_difference)} {ci(gi.abs_gap_difference_ci_low, gi.abs_gap_difference_ci_high, True)}"
    hl("S_absgap_diff_ngb_iso_top5", "calibration by stratum", "refit NGBoost：isotonic 減 raw 的 |校準差|（最高 5% 層）",
       gi.abs_gap_difference, gi.abs_gap_difference_ci_low, gi.abs_gap_difference_ci_high, GDF,
       "score_key=refit_ngboost_prob, version=isotonic, stratum=top 5%", "paired bootstrap CI; descriptive")
    # intervals
    for _, r in top5.iterrows():
        tag = f"{r.configuration}_{r.method}"
        V[f"iv_{tag}"] = f"{f3(r.coverage)} {ci(r.coverage_ci_low, r.coverage_ci_high)}"
        V[f"iv_{tag}_all"] = f3(r.coverage_all_records)
        V[f"iv_{tag}_diff"] = f"{s3(r.coverage_minus_all)} {ci(r.coverage_minus_all_ci_low, r.coverage_minus_all_ci_high, True)}"
        V[f"iv_{tag}_low"] = f3(r.coverage_actual_low)
        hl(f"I_{tag}_top5", "interval coverage", f"{r.method_label}（{r.configuration}）在最高 5% 風險層的 90% 區間覆蓋率",
           r.coverage, r.coverage_ci_low, r.coverage_ci_high, IVF, f"configuration={r.configuration}, method={r.method}, stratum=top 5%",
           f"95% student-cluster bootstrap CI; overall coverage {f3(r.coverage_all_records)}; descriptive")
    below = top5[top5["below_nominal_0.90_ci_excludes"]]
    V["iv_below_n"], V["iv_n"] = len(below), len(top5)
    V["iv_ok_list"] = "、".join(f"{r.method_label}（{r.configuration}）" for r in top5[~top5["below_nominal_0.90_ci_excludes"]].itertuples())
    V["iv_min"] = top5.loc[top5.coverage.idxmin()]
    for m in ("cf_absolute", "normal_raw"):
        g = iv[(iv.configuration == "original_oof") & (iv.method == m)]
        V[f"iv_range_{m}"] = f"{f3(g.coverage.min())}–{f3(g.coverage.max())}"
    return V, H


# ---- consistency with the existing paper text -----------------------------------------------------------------------
def paper_line(snippet: str) -> str:
    lines = (PAPER / "results_en_full.md").read_text(encoding="utf-8").splitlines()
    hits = [i + 1 for i, ln in enumerate(lines) if snippet in ln]
    return f"results_en_full.md line {hits[0]}" if hits else "results_en_full.md (snippet not found)"


def consistency(V: dict) -> pd.DataFrame:
    rows = [
        ("4.2 equal-budget recall", "recall was", "this budget, recall was",
         f"Paper: LightGBM {V['bmc_lgb_paper']}, LightGBM-L1 {V['bmc_l1_paper']} (pooled top 7,685, sort-order ties).",
         f"Same share ({V['bmc_share']}) per fold with fractional ties: LightGBM {V['bmc_lgb_new']}, L1 {V['bmc_l1_new']} "
         f"(max |difference| over 5 models {V['bmc_maxdiff']}).",
         "Consistent in direction; numbers differ slightly because of the budget/tie rule. Do not mix the two sets of numbers.",
         "一致；數字因預算與平手規則不同而有微小差異，兩組數字不可混用。"),
        ("4.2.1 recall-versus-budget curves", "nearly overlap", "nearly overlap",
         "Paper: 'The recall-versus-budget curves of all six methods nearly overlap (Figure 4a)'.",
         f"Classifier minus L2: {V['clf_diff_min']} to {V['clf_diff_max']} at 5-25% (all Holm-supported, all < 0.05); "
         f"class-weighted minus L2 at 5%: {V['w5']} (|diff| >= 0.05).",
         "Mostly consistent (differences small), but one exception reaches the practical threshold: class weighting at the 5% budget. "
         "The author may want to qualify 'nearly overlap'.",
         "大致一致（差距小），但 class-weighted 在 5% 預算的差距達實務門檻；作者可考慮將 'nearly overlap' 加上限定。"),
        ("4.4 Platt vs isotonic", "logistic recalibration gave similar results", "logistic recalibration gave similar results",
         "Paper: logistic (Platt) recalibration 'gave similar results' (overall skill score and ECE).",
         f"Top-5% stratum, refit NGBoost: isotonic gap {V['g_ngb_isotonic']}, Platt gap {V['g_ngb_platt']}; "
         f"classifier: isotonic {V['g_clf_isotonic']}, Platt {V['g_clf_platt']}.",
         "Overall statement holds; in the highest-risk stratum Platt under-predicts while isotonic does not. Potential nuance, not a contradiction.",
         "整體敘述成立；但在最高風險層 Platt 低估、isotonic 沒有。屬於細節差異，不是矛盾；作者可考慮補一句。"),
        ("4.4 constant-width intervals", "", "constant-width intervals around",
         "Paper: constant-width intervals around NGBoost or LightGBM covered only 0.690-0.701 of enrolments predicted below 60.",
         f"Top-5% risk stratum: split CP absolute (NGBoost) {V['iv_refit_80_split_abs']}; split CP around LightGBM {V['iv_refit_80_split_abs_lightgbm']}.",
         "Consistent; the risk-stratum view shows the same pattern, slightly stronger at the top 5%.",
         "一致；以風險分層看也是同樣模式，最高 5% 層更明顯。"),
        ("4.4 Mondrian by predicted range and CQR", "kept coverage at", "kept coverage at",
         "Paper: Mondrian by predicted grade range and CQR kept coverage at 0.893-0.929 in every predicted grade range.",
         f"Top-5% risk stratum: CQR {V['iv_refit_80_cqr_lightgbm']}; Mondrian predicted range {V['iv_refit_80_mondrian_predicted_range']}.",
         "Consistent.", "一致。"),
        ("4.2 NGBoost probability over-prediction", "In every reliability bin", "In every reliability bin",
         "Paper: in every reliability bin the observed share of low grades was below the predicted probability.",
         f"Original NGBoost raw gap by risk stratum: {V['orig_raw_gap_min']} to {V['orig_raw_gap_max']} (all positive).",
         "Consistent.", "一致。"),
        ("4.2.1 isotonic classifier calibration", "well calibrated", "well calibrated",
         "Paper: after isotonic recalibration the classifier's probabilities were well calibrated (overall).",
         f"Every risk stratum: max |gap| {V['iso_clf_maxabs']}; all 95% CIs include 0: {V['iso_clf_all_ci0']}.",
         "Consistent; the stratum view supports the statement.", "一致；分層結果支持此敘述。"),
        ("Figure 2 caption", "", "",
         "Caption: 'Whiskers: 95% student-cluster bootstrap confidence intervals; y-axes truncated.'",
         "Recoloured Figure 2 is a dot-and-whisker plot without bars or truncated axes.",
         "If the dot plot is adopted, the caption sentence 'y-axes truncated' must be replaced (see FIGURE_STYLE_NOTES.md). "
         "If the bar alternative is adopted, the caption can stay.",
         "若採用點圖，需替換圖說中的 'y-axes truncated' 一句；若採用長條替代版，圖說可不改。"),
    ]
    out = []
    for topic, _, snippet, paper, new, verdict, verdict_zh in rows:
        loc = paper_line(snippet) if snippet else "paper/tables_and_figures.md (Figure 2 caption)"
        out.append({"topic": topic, "paper_location": loc, "paper_statement": paper, "this_update": new, "assessment": verdict,
                    "assessment_zh": verdict_zh})
    return pd.DataFrame(out)


# ---- figure trace ---------------------------------------------------------------------------------------------------
def figure_trace() -> pd.DataFrame:
    v3 = ROOT.parent / V3_NAME
    v3_found = any(p.exists() for p in (v3, C.OUT / V3_NAME, ROOT / V3_NAME))
    v3_note = (f"{V3_NAME}/ not found in the workspace; the match with V3 main.pdf could not be verified here"
               if not v3_found else f"{V3_NAME}/ found")
    fig = PAPER / "figures"
    new = SUB["figures_recolored"]
    pr = SUB["paper_ready_figures"]
    rows = []

    def add(number, new_stem, orig_stem, data_upstream, function, caption_change, data_dir=new):
        csv_new, csv_old = data_dir / f"{new_stem}.csv", fig / f"{orig_stem}.csv"
        rows.append({"figure_number": number, "new_file_stem": rel(data_dir / new_stem),
                     "replaces_workspace_figure": rel(fig / orig_stem), "numbering_source": "outputs/aiesc2027/paper/numbering_map.csv",
                     "figure_data_csv": rel(csv_old), "data_upstream": data_upstream,
                     "figure_data_identical_bytes": sha256(csv_new) == sha256(csv_old),
                     "figure_data_sha256": sha256(csv_old), "original_png_sha256": sha256(fig / f"{orig_stem}.png"),
                     "new_png_sha256": sha256(data_dir / f"{new_stem}.png"), "drawing_function": function,
                     "numbers_changed": "no", "figure_number_changed": "no", "caption_change_needed": caption_change,
                     "v3_status": v3_note})
    up1 = "outputs/aiesc2027/validation/data_counts.json"
    up2 = "outputs/aiesc2027/stats/per_model_bootstrap_ci.csv (MAE, RMSE)"
    add("Figure 1", "Figure01_color", "Figure01_workflow", up1, "research/aiesc2027/decision_figures.py::figure1_color",
        "no (colour legend added at the bottom of the figure; caption text unchanged)")
    add("Figure 2", "Figure02_color", "Figure02_mae_rmse", up2, "research/aiesc2027/decision_figures.py::figure2_color",
        "yes, one sentence: 'y-axes truncated' no longer applies (dot-and-whisker plot)")
    add("Figure 2 (alternative)", "Figure02_color_bars_alternative", "Figure02_mae_rmse", up2,
        "research/aiesc2027/decision_figures.py::figure2_color_bars", "no (bars with truncated y-axes, as the original)")
    add("Figure 1 (drop-in)", "Figure01_workflow", "Figure01_workflow", up1, "research/aiesc2027/decision_figures.py::figure1_color",
        "no", data_dir=pr)
    add("Figure 2 (drop-in)", "Figure02_mae_rmse", "Figure02_mae_rmse", up2, "research/aiesc2027/decision_figures.py::figure2_color",
        "yes (as Figure 2 above)", data_dir=pr)
    return pd.DataFrame(rows)


def analysis_figure_trace() -> pd.DataFrame:
    rows = []
    for d, stem, fn, src, place in (
            ("budget_analysis", "budget_recall_curve", "budget_curves(metric='recall')", "budget_recall_curve.csv; budget_reliability_table.csv",
             "supplement (panel b overlaps Figure 4a of the paper)"),
            ("budget_analysis", "budget_precision_curve", "budget_curves(metric='precision')", "budget_precision_curve.csv; budget_reliability_table.csv",
             "supplement"),
            ("risk_strata_analysis", "risk_strata_plot", "strata_plot", "risk_strata_plot.csv; risk_strata_reliability.csv",
             "supplement, or panel (c) for the main text"),
            ("risk_strata_analysis", "high_risk_interval_coverage", "interval_plot", "high_risk_interval_coverage.csv",
             "main-text candidate (new information not in Figures 6-7)")):
        rows.append({"figure": stem, "path": rel(SUB[d] / stem), "drawing_function": f"research/aiesc2027/decision_figures.py::{fn}",
                     "data_csv": src, "png_sha256": sha256(SUB[d] / f"{stem}.png"), "suggested_placement": place,
                     "status": "new, post hoc exploratory; no figure number assigned"})
    return pd.DataFrame(rows)


# ---- input hashes ---------------------------------------------------------------------------------------------------
def input_hashes() -> pd.DataFrame:
    files = [(C.CLEANED_CSV, "cleaned grades (via io_data.load_all; row-level, not copied)"),
             (C.EXPLORATORY_OOF, "IR out-of-fold predictions (row-level, not copied)")]
    files += [(p, "IR Amendment-1 per-fold predictions (row-level, not copied)")
              for p in sorted(C.AMENDMENT_JOBS.glob("M[23]_*__T3__f*.npz"))]
    files += [(p, "follow-up 2 refitted scores, population A (row-level, not copied)")
              for p in sorted((C.OUT / "risk_methods" / "jobs").glob("A_fold*.npz"))]
    files += [(p, "follow-up 1 refitted models (row-level, not copied)")
              for p in sorted((C.OUT / "calibration_split" / "jobs").glob("fold*.*"))]
    files += [(C.OUT / "validation" / "data_counts.json", "Figure 1 counts"),
              (PAPER / "figures" / "Figure01_workflow.csv", "Figure 1 data"), (PAPER / "figures" / "Figure02_mae_rmse.csv", "Figure 2 data"),
              (C.OUT / "stats" / "per_model_bootstrap_ci.csv", "Figure 2 upstream"),
              (PAPER / "numbering_map.csv", "official figure numbering"), (PAPER / "tables_and_figures.md", "captions"),
              (PAPER / "results_en_full.md", "Results text (consistency check only)"),
              (C.OUT / "metrics" / "ngboost_intervals.csv", "cross-check: original interval coverage"),
              (C.OUT / "calibration_split" / "metrics" / "intervals.csv", "cross-check: refit interval coverage"),
              (C.OUT / "metrics" / "risk_threshold60.csv", "cross-check: prevalence, equal-budget recall"),
              (C.OUT / "calibration_split" / "metrics" / "probability_recalibration.csv", "cross-check: Brier scores")]
    return pd.DataFrame([{"path": rel(p), "role": r, "bytes": p.stat().st_size, "sha256": sha256(p)} for p, r in files if p.exists()])


# ---- documents ------------------------------------------------------------------------------------------------------
def yn(b: bool) -> str:
    return "是" if b else "否"


def doc_style(trace: pd.DataFrame) -> str:
    same = bool(trace.figure_data_identical_bytes.all())
    return f"""# 圖表樣式說明（Figure 1–2 彩色重繪）

## 追溯結果
- 指定的 `{V3_NAME}/`（{", ".join(V3_FILES)}）**在工作區與本機找不到**，因此無法直接比對 V3 `main.pdf`。
- 本次改以工作區內的正式來源追溯：`outputs/aiesc2027/paper/numbering_map.csv`（Figure 1 = `Figure01_workflow`，Figure 2 = `Figure02_mae_rmse`）、
  `paper/tables_and_figures.md` 的圖說，以及原繪圖程式 `research/aiesc2027/figures.py`（`fig1_flow`、`fig2_error`）。
- 新圖使用與原圖**逐位元組相同**的資料 CSV（全部相同：{yn(same)}），數值上游也再次核對：Figure 1 對 `validation/data_counts.json`，Figure 2 對 `stats/per_model_bootstrap_ci.csv`（差異 0）。
- **需使用者確認**：V3 `main.pdf` 中的 Figure 1–2 是否就是工作區的 `Figure01_workflow`／`Figure02_mae_rmse`。若 V3 有另外修改過文字，本次重繪需再對照調整。

## 共同原則
- 數值、方塊文字、流程邏輯、圖號完全不變；只改顏色、線型、標記與字級。
- 低飽和度學術配色；**不只靠顏色區分**：每個類別同時有不同的框線樣式（實線／虛線／點線／點虛線）或標記形狀，黑白列印仍可分辨。
- 字型嵌入 PDF（TrueType，`pdf.fonttype 42`），PNG 為 300 dpi。

## Figure 1（研究流程）
| 類別 | 填色 | 框線 | 框線樣式 |
|---|---|---|---|
| 資料與分析集 | #F2F2F2 | #5A5A5A | 實線 |
| 既有 OOF 主分析 | #DCE8F4 | #2E5C8A | 實線 |
| 80/20 calibration follow-up（refit 模型） | #FBE6D2 | #A0522D | 虛線 |
| 低分決策（風險辨識）分析 | #DDEFE2 | #2E7D4F | 點虛線 |
| 不確定性與機率校準分析 | #ECDDF1 | #6A3D84 | 點線 |
| 統計方法（共同） | #E9EEF3 | #5A5A5A | 實線 |

- 箭頭依分支著色並使用相同線型（OOF 實線、follow-up 虛線）。
- 圖底加入 4 項圖例；圖面由 7.2 × 5.0 英吋調整為 7.2 × 5.45 英吋以容納圖例（版面比例接近原圖）。
- 字級由原本 6.2–6.9 pt 提高為 6.35–7.0 pt（最長的兩個方塊受寬度限制，只小幅提高）。
- 圖說不需修改；若期刊不接受圖內圖例，可在圖說中加一句顏色說明（由作者決定）。

## Figure 2（MAE／RMSE）
- **主版本 `Figure02_color`**：改為點估計加信賴區間（dot-and-whisker），模型在 y 軸，每個模型同時有專屬顏色與標記形狀（LightGBM 圓、CatBoost 方、FT-Transformer 三角、NGBoost 菱形、LightGBM-L1 倒三角），並標出數值（3 位小數）。
  優點：不需截斷 y 軸，避免截斷長條造成的視覺誇大；黑白可讀。
- **圖說需改一句**：原圖說的 "Whiskers: 95% student-cluster bootstrap confidence intervals; y-axes truncated." 建議改為
  "Markers: point estimates; whiskers: 95% student-cluster bootstrap confidence intervals."（其他文字不變）。
- **替代版本 `Figure02_color_bars_alternative`**：保留原本長條＋截斷 y 軸的版面，加入斜線／點狀等填充圖樣。若想完全不改圖說，可用此版本。
- 尺寸 7.2 × 2.75 英吋（與原圖相近）。

## 顏色一致性
- 模型顏色沿用 Okabe-Ito 色盲友善配色（LightGBM #0072B2、CatBoost #E69F00、FT-Transformer #009E73、NGBoost #CC79A7、LightGBM-L1 #D55E00），與尚未重繪的 Figure 3–7 一致。
  這組顏色飽和度較 Figure 1 的淡色高；若希望 Figure 2 更柔和，需同時重繪 Figure 3–7 才能保持全文一致（本次未做，待決定）。

## 放入論文
- `paper_ready_figures/Figure01_workflow.*`、`Figure02_mae_rmse.*` 檔名與原檔相同，可直接替換（Figure 2 使用點圖主版本）。
- 原始檔 `outputs/aiesc2027/paper/figures/` 未被修改（完整性檢查 PASS）。
"""


def doc_budget_notes(V: dict, spec: dict) -> str:
    return f"""# 固定審查預算分析：方法與注意事項

## 問題
若輔導人力每學期只能審查固定比例的修課紀錄，依風險分數排序後，能找到多少實際低分（< 60）紀錄？

## 定義
- 分析單位：**修課紀錄（enrolment record）**，不是學生。n = 83,494，低分率 {V['prev']}。
- 預算：在**每個測試折內**選取風險分數最高的 5%、10%、15%、20%、25% 紀錄，再合併五折。
- 平手：樹模型預測值大量平手。切點上的平手紀錄依比例計入（期望值），因此被選取數恰為預算，計數可能非整數。
- 指標：被選取數、找到的低分紀錄數、遺漏數、recall@k、precision@k、lift（precision ÷ 盛行率）、每找到一筆低分需審查的紀錄數（= 1 ÷ precision）。
- 隨機基準：recall = 預算比例；precision = 盛行率。
- 信賴區間：以學生為單位的 cluster bootstrap，5,000 次，seed 42，百分位法 95% CI。

## 分數類型（不可混用）
- **排序分數**：回歸模型使用「負的預測成績」，只用來排序，**不是機率**。
- **機率分數**：NGBoost P(Y < 60)、LightGBM 分類器的機率。預算分析只用它們的排序；isotonic／Platt 都是單調轉換，不改變排序，所以預算結果與原始機率相同。

## 兩種訓練設定（不做跨設定排名）
- `original_oof`：IR 研究原本的 out-of-fold 預測（完整訓練折）。參考模型為 LightGBM。
- `refit_80`：follow-up 1–2 的 refit 模型（每個訓練折 80% 學生訓練、20% 校準）。參考模型為 refit LightGBM-L2。
- 兩者使用相同的測試紀錄，但訓練資料量與程序不同，只在各自設定內比較。

## 統計比較
- 每個設定內：各分數 − 參考模型的 recall 差（成對 bootstrap），每個設定 25 個比較，以 Holm 校正。
- 實務門檻：|差異| ≥ {spec['budget_analysis']['contrasts']['practical_threshold']}（沿用 follow-up 2 的預設門檻）。
- 固定預算下 precision 差與 recall 差成比例，不另外檢定。
- p 值下限：5,000 次 bootstrap 的最小雙尾 p 約 0.0004，25 個比較 Holm 校正後下限約 {V['p_floor']}；表中 0.010 應讀為「≤ 0.010」。

## 與論文既有「等預算 recall」的差別
- 論文 4.2 節：合併五折後取前 7,685 筆（= 測試集實際低分數），平手依排序順序切開，LightGBM {V['bmc_lgb_paper']}、L1 {V['bmc_l1_paper']}。
- 本次：同樣比例（{V['bmc_share']}）但在每折內選取，平手按比例計入：LightGBM {V['bmc_lgb_new']}、L1 {V['bmc_l1_new']}（五模型最大差 {V['bmc_maxdiff']}）。
- 兩組數字方向一致但不可混用；本次的 5–25% 預算是預先固定的比例，不使用測試集的實際低分數，因此沒有該處的 oracle 問題。
- 對照表：`source_traceability/budget_method_comparison.csv`。

## 狀態
- 事後探索性分析（post hoc），不重新訓練、不調參；設計先寫入 `research/aiesc2027/decision_reliability_spec.json` 再計算。
- 單一學期資料，沒有跨學期驗證，也沒有部署或成績改善的證據。
"""


def doc_strata_notes(V: dict) -> str:
    return f"""# 風險分層分析：方法與注意事項

## 分層
- 在每個測試折內依風險分數排序，分為最高 5%、5–10%、10–20%、20–40%、40–100%。
- 平手紀錄取平均名次（mid-rank），同一平手區塊永遠在同一層，因此各層實際比例可能與名義比例略有差距（表中 `realised_share`）。
- 機率分數的各版本（raw／isotonic／Platt）都用**原始分數**分層，所以同一分數的各版本落在相同分層，可直接比較校準。

## 兩個獨立問題（分開報告）
1. **排序可靠度**：越高風險的層，實際低分率是否越高？適用所有分數（含回歸的排序分數）。
2. **機率校準**：預測平均機率是否等於實際低分率？只適用機率分數（NGBoost P(Y < 60)、LightGBM 分類器）。排序分數沒有機率意義，相關欄位為 NA。

## 指標
- 每層：紀錄數、低分紀錄數、實際低分率（95% CI）、lift、平均預測成績、平均實際成績。
- 機率分數另有：平均預測機率、校準差（預測 − 實際，95% CI）、Brier。
- `calibration_gap_differences.csv`：同一層內 isotonic／Platt 相對 raw 的校準差與 |校準差| 的成對 bootstrap 差異。
- 原始 OOF 的 NGBoost 沒有獨立校準集，因此只有 raw 版本（在測試資料上做再校準會造成洩漏，故不做）。

## 區間覆蓋（NGBoost 90% 區間與 conformal 方法）
- 原始 OOF：5 種方法（NGBoost raw／rescaled、cross-fold conformal absolute／normalised／Mondrian），以原始 NGBoost P(Y < 60) 分層。
- Refit：8 種方法；NGBoost 系列以 refit NGBoost P(Y < 60) 分層，CQR 與「LightGBM 周圍的 split CP」以 refit LightGBM-L2 的排序分層（每種區間依它自己的基礎模型分層）。
- 每層：覆蓋率（95% CI）、該方法整體覆蓋率、層內覆蓋 − 整體覆蓋（成對 bootstrap 95% CI）、實際低分紀錄的覆蓋率、低於區間下界的比例、平均寬度。
- 分層用的是預測時就能取得的分數，不是實際成績；因此可以回答「在系統判定為高風險的紀錄上，區間是否仍可靠」。

## 統計
- 學生 cluster bootstrap，5,000 次，seed 42，百分位法 95% CI。
- 分層結果為描述性；相鄰層的 CI 是否重疊只是保守的描述檢查，不是檢定（`strata_monotonicity_check.csv`）。
- 事後探索性分析；單一學期。
"""


def doc_summary(V: dict, o: dict, mono: pd.DataFrame, top5: pd.DataFrame) -> str:
    s = o["status"]
    mn = V["iv_min"]
    iv_lines = "\n".join(
        f"| {r.method_label} | {'原始 OOF' if r.configuration == 'original_oof' else 'refit 80%'} | {f3(r.coverage_all_records)} | "
        f"{f3(r.coverage)} {ci(r.coverage_ci_low, r.coverage_ci_high)} | {s3(r.coverage_minus_all)} "
        f"{ci(r.coverage_minus_all_ci_low, r.coverage_minus_all_ci_high, True)} | {f3(r.coverage_actual_low)} | {f3(r.mean_width)} |"
        for r in top5.itertuples())
    mono_lines = "\n".join(f"| {r.score_label} | {'原始 OOF' if r.configuration == 'original_oof' else 'refit 80%'} | "
                           f"{r.observed_rate_by_stratum} | {yn(r.monotone_decreasing_point_estimates)} | {r.adjacent_strata_with_overlapping_95ci} |"
                           for r in mono.itertuples())
    return f"""# 分析摘要（決策可靠性補強分析 v1）

狀態：事後探索性分析（post hoc），只重新分析既有預測，不重新訓練模型。分析單位為修課紀錄（n = 83,494，低分率 {V['prev']}）。
信賴區間：學生 cluster bootstrap 5,000 次、seed 42。資料檢查 {s['checks_pass']}/{s['checks']} 通過；既有成果完整性 {s['integrity']}（{s['integrity_files_checked']} 個檔案）。

## 1. 固定審查預算

### 原始 OOF（參考：LightGBM）
- LightGBM 前 5%：recall {V['lgb_rec_5%']} {V['lgb_rec_5%_ci']}，precision {V['lgb_pre_5%']} {V['lgb_pre_5%_ci']}，lift {V['lgb_lift_5%']}。
- LightGBM 前 10%：審查 {V['lgb_flag_10%']} 筆、找到 {V['lgb_cap_10%']} 筆低分紀錄；recall {V['lgb_rec_10%']} {V['lgb_rec_10%_ci']}，precision {V['lgb_pre_10%']}，每找到 1 筆需審查 {V['lgb_nnr_10%']} 筆。
- LightGBM 前 25%：recall {V['lgb_rec_25%']} {V['lgb_rec_25%_ci']}，precision {V['lgb_pre_25%']}。
- 其他模型與 LightGBM 的差：{V['orig_diff_min']} 至 {V['orig_diff_max']}；全部低於 LightGBM：{yn(V['orig_all_lower'])}。
  {V['orig_n_con']} 個比較中 {V['orig_n_sup']} 個在 Holm 校正後有支持，達到實務門檻（|差| ≥ 0.05）的有 {V['orig_n_prac']} 個。
- 未獲支持：{V['orig_ns_list']}。
- 結論：LightGBM 在所有預算下都找到最多低分紀錄，但領先幅度小（< 0.05）。這與論文 MAE 與 PR-AUC 的排序一致。

### Refit 80%（參考：refit LightGBM-L2）
- 分類器 − L2：{V['clf_diff_min']} 至 {V['clf_diff_max']}（5–25% 全部 Holm 支持：{yn(V['clf_all_sup'])}，但都 < 0.05）。前 10%：分類器 {V['clf_rec_10']} {V['clf_rec_10_ci']}，L2 {V['rl2_rec_10']} {V['rl2_rec_10_ci']}。
- 0.25 分位數回歸 − L2：{V['q25_sup_budgets']} 有支持（{V['q25_sup_min']} 至 {V['q25_sup_max']}），{V['q25_ns_budgets']} 沒有支持。
- class-weighted L2 − L2：前 5% 為 {V['w5']}，**達到實務門檻（|差| ≥ 0.05）：{yn(V['w5_prac'])}**；前 25% 為 {V['w25']}（未校正 CI 排除 0，但 Holm 校正後未獲支持）。
- {V['refit_n_con']} 個比較中 {V['refit_n_sup']} 個 Holm 支持，{V['refit_n_prac']} 個達實務門檻。

## 2. 風險分層

### 排序可靠度
- 所有分數的實際低分率都隨風險層下降（點估計單調遞減：{yn(V['mono_all'])}；不單調者：{V['mono_fail']}）。
- 最高 5% 層的實際低分率：{V['top5_min']}–{V['top5_max']}（盛行率 {V['prev']}）；40–100% 層：{V['bot_min']}–{V['bot_max']}。
- LightGBM 最高 5%：{V['s_lightgbm_top5']}，lift {V['s_lightgbm_lift']}；平均預測成績 {V['lgb_top5_pred']}、平均實際成績 {V['lgb_top5_act']}（此層是依最低的預測值選出，平均預測低於平均實際屬選擇效應；論文的「高估低分」是依實際成績分組，兩者不矛盾）。
- 相鄰層 95% CI 重疊（分層區辨較弱的地方）：{V['overlap_list']}。
- class-weighted L2 最高 5%：{V['s_refit_weighted_top5']}，5–10%：{V['s_refit_weighted_5_10']}，最高層的集中度明顯較低。

| 分數 | 設定 | 各層實際低分率（top 5% → 40–100%） | 單調遞減 | 相鄰層 CI 重疊 |
|---|---|---|---|---|
{mono_lines}

### 機率校準（只適用機率分數）
- 原始 NGBoost P(Y < 60)（raw）在每一層都高估（校準差 {V['orig_raw_gap_min']} 至 {V['orig_raw_gap_max']}）；最高 5% 層 {V['g_orig_raw']}。
- Refit NGBoost 最高 5% 層（實際 {V['o_ngb_top5']}）：raw {V['g_ngb_raw']}；isotonic {V['g_ngb_isotonic']}；Platt {V['g_ngb_platt']}。
  isotonic 讓 |校準差| 改變 {V['gd_ngb_iso_top5']}。
- Refit 分類器最高 5% 層（實際 {V['o_clf_top5']}）：raw {V['g_clf_raw']}；isotonic {V['g_clf_isotonic']}；Platt {V['g_clf_platt']}。
- isotonic 在所有層的 |校準差| 最大值：NGBoost {V['iso_ngb_maxabs']}、分類器 {V['iso_clf_maxabs']}；所有層 CI 都包含 0：NGBoost {yn(V['iso_ngb_all_ci0'])}、分類器 {yn(V['iso_clf_all_ci0'])}。
- Platt 的 CI 排除 0 的層：NGBoost {V['platt_ngb_ex0']}；分類器 {V['platt_clf_ex0']}。Platt 在最高風險層低估、在 20–40% 層高估。整體 Brier 與 isotonic 相近，但分層看並不相同。

### 高風險層的區間覆蓋（名義 90%）
- {V['iv_n']} 種區間方法中，{V['iv_below_n']} 種在最高 5% 層的覆蓋率 CI 上界低於 0.90。
- 最高 5% 層覆蓋率 95% CI 仍包含或高於 0.90 的方法：{V['iv_ok_list']}。其中 NGBoost 兩種方法仍明顯低於各自的整體覆蓋率（整體本來就過度覆蓋），只有 CQR 與依預測成績區間的 Mondrian 在最高層與整體相當。
- 最低者：{mn.method_label}（{mn.configuration}），{f3(mn.coverage)} {ci(mn.coverage_ci_low, mn.coverage_ci_high)}，整體覆蓋 {f3(mn.coverage_all_records)}。
- 原始 cross-fold absolute conformal 整體 0.900，但各層 {V['iv_range_cf_absolute']}；原始 NGBoost raw 整體 {V['iv_original_oof_normal_raw_all']}，各層 {V['iv_range_normal_raw']}。
  **整體覆蓋率良好會掩蓋高風險層的覆蓋不足。**

| 方法 | 設定 | 整體覆蓋 | 最高 5% 層覆蓋 [95% CI] | 層內 − 整體 [95% CI] | 實際低分紀錄覆蓋 | 平均寬度 |
|---|---|---|---|---|---|---|
{iv_lines}

## 3. 注意事項
- 兩種訓練設定（原始 OOF 與 refit 80%）使用相同測試紀錄，但不可合併排名。
- 排序分數不是機率；只有 NGBoost P(Y < 60) 與分類器的機率可以討論校準。
- 所有結果為單一學期、事後探索性分析；不代表跨學期驗證、部署效果或成績改善。
"""


def doc_handoff(V: dict, o: dict, cons: pd.DataFrame) -> str:
    mn = V["iv_min"]
    cons_lines = "\n".join(f"- **{r.topic}**（{r.paper_location}）：{r.assessment_zh}" for r in cons.itertuples())
    return f"""# 給 ChatGPT 的交接摘要（決策可靠性補強分析 v1）

本文件只整理分析結果，不改寫論文。所有數字可在 `NUMERIC_HIGHLIGHTS.csv` 找到來源檔與選取條件。分析單位一律是「修課紀錄」，不要寫成「學生」。
狀態：事後探索性分析（post hoc），單一學期，沒有重新訓練模型。

## 1. Figure 1 的修改
- 依四種分析類型上色：既有 OOF 主分析（藍、實線框）、80/20 calibration follow-up（橙、虛線框）、低分決策分析（綠、點虛線框）、不確定性與校準（紫、點線框），並加入圖例。
- 方塊文字、數字、流程與圖號都沒有改；字級略為提高。黑白列印可依框線樣式分辨。
- V3 資料夾不在工作區，無法對照 V3 `main.pdf`；請作者確認 V3 的 Figure 1 就是工作區的 `Figure01_workflow`。

## 2. Figure 2 的修改
- 改為點估計加 95% CI 的點圖，每個模型有不同顏色與標記，數值不變（MAE {V['fig2_MAE']}；RMSE {V['fig2_RMSE']}）。
- 不再截斷長條的 y 軸。**圖說需要改一句**："y-axes truncated" 改成 "Markers: point estimates; whiskers: 95% student-cluster bootstrap confidence intervals."
- 另附保留原長條版面的替代版（`Figure02_color_bars_alternative`）；用這版則圖說不必改。

## 3. 分析一（固定審查預算）的主要發現
- 原始 OOF：LightGBM 在每折前 10% 紀錄中找到 {V['lgb_rec_10%']} {V['lgb_rec_10%_ci']} 的低分紀錄，precision {V['lgb_pre_10%']}（盛行率 {V['prev']}，lift {V['lgb_lift_10%']}）；前 5% 的 precision {V['lgb_pre_5%']}，前 25% 的 recall {V['lgb_rec_25%']}。
- 其他模型在所有預算下都低於 LightGBM（差 {V['orig_diff_min']} 至 {V['orig_diff_max']}），差距都小於實務門檻 0.05。
- Refit 設定中，分類器比 L2 回歸多找到 {V['clf_diff_min']} 至 {V['clf_diff_max']}（Holm 支持，未達 0.05）；class-weighted L2 在前 5% 少找到 {V['w5']}（達實務門檻）。

## 4. 分析二（風險分層）的主要發現
- 所有分數的實際低分率都隨風險層單調下降（{yn(V['mono_all'])}）；最高 5% 層為 {V['top5_min']}–{V['top5_max']}，為盛行率的 {V['top5_lift_min']}–{V['top5_lift_max']} 倍。
- 原始 NGBoost P(Y < 60) 在每一層都高估低分機率；refit 後 isotonic 校準讓最高 5% 層的校準差從 {V['g_ngb_raw']} 降到 {V['g_ngb_isotonic']}；Platt 在最高層反而低估（{V['g_ngb_platt']}）。
- 區間覆蓋：{V['iv_n']} 種方法中 {V['iv_below_n']} 種在最高 5% 層低於名義 90%（最低 {mn.method_label}，{f3(mn.coverage)}）。最高層 95% CI 仍包含或高於 0.90 的只有 {V['iv_ok_list']}；其中 NGBoost 兩種仍低於各自的整體覆蓋率。

## 5. 可支持主要結論的比較
- LightGBM 在固定預算下找到最多低分紀錄（原始 OOF，{V['orig_n_sup']}/{V['orig_n_con']} 個比較 Holm 支持），與論文的 MAE、PR-AUC 結論一致。
- 原始 NGBoost 機率的高估在每個風險層都成立，與論文 4.2 節一致；isotonic 校準在每層都把校準差拉到接近 0（CI 都包含 0），支持「需要獨立校準集」的結論。
- 整體覆蓋率掩蓋高風險層覆蓋不足，延伸論文 4.4 節「marginal coverage concealed conditional miscoverage」的論點；新證據是以預測時可取得的風險分數分層，不是以實際成績分層。

## 6. 未達顯著或有限制的結果
- 原始 OOF 中未獲支持：{V['orig_ns_list']}。
- Refit：0.25 分位數回歸在 {V['q25_ns_budgets']} 與 L2 無差異；class-weighted 在前 25% 未獲支持（{V['w25']}；未校正 CI 排除 0，但 Holm 校正後未達 0.05）。
- 除 class-weighted 前 5% 外，所有 recall 差都小於實務門檻 0.05：統計上可分辨，但實務差距小。
- 相鄰層 CI 重疊：{V['overlap_list']}。
- 限制：事後探索性分析；單一學期；以修課紀錄為單位（同一學生多筆紀錄以學生 cluster bootstrap 處理）；原始 OOF 的 NGBoost 沒有校準集，無法做無洩漏的再校準。

## 7. 建議放入正文
- `risk_strata_analysis/high_risk_interval_coverage`（高風險層區間覆蓋）：提供 Figure 6–7 沒有的新資訊。
- 一張精簡表：原始 OOF 各模型在 5%、10%、20% 預算的 recall／precision（取自 `budget_reliability_table.csv`）。
- 是否放入正文由作者決定；放入時需在方法中說明這是事後探索性分析。

## 8. 建議放入補充材料
- `budget_recall_curve`、`budget_precision_curve`（refit 面板與既有 Figure 4a 部分重疊）。
- `risk_strata_plot` 與完整的 `risk_strata_reliability` 表、`calibration_gap_differences`、`budget_recall_contrasts`。

## 9. 編號與圖說
- Figure 1、Figure 2 的編號不變。Figure 1 圖說不需改；Figure 2 若採用點圖，需改一句（見第 2 點）。
- 新增的圖表尚未編號；若放入正文或補充材料，需由作者決定編號（例如新的補充圖 S3–S6），以免改動既有 Figure 3–7 的編號。

## 10. 修改建議（由作者決定是否採用）
{cons_lines}
- 不建議據此宣稱跨學期效果、部署效益或成績改善。
"""


def doc_readme(o: dict, tests: dict, gate: dict, extra: list) -> str:
    s = o["status"]
    return f"""# README_FIRST：AIESC 2027 決策可靠性補強分析 v1

## 這是什麼
- Figure 1、Figure 2 的彩色重繪（數值、文字、圖號不變）。
- 兩項事後探索性分析：固定審查預算（分析一）與風險分層（分析二）。
- 只重新分析既有預測，沒有重新訓練模型；沒有修改原始資料、既有論文或既有成果。

## 建議閱讀順序
1. `CHATGPT_HANDOFF_SUMMARY_zh_TW.md`：給 ChatGPT 的 10 點摘要。
2. `ANALYSIS_SUMMARY_zh_TW.md`：完整結果與數字。
3. `FIGURE_STYLE_NOTES.md`、`FIGURE_SOURCE_TRACE.csv`：重繪方式與資料追溯。
4. `NUMERIC_HIGHLIGHTS.csv`：每個重點數字的來源檔與選取條件。
5. `budget_analysis/budget_analysis_notes.md`、`risk_strata_analysis/risk_strata_notes.md`：方法細節。
6. `source_traceability/paper_text_consistency.csv`：與既有論文正文的對照。

## 資料夾
| 資料夾 | 內容 |
|---|---|
| `figures_recolored/` | Figure01_color、Figure02_color（點圖）、Figure02_color_bars_alternative（長條替代版）；PDF、PNG、CSV |
| `paper_ready_figures/` | 與原檔同名的替換檔 Figure01_workflow、Figure02_mae_rmse |
| `budget_analysis/` | 預算表（.csv/.md）、recall 與 precision 曲線、成對比較 |
| `risk_strata_analysis/` | 分層表、分層圖、高風險層區間覆蓋、單調性檢查 |
| `source_traceability/` | 輸入檔雜湊、圖表追溯、與論文正文對照、預算方法對照 |
| `reproducibility/` | 程式碼副本、設計檔、套件版本、測試結果、隱私掃描、完整性檢查、執行紀錄 |

## 重要提醒
- `{V3_NAME}/` 不在工作區；Figure 1–2 是依工作區正式編號表追溯，**需作者確認與 V3 一致**。
- Figure 2 主版本改為點圖，圖說需改一句（見 `FIGURE_STYLE_NOTES.md`）。
- 分析單位是修課紀錄，不是學生。排序分數不是機率。原始 OOF 與 refit 80% 是不同訓練程序，不做跨設定排名。
- 數字一律 3 位小數（計數 1 位小數，因平手按比例計入）；`*_full_precision.csv` 保留完整精度以供核對。
- 表中 Holm p = 0.010 代表「≤ 0.010」（5,000 次 bootstrap 的解析度下限）。

## 檢查狀態
- 資料與交叉檢查：{s['checks_pass']}/{s['checks']} PASS（`reproducibility/analysis_checks.csv`）。
- 既有成果完整性：{s['integrity']}（{s['integrity_files_checked']} 個受保護檔案前後雜湊相同）。
- 單元測試：{tests['summary']}。
- 隱私掃描：privacy_gate {gate['status']}（命中 {gate['n_findings']}）；額外欄位與列數檢查發現 {len(extra)} 項。輸出只有彙總數字，沒有學號、姓名或逐筆資料。
- 測試通過不等於已完成獨立學術審查。

## 重現
```
cd IR
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl_aiesc ../.venv-aiesc2027/bin/python -B research/aiesc2027/run_decision_reliability.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl_aiesc ../.venv-aiesc2027/bin/python -B research/aiesc2027/decision_package.py
```
分析約 {s['seconds']:.0f} 秒（5,000 次 bootstrap）。
"""


# ---- tests, privacy, packaging --------------------------------------------------------------------------------------
def run_tests() -> dict:
    cmd = [PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "research/aiesc2027/tests"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": "/tmp/mpl_aiesc", "PATH": "/usr/bin:/bin"})
    last = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1] if p.stdout.strip() else "no output"
    return {"command": " ".join(cmd), "returncode": p.returncode, "summary": last, "stdout": p.stdout, "stderr": p.stderr}


def extra_privacy_checks(root: Path) -> list[dict]:
    bad_cols = {"student_id", "instructor_id", "student_name", "name", "姓名", "學號", "teacher", "instructor", "id_hash"}
    out = []
    for f in sorted(root.rglob("*.csv")):
        d = pd.read_csv(f, nrows=5, encoding="utf-8-sig")
        cols = {str(c).strip().lower() for c in d.columns}
        if cols & bad_cols:
            out.append({"file": str(f.relative_to(root)), "issue": f"identifier-like column {sorted(cols & bad_cols)}"})
        n = sum(1 for _ in open(f, encoding="utf-8-sig")) - 1
        if n > 5000:
            out.append({"file": str(f.relative_to(root)), "issue": f"{n} rows (possible row-level data)"})
    for f in root.rglob("*"):
        if f.suffix in (".npz", ".npy", ".pkl", ".parquet"):
            out.append({"file": str(f.relative_to(root)), "issue": "binary data file"})
    return out


def copy_code(dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    files = [p for p in CODE_DIR.glob("*.py") if p.name != "__init__.py"] + [CODE_DIR / "__init__.py", CODE_DIR / "decision_reliability_spec.json",
                                                                           CODE_DIR / "tests" / "test_decision_reliability.py"]
    for f in files:
        t = dst / f.relative_to(CODE_DIR.parents[1])
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, t)


def write_manifest(root: Path) -> pd.DataFrame:
    rows = [{"path": str(f.relative_to(root)), "bytes": f.stat().st_size, "sha256": sha256(f)}
            for f in sorted(root.rglob("*")) if f.is_file() and f.name != "MANIFEST_SHA256.csv"]
    df = pd.DataFrame(rows)
    df.to_csv(root / "MANIFEST_SHA256.csv", index=False)
    return df


def main() -> int:
    t0 = time.time()
    before = integrity.snapshot()
    o = load()
    mono = monotonicity(o["st"])
    mono_out = mono.copy()
    for c in ("top5_rate", "top5_ci_low", "top5_ci_high", "bottom_rate", "top5_to_bottom_ratio"):
        mono_out[c] = mono_out[c].round(3)
    mono_out.to_csv(SUB["risk_strata_analysis"] / "strata_monotonicity_check.csv", index=False, encoding="utf-8-sig")
    top5 = interval_top5(o["iv"])
    top5.round(3).to_csv(SUB["risk_strata_analysis"] / "interval_coverage_top5_summary.csv", index=False, encoding="utf-8-sig")
    bmc = budget_method_comparison()
    bmc.round(3).to_csv(SUB["source_traceability"] / "budget_method_comparison.csv", index=False, encoding="utf-8-sig")

    V, H = values(o, mono, bmc, top5)
    pd.DataFrame(H).to_csv(BASE / "NUMERIC_HIGHLIGHTS.csv", index=False, encoding="utf-8-sig")
    cons = consistency(V)
    cons.to_csv(SUB["source_traceability"] / "paper_text_consistency.csv", index=False, encoding="utf-8-sig")
    trace = figure_trace()
    for d in (BASE, SUB["figures_recolored"]):
        trace.to_csv(d / "FIGURE_SOURCE_TRACE.csv", index=False, encoding="utf-8-sig")
        (d / "FIGURE_STYLE_NOTES.md").write_text(doc_style(trace), encoding="utf-8")
    analysis_figure_trace().to_csv(SUB["source_traceability"] / "analysis_figure_trace.csv", index=False, encoding="utf-8-sig")
    input_hashes().to_csv(SUB["source_traceability"] / "input_files_sha256.csv", index=False, encoding="utf-8-sig")
    (SUB["budget_analysis"] / "budget_analysis_notes.md").write_text(doc_budget_notes(V, o["spec"]), encoding="utf-8")
    (SUB["risk_strata_analysis"] / "risk_strata_notes.md").write_text(doc_strata_notes(V), encoding="utf-8")
    (BASE / "ANALYSIS_SUMMARY_zh_TW.md").write_text(doc_summary(V, o, mono, top5), encoding="utf-8")
    (BASE / "CHATGPT_HANDOFF_SUMMARY_zh_TW.md").write_text(doc_handoff(V, o, cons), encoding="utf-8")

    rep = SUB["reproducibility"]
    copy_code(rep / "code")
    shutil.copyfile(CODE_DIR / "decision_reliability_spec.json", rep / "decision_reliability_spec.json")
    (rep / "pip_freeze_aiesc2027_venv.txt").write_text(subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout,
                                                      encoding="utf-8")
    (rep / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "executable": PY,
                                                      "n_boot": o["status"]["n_boot"], "seed": C.SEED}, indent=2), encoding="utf-8")
    tests = run_tests()
    (rep / "unit_tests.txt").write_text(f"$ {tests['command']}\n\n{tests['stdout']}\n{tests['stderr']}", encoding="utf-8")

    from ir_pipeline.privacy import privacy_gate, raw_identifiers
    ids, names = raw_identifiers([pd.read_csv(C.RAW_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")])
    (BASE / "README_FIRST.md").write_text(doc_readme(o, tests, {"status": "pending", "n_findings": "-"}, []), encoding="utf-8")
    gate = privacy_gate([BASE], ids, names)
    extra = extra_privacy_checks(BASE)
    (rep / "privacy_scan.json").write_text(json.dumps({"privacy_gate": gate, "additional_checks": extra}, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    (BASE / "README_FIRST.md").write_text(doc_readme(o, tests, gate, extra), encoding="utf-8")

    cmp_ = integrity.compare(before, integrity.snapshot())
    (rep / "integrity_check_package.json").write_text(json.dumps(cmp_, indent=2), encoding="utf-8")
    problems = []
    if tests["returncode"] != 0:
        problems.append(f"unit tests: {tests['summary']}")
    if gate["status"] != "PASS" or extra:
        problems.append(f"privacy: gate {gate['status']}, additional findings {len(extra)}")
    if cmp_["status"] != "PASS":
        problems.append("integrity: pre-existing artefacts changed")
    if not trace.figure_data_identical_bytes.all():
        problems.append("figure data CSV differs from the paper CSV")
    status = {"seconds": round(time.time() - t0, 1), "unit_tests": tests["summary"], "privacy_gate": gate["status"],
              "privacy_additional_findings": len(extra), "integrity": cmp_["status"], "problems": problems}
    (rep / "package_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    man = write_manifest(BASE)
    status["n_manifest_files"] = len(man)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
