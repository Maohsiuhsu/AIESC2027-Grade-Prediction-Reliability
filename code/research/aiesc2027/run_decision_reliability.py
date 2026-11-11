"""AIESC 2027 decision-reliability update v1: recoloured Figures 1-2, fixed-budget and risk-stratum analyses.

Reads saved out-of-fold scores only (no model is trained) and writes to outputs/aiesc2027/decision_reliability_update_v1/.
Documents and the manifest are produced afterwards by decision_package.py.

Run: python -B research/aiesc2027/run_decision_reliability.py [--n-boot 5000]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
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
from research.aiesc2027 import decision_figures as F  # noqa: E402
from research.aiesc2027 import decision_reliability as D  # noqa: E402
from research.aiesc2027 import integrity  # noqa: E402

BASE = C.OUT / "decision_reliability_update_v1"
DIRS = {k: BASE / k for k in ("figures_recolored", "paper_ready_figures", "budget_analysis", "risk_strata_analysis",
                              "source_traceability", "reproducibility")}
log = logging.getLogger("aiesc.decision")


def r3(df: pd.DataFrame, count_cols=()) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(1 if c in count_cols else 3)
    return out


def ci(a, b):
    return f"[{a:.3f}, {b:.3f}]"


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


# ---- figures 1-2 -----------------------------------------------------------------------------------------------
def recolour(checks: list) -> pd.DataFrame:
    counts = json.loads((C.OUT / "validation" / "data_counts.json").read_text(encoding="utf-8"))
    paper_csv = C.OUT / "paper" / "figures" / "Figure02_mae_rmse.csv"
    fig2 = pd.read_csv(paper_csv, encoding="utf-8-sig")
    ci_ = pd.read_csv(C.OUT / "stats" / "per_model_bootstrap_ci.csv", encoding="utf-8-sig")
    lab = {k: v["label"] for k, v in C.MODELS.items()}
    ref = ci_[ci_.metric.isin(["MAE", "RMSE"])].assign(model=lambda d: d["model"].map(lab))
    m = fig2.merge(ref, on=["model", "metric"], suffixes=("", "_stats"))
    diff = float(np.max(np.abs(m[["estimate", "ci_low", "ci_high"]].to_numpy()
                               - m[["estimate_stats", "ci_low_stats", "ci_high_stats"]].to_numpy())))
    checks.append({"check": "Figure 2 data = stats/per_model_bootstrap_ci.csv (MAE, RMSE)", "status": "PASS" if len(m) == 10 and diff < 1e-9 else "FAIL",
                   "detail": f"{len(m)} rows; max |diff| {diff:.1e}"})
    fig1 = pd.read_csv(C.OUT / "paper" / "figures" / "Figure01_workflow.csv", encoding="utf-8-sig")
    same = dict(zip(fig1.step, fig1.n)) == {"raw": counts["raw_rows"], "term": counts["term_rows"],
                                            "zeros_excluded": counts["zero_rows"], "analysis": counts["analysis_rows"],
                                            "students": counts["students"]}
    checks.append({"check": "Figure 1 counts = validation/data_counts.json", "status": "PASS" if same else "FAIL", "detail": ""})
    out = DIRS["figures_recolored"]
    F.figure1_color(counts, out, "Figure01_color")
    F.figure2_color(fig2, out, "Figure02_color")
    F.figure2_color_bars(fig2, out, "Figure02_color_bars_alternative")
    F.figure1_color(counts, DIRS["paper_ready_figures"], "Figure01_workflow")
    F.figure2_color(fig2, DIRS["paper_ready_figures"], "Figure02_mae_rmse")
    src = C.OUT / "paper" / "figures"
    for d, stem, orig in ((out, "Figure01_color", "Figure01_workflow"), (out, "Figure02_color", "Figure02_mae_rmse"),
                          (out, "Figure02_color_bars_alternative", "Figure02_mae_rmse"),
                          (DIRS["paper_ready_figures"], "Figure01_workflow", "Figure01_workflow"),
                          (DIRS["paper_ready_figures"], "Figure02_mae_rmse", "Figure02_mae_rmse")):
        shutil.copyfile(src / f"{orig}.csv", d / f"{stem}.csv")
    return fig2


# ---- tables --------------------------------------------------------------------------------------------------
def budget_outputs(table: pd.DataFrame, curve: pd.DataFrame, con: pd.DataFrame):
    out = DIRS["budget_analysis"]
    counts = ("records_flagged", "low_grade_captured", "low_grade_total", "low_grade_missed")
    table.to_csv(out / "budget_reliability_table_full_precision.csv", index=False, encoding="utf-8-sig")
    r3(table, counts).to_csv(out / "budget_reliability_table.csv", index=False, encoding="utf-8-sig")
    con.to_csv(out / "budget_recall_contrasts_full_precision.csv", index=False, encoding="utf-8-sig")
    r3(con).to_csv(out / "budget_recall_contrasts.csv", index=False, encoding="utf-8-sig")
    keep = ["configuration", "score_key", "score_label", "score_type", "budget_share", "records_flagged", "low_grade_captured",
            "low_grade_total", "prevalence"]
    r3(curve[keep + ["recall", "random_recall"]], counts).to_csv(out / "budget_recall_curve.csv", index=False, encoding="utf-8-sig")
    r3(curve[keep + ["precision", "lift"]], counts).to_csv(out / "budget_precision_curve.csv", index=False, encoding="utf-8-sig")
    parts = []
    for config, lab in D.CONFIG_LABEL.items():
        t = table[table.configuration == config]
        rows = pd.DataFrame({
            "Score": t["score_label"] + np.where(t["score_type"] == "probability", " (probability)", " (ranking score)"),
            "Budget": (t["budget_share"] * 100).round().astype(int).astype(str) + "%",
            "Flagged / captured": t["records_flagged"].round(1).astype(str) + " / " + t["low_grade_captured"].round(1).astype(str),
            "Recall [95% CI]": [f"{a:.3f} {ci(b, c)}" for a, b, c in zip(t["recall"], t["recall_ci_low"], t["recall_ci_high"])],
            "Precision [95% CI]": [f"{a:.3f} {ci(b, c)}" for a, b, c in zip(t["precision"], t["precision_ci_low"], t["precision_ci_high"])],
            "Lift": t["lift"].map("{:.3f}".format),
            "Records per low grade found": t["records_per_low_grade_found"].map("{:.3f}".format)})
        c = con[con.configuration == config]
        crow = pd.DataFrame({
            "Score": c["score_label"], "Reference": c["reference_label"],
            "Budget": (c["budget_share"] * 100).round().astype(int).astype(str) + "%",
            "Recall difference [95% CI]": [f"{a:+.3f} [{b:+.3f}, {d:+.3f}]" for a, b, d in
                                           zip(c["recall_difference"], c["recall_ci_low_95"], c["recall_ci_high_95"])],
            "Holm p": c["p_holm"].map(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"),
            "Holm < 0.05": np.where(c["supported_holm_0.05"], "yes", "no"),
            "|diff| >= 0.05": np.where(c["practically_important"], "yes", "no")})
        parts += [f"## {lab}", "", f"Prevalence of grade < 60: {t['prevalence'].iat[0]:.3f}; records: "
                  f"{int(round(t['records_flagged'].iat[0] / t['budget_share'].iat[0])):,d}.", "", md_table(rows), "",
                  f"### Paired recall differences ({len(c)} contrasts, Holm within this configuration)", "", md_table(crow), ""]
    head = ["# Fixed review-budget reliability (enrolment records, grade < 60)", "",
            "Within each test fold the share x of records with the highest risk score is flagged; ties at the cut-off "
            "count fractionally, so counts can be non-integer. 95% CIs: student-cluster bootstrap (5,000, seed 42). "
            "Precision differences at a fixed budget move with recall differences and are not tested separately.", ""]
    (out / "budget_reliability_table.md").write_text("\n".join(head + parts), encoding="utf-8")


def strata_outputs(st: pd.DataFrame, gapdiff: pd.DataFrame, iv: pd.DataFrame):
    out = DIRS["risk_strata_analysis"]
    st.to_csv(out / "risk_strata_reliability_full_precision.csv", index=False, encoding="utf-8-sig")
    r3(st).to_csv(out / "risk_strata_reliability.csv", index=False, encoding="utf-8-sig")
    gapdiff.to_csv(out / "calibration_gap_differences_full_precision.csv", index=False, encoding="utf-8-sig")
    r3(gapdiff).to_csv(out / "calibration_gap_differences.csv", index=False, encoding="utf-8-sig")
    iv.to_csv(out / "high_risk_interval_coverage_full_precision.csv", index=False, encoding="utf-8-sig")
    r3(iv).to_csv(out / "high_risk_interval_coverage.csv", index=False, encoding="utf-8-sig")
    plot_cols = ["configuration", "score_key", "score_label", "score_type", "probability_version", "stratum", "stratum_order",
                 "n_records", "n_low_grade", "observed_low_rate", "observed_low_rate_ci_low", "observed_low_rate_ci_high",
                 "mean_predicted_probability"]
    r3(st[plot_cols]).to_csv(out / "risk_strata_plot.csv", index=False, encoding="utf-8-sig")
    parts = ["# Risk-stratum reliability (enrolment records, grade < 60)", "",
             "Strata: within-fold shares of the risk ranking (mid-ranks for ties, so realised shares can differ slightly). "
             "Ranking reliability (observed rate) applies to every score; probability calibration only to probability "
             "scores. 95% CIs: student-cluster bootstrap (5,000, seed 42).", ""]
    for config, lab in D.CONFIG_LABEL.items():
        t = st[st.configuration == config]
        rk = t[t.probability_version == "NA (ranking only)"]
        tab = pd.DataFrame({"Score": rk["score_label"], "Stratum": rk["stratum"], "Records": rk["n_records"].map("{:,d}".format),
                            "Low-grade records": rk["n_low_grade"].map("{:,d}".format),
                            "Observed rate [95% CI]": [f"{a:.3f} {ci(b, c)}" for a, b, c in zip(
                                rk["observed_low_rate"], rk["observed_low_rate_ci_low"], rk["observed_low_rate_ci_high"])],
                            "Lift": rk["lift"].map("{:.3f}".format),
                            "Mean predicted grade": rk["mean_predicted_grade"].map(lambda v: "NA" if pd.isna(v) else f"{v:.3f}"),
                            "Mean actual grade": rk["mean_actual_grade"].map("{:.3f}".format)})
        parts += [f"## {lab}: ranking reliability", "", md_table(tab), ""]
        pr = t[t.probability_version != "NA (ranking only)"]
        if len(pr):
            tab = pd.DataFrame({"Score": pr["score_label"], "Version": pr["probability_version"], "Stratum": pr["stratum"],
                                "Observed rate": pr["observed_low_rate"].map("{:.3f}".format),
                                "Mean predicted P [95% CI]": [f"{a:.3f} {ci(b, c)}" for a, b, c in zip(
                                    pr["mean_predicted_probability"], pr["mean_predicted_probability_ci_low"],
                                    pr["mean_predicted_probability_ci_high"])],
                                "Gap (pred - obs) [95% CI]": [f"{a:+.3f} [{b:+.3f}, {c:+.3f}]" for a, b, c in zip(
                                    pr["calibration_gap"], pr["calibration_gap_ci_low"], pr["calibration_gap_ci_high"])],
                                "Brier": pr["brier"].map("{:.3f}".format)})
            parts += [f"## {lab}: probability calibration", "", md_table(tab), ""]
    (out / "risk_strata_reliability.md").write_text("\n".join(parts), encoding="utf-8")


# ---- cross-checks against published outputs -----------------------------------------------------------------------
def cross_checks(st: pd.DataFrame, iv: pd.DataFrame, table: pd.DataFrame) -> list[dict]:
    rows = []

    def add(name, a, b, tol=1e-9):
        rows.append({"check": name, "status": "PASS" if abs(a - b) <= tol else "FAIL", "detail": f"{a:.6f} vs {b:.6f}"})
    ni = pd.read_csv(C.OUT / "metrics" / "ngboost_intervals.csv", encoding="utf-8-sig")
    ci_ = pd.read_csv(C.OUT / "calibration_split" / "metrics" / "intervals.csv", encoding="utf-8-sig")
    for m in ("normal_raw", "normal_rescaled", "cf_absolute", "cf_normalised", "cf_mondrian"):
        pub = ni[(ni.method == m) & (ni.nominal == 0.9) & (ni.group_by == "all")]["coverage"].iat[0]
        add(f"original {m}: overall 90% coverage = metrics/ngboost_intervals.csv", iv[(iv.method == m)]["coverage_all_records"].iat[0], pub)
    for m in D.CA.METHODS:
        pub = ci_[(ci_.method == m) & (ci_.nominal == 0.9) & (ci_.group_by == "all")]["coverage"].iat[0]
        add(f"refit {m}: overall 90% coverage = calibration_split/metrics/intervals.csv",
            iv[(iv.method == m)]["coverage_all_records"].iat[0], pub)
    prev = pd.read_csv(C.OUT / "metrics" / "risk_threshold60.csv", encoding="utf-8-sig")["prevalence"].iat[0]
    for key, g in st[st.probability_version == "NA (ranking only)"].groupby("score_key"):
        add(f"{key}: strata cover every record and pooled low-grade rate = prevalence", g["n_low_grade"].sum() / g["n_records"].sum(), prev)
    pp = pd.read_csv(C.OUT / "calibration_split" / "metrics" / "probability_recalibration.csv", encoding="utf-8-sig").set_index("version")
    for v, ver in (("raw", "refit_raw"), ("isotonic", "refit_isotonic"), ("platt", "refit_platt")):
        g = st[(st.score_key == "refit_ngboost_prob") & (st.probability_version == v)]
        add(f"refit NGBoost {v}: pooled Brier = probability_recalibration.csv", float((g["brier"] * g["n_records"]).sum() / g["n_records"].sum()),
            pp.at[ver, "brier"], 1e-9)
    g = st[(st.score_key == "ngboost_prob") & (st.probability_version == "raw")]
    add("original NGBoost raw: pooled Brier = probability_recalibration.csv", float((g["brier"] * g["n_records"]).sum() / g["n_records"].sum()),
        pp.at["original_full_fold", "brier"])
    for key, g in table.groupby("score_key"):
        add(f"{key}: flagged records at 10% = 10% of all records", g[g.budget_share == 0.1]["records_flagged"].iat[0], 0.1 * C.EXPECTED_N, 1e-6)
    return rows


def run(args) -> dict:
    t0 = time.time()
    before = integrity.snapshot()
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    fig2 = recolour(checks)
    log.info("figures 1-2 recoloured")

    inp = D.load_inputs()
    checks += inp["checks"].to_dict("records")
    bad = [c for c in checks if c["status"] != "PASS"]
    if bad:
        raise RuntimeError(f"input checks failed: {bad}")
    y, fold, students, scores = inp["y"], inp["fold"], inp["students"], inp["scores"]

    point, curve = D.budget_point(y, fold, scores)
    reps = D.budget_bootstrap(y, fold, students, scores, n_boot=args.n_boot)
    table, con = D.budget_tables(point, reps, scores)
    log.info("budget analysis done (%d scores)", len(scores))
    st, gapdiff = D.strata_tables(y, fold, students, scores, n_boot=args.n_boot)
    log.info("risk strata done")
    iv = D.interval_strata(y, fold, students, scores, inp["iv_orig"], inp["iv_refit"], n_boot=args.n_boot)
    log.info("interval strata done")

    xc = cross_checks(st, iv, table)
    checks += xc
    pd.DataFrame(checks).to_csv(DIRS["reproducibility"] / "analysis_checks.csv", index=False, encoding="utf-8-sig")
    if any(c["status"] != "PASS" for c in checks):
        raise RuntimeError("cross-checks failed; see reproducibility/analysis_checks.csv")

    budget_outputs(table, curve, con)
    strata_outputs(st, gapdiff, iv)
    F.budget_curves(curve, table, "recall", DIRS["budget_analysis"], "budget_recall_curve")
    F.budget_curves(curve, table, "precision", DIRS["budget_analysis"], "budget_precision_curve")
    F.strata_plot(st, DIRS["risk_strata_analysis"], "risk_strata_plot")
    F.interval_plot(iv, DIRS["risk_strata_analysis"], "high_risk_interval_coverage")
    cmp_ = integrity.compare(before, integrity.snapshot())
    (DIRS["reproducibility"] / "integrity_check_analysis.json").write_text(json.dumps(cmp_, indent=2), encoding="utf-8")
    if cmp_["status"] != "PASS":
        raise RuntimeError(f"pre-existing artefacts changed: {cmp_}")
    status = {"seconds": round(time.time() - t0, 1), "n_boot": args.n_boot, "checks": len(checks),
              "checks_pass": sum(c["status"] == "PASS" for c in checks), "integrity": cmp_["status"],
              "integrity_files_checked": cmp_["files_checked"]}
    (DIRS["reproducibility"] / "analysis_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    args = p.parse_args(argv)
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "reproducibility").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(BASE / "reproducibility" / "analysis.log", mode="w")])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    status = run(args)
    log.info("done: %s", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
