"""Paper tables (CSV + Markdown + LaTeX). Values are rounded to 3 decimals; p-values below 0.001 are
written as "<0.001"."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def f3(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "Not Available"
    if isinstance(x, (int, np.integer)):
        return f"{x:,d}"
    if isinstance(x, (float, np.floating)) and float(x).is_integer() and abs(x) >= 1000:
        return f"{int(x):,d}"
    return f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x)


def fp(p) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ci(lo, hi) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    esc = lambda s: str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("<", r"$<$").replace(">=", r"$\geq$")  # noqa: E731
    cols = "l" + "r" * (df.shape[1] - 1)
    body = [" & ".join(esc(c) for c in df.columns) + r" \\ \midrule"]
    body += [" & ".join(esc(v) for v in r.values) + r" \\" for _, r in df.iterrows()]
    return "\n".join([r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
                      r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{cols}}}", r"\toprule", *body,
                      r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def save(df: pd.DataFrame, raw: pd.DataFrame | None, out: Path, stem: str, caption: str, notes: list[str]):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{stem}.csv", index=False, encoding="utf-8-sig")
    if raw is not None:
        raw.to_csv(out / f"{stem}_full_precision.csv", index=False, encoding="utf-8-sig")
    md = f"**{caption}**\n\n" + to_markdown(df) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    (out / f"{stem}.tex").write_text(to_latex(df, caption, f"tab:{stem}") + "\n", encoding="utf-8")
    return md


def _pm(per_model, key, metric):
    r = per_model[(per_model.model == key) & (per_model.metric == metric)].iloc[0]
    return r["estimate"], r["ci_low"], r["ci_high"]


def table1(t1: pd.DataFrame, folds: pd.DataFrame, out: Path) -> str:
    df = t1.assign(n=t1["n"].map(lambda v: f"{int(v):,d}"))
    df.columns = ["Item", "n", "Share / note"]
    md = save(df, t1, out, "table1_dataset", "Table 1. Dataset and class distribution",
              ["Primary population: 113-2 enrolments with a valid term grade; 2,073 zero grades without a status code "
               "are excluded (they cannot be distinguished from withdrawals or absences).",
               "Risk label (evaluation only): actual term grade < 60, a fixed research threshold. The official pass "
               "threshold is programme-specific (60 for undergraduate / associate programmes, 70 for master's and "
               "doctoral programmes) and is used only in a sensitivity analysis.",
               "Grade ranges are post-hoc evaluation strata and are never model inputs."])
    fd = folds.copy()
    fd["prevalence_below_60"] = fd["prevalence_below_60"].map(f3)
    fd["mean_grade"] = fd["mean_grade"].map(f3)
    fd.to_csv(out / "table1b_folds.csv", index=False, encoding="utf-8-sig")
    (out / "table1b_folds.md").write_text("**Table 1b. Student-grouped folds**\n\n" + to_markdown(fd) + "\n", encoding="utf-8")
    return md


def table2(overall: pd.DataFrame, per_model: pd.DataFrame, diff: pd.DataFrame, out: Path) -> str:
    rows = []
    for _, r in overall.iterrows():
        k = r["key"]
        mae, ml, mh = _pm(per_model, k, "MAE")
        rmse, rl, rh = _pm(per_model, k, "RMSE")
        row = {"Model": r["model"], "Training loss": r["loss"], "MAE": f3(mae), "MAE 95% CI": ci(ml, mh),
               "RMSE": f3(rmse), "RMSE 95% CI": ci(rl, rh)}
        for m in ("MAE", "RMSE"):
            d = diff[(diff.model == k) & (diff.metric == m)]
            row[f"Δ{m} vs LightGBM [95% CI]"] = "reference" if d.empty else f"{d['difference'].iat[0]:+.3f} {ci(d['ci_low_95'].iat[0], d['ci_high_95'].iat[0])}"
        row.update({"Mean signed error": f3(r["mean_signed_error"]), "Median AE": f3(r["median_ae"]),
                    "P90 AE": f3(r["ae_p90"]), "SD ratio (pred/actual)": f3(r["sd_ratio"])})
        rows.append(row)
    return save(pd.DataFrame(rows), overall, out, "table2_overall",
                f"Table 2. Overall prediction performance (unseen students, 5-fold student-grouped CV, n = {int(overall['n'].iat[0]):,d})",
                ["Existing out-of-fold predictions of the IR study, re-verified (identical rows, folds and targets).",
                 "95% CIs: student-cluster percentile bootstrap, 5,000 replicates, seed 42. Δ = model − LightGBM (negative = lower error).",
                 "Mean signed error = predicted − actual. SD ratio < 1 indicates predictions compressed toward the centre."])


def table3(risk: pd.DataFrame, per_model: pd.DataFrame, ngb_prob: pd.DataFrame, out: Path) -> str:
    rows = []
    for _, r in risk.iterrows():
        k = r["key"]
        row = {"Model": r["model"], "TP": f3(int(r["TP"])), "FP": f3(int(r["FP"])), "FN": f3(int(r["FN"])),
               "TN": f3(int(r["TN"]))}
        for m, name in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("specificity", "Specificity"),
                        ("balanced_accuracy", "Balanced acc."), ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")):
            est, lo, hi = _pm(per_model, k, m)
            row[name] = f"{est:.3f} {ci(lo, hi)}"
        row["Risk probability"] = "Not Available" if k != "ngboost" else "P(Y<60), see below"
        rows.append(row)
    p = ngb_prob[(ngb_prob.threshold == "60") & (ngb_prob.version == "normal_raw")].iloc[0]
    est, lo, hi = _pm(per_model, "ngboost_prob", "pr_auc")
    est2, lo2, hi2 = _pm(per_model, "ngboost_prob", "roc_auc")
    rows.append({"Model": "NGBoost P(Y<60) (probabilistic)", "TP": "-", "FP": "-", "FN": "-", "TN": "-",
                 "Precision": f3(p["precision_at_p0.5"]) + " (p≥0.5)", "Recall": f3(p["recall_at_p0.5"]) + " (p≥0.5)",
                 "F1": f3(p["f1_at_p0.5"]) + " (p≥0.5)", "Specificity": "-", "Balanced acc.": "-",
                 "PR-AUC": f"{est:.3f} {ci(lo, hi)}", "ROC-AUC": f"{est2:.3f} {ci(lo2, hi2)}",
                 "Risk probability": f"Brier {p['brier']:.3f}; BSS {p['brier_skill_score']:+.3f}; ECE {p['ece_10_bins']:.3f}"})
    prev = float(risk["prevalence"].iat[0])
    return save(pd.DataFrame(rows), risk, out, "table3_risk", f"Table 3. Academic risk identification (actual grade < 60; prevalence {prev:.3%})",
                ["Classification rule (fixed score threshold): flagged when the PREDICTED grade < 60. Positive class = actual grade < 60.",
                 "PR-AUC / ROC-AUC rank records by the negative predicted grade; this ranking score is NOT a calibrated risk probability.",
                 "Only NGBoost provides a predictive distribution; P(Y<60) = Φ((60 − μ)/σ). For the other models a risk probability is Not Available "
                 "(a point prediction is not a probability). The p ≥ 0.5 rule is identical to the point rule for a symmetric Normal.",
                 "BSS = Brier skill score against the training-fold prevalence (negative = worse than that constant forecast).",
                 "95% CIs: student-cluster bootstrap (5,000 replicates, seed 42)."])


def table4(bias: pd.DataFrame, out: Path) -> str:
    df = bias.drop(columns=["key"]).copy()
    for c in df.columns:
        if c not in ("model", "grade_range", "n"):
            df[c] = df[c].map(f3)
    df["n"] = df["n"].map(lambda v: f"{int(v):,d}")
    df.columns = ["Model", "Actual grade range", "n", "Mean actual", "Mean predicted", "Median actual", "Median predicted",
                  "MAE", "RMSE", "Mean signed error", "P90 |error|"]
    return save(df, bias, out, "table4_bias", "Table 4. Prediction bias across actual grade ranges",
                ["Mean signed error = predicted − actual (positive = overestimation).",
                 "Grade ranges are defined by the ACTUAL grade after the fact (evaluation strata, not inputs). Conditioning on the actual "
                 "grade mechanically induces regression toward the mean, so range-wise errors describe where errors fall, not difficulty "
                 "of the ranges."])


def table5(unc: dict, out: Path) -> str:
    iv = unc["intervals"]
    names = {"normal_raw": "NGBoost Normal (raw)", "normal_rescaled": "NGBoost Normal (in-fold rescaled)",
             "cf_absolute": "Cross-fold conformal, absolute (post hoc)", "cf_normalised": "Cross-fold conformal, normalised (post hoc)",
             "cf_mondrian": "Cross-fold Mondrian by course category (post hoc)"}
    rows = []
    for method, name in names.items():
        for lv in C.INTERVAL_LEVELS:
            r = iv[(iv.method == method) & (iv.nominal == lv) & (iv.group_by == "all")].iloc[0]
            lo = iv[(iv.method == method) & (iv.nominal == lv) & (iv.group == "0-59")].iloc[0]
            hi = iv[(iv.method == method) & (iv.nominal == lv) & (iv.group == "90-100")].iloc[0]
            rows.append({"Method": name, "Nominal": f"{lv:.2f}", "Coverage": f3(r["coverage"]),
                         "Coverage error": f"{r['coverage_error']:+.3f}", "Mean width": f3(r["mean_width"]),
                         "Median width": f3(r["median_width"]), "Coverage, actual 0-59": f3(lo["coverage"]),
                         "Coverage, actual 90-100": f3(hi["coverage"])})
    sc = unc["scores"]
    n = int(iv.loc[iv.group_by == "all", "n"].iat[0])
    md = save(pd.DataFrame(rows), iv, out, "table5_uncertainty", f"Table 5. Predictive uncertainty and calibration of NGBoost (unseen students, n = {n:,d})",
              [f"Proper scores (Normal): NLL raw {sc.loc[0, 'NLL']:.3f}, in-fold rescaled {sc.loc[1, 'NLL']:.3f}; "
               f"CRPS raw {sc.loc[0, 'CRPS']:.3f}, rescaled {sc.loc[1, 'CRPS']:.3f}.",
               "Coverage error = empirical − nominal coverage. Intervals clipped to [0, 100], as in the IR study.",
               "In-fold rescaled: σ multiplied by a factor estimated on the grouped 10% validation part of each outer TRAINING fold.",
               "Cross-fold conformal (post hoc): the quantile for test fold k uses residuals of the other four folds only (student-disjoint); "
               "no separate calibration set exists, so the split-conformal finite-sample guarantee does not strictly apply. "
               "Split / Mondrian conformal and CQR with an in-fold calibration set: Table 6 (follow-up with refitted models).",
               "Per-range and per-course-category coverage: table5_uncertainty_full_precision.csv."])
    return md


def table_stats(diff: pd.DataFrame, out: Path) -> str:
    df = diff.copy()
    df["Model"] = df["model"].map(lambda k: C.MODELS[k]["label"])
    t = pd.DataFrame({
        "Model vs LightGBM": df["Model"], "Metric": df["metric"], "Model": df["model_value"].map(f3),
        "LightGBM": df["reference_value"].map(f3), "Difference": df["difference"].map(lambda v: f"{v:+.3f}"),
        "95% CI": [ci(a, b) for a, b in zip(df["ci_low_95"], df["ci_high_95"])],
        f"Bonferroni CI (α/{int(df['family_size'].iat[0])})": [ci(a, b) for a, b in zip(df["ci_low_bonf"], df["ci_high_bonf"])],
        "p (bootstrap)": df["p_boot"].map(fp), "p (Holm)": df["p_holm"].map(fp), "Direction": df["direction"],
        "Holm-supported": df["supported_holm_0.05"].map({True: "yes", False: "no"}),
        "≥ practical threshold": df["practically_important"].map({True: "yes", False: "no"}),
    })
    return save(t, diff, out, "table_s1_paired_comparisons", "Table S1. Paired student-cluster bootstrap comparisons against LightGBM",
                ["Family: 4 models × 9 metrics = 36 comparisons; Holm step-down adjustment and Bonferroni-adjusted CIs.",
                 "Practical thresholds (fixed before the risk metrics were computed): 0.5 grade points for MAE / RMSE; 0.05 for the "
                 "classification and ranking metrics.",
                 "Exploratory: the same data were used repeatedly for model selection in the IR study; these are not independent tests."])
