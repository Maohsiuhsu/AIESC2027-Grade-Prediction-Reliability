"""Colour figures of the decision-reliability update: recoloured Figures 1-2 and the budget / stratum figures.

Every encoding that uses colour also uses a second channel (marker, line style, border style or hatching), so the
figures remain readable in greyscale. Model hues follow research/aiesc2027/figures.py (Okabe-Ito) so that the
recoloured Figure 2 matches Figures 3-7.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from .decision_reliability import STRATA  # noqa: E402
from .figures import COLORS  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 100, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

MODEL_STYLE = {   # label -> (colour, marker, line style)
    "LightGBM": (COLORS["LightGBM"], "o", "-"),
    "CatBoost": (COLORS["CatBoost"], "s", (0, (5, 2))),
    "FT-Transformer": (COLORS["FT-Transformer"], "^", (0, (1, 1.5))),
    "NGBoost": (COLORS["NGBoost"], "D", (0, (6, 2, 1, 2))),
    "LightGBM-L1": (COLORS["LightGBM-L1"], "v", (0, (3, 1, 1, 1, 1, 1))),
    "NGBoost P(Y<60)": (COLORS["NGBoost P(Y<60)"], "P", (0, (8, 3))),
}
REFIT_STYLE = {
    "refit_lgb": ("#0072B2", "o", "-"), "refit_l1": ("#D55E00", "v", (0, (3, 1, 1, 1, 1, 1))),
    "refit_classifier": ("#4D4D4D", "X", (0, (5, 2))), "refit_quantile_q25": ("#56B4E9", "<", (0, (1, 1.5))),
    "refit_weighted": ("#E69F00", "s", (0, (6, 2, 1, 2))), "refit_ngboost_prob": ("#7F3C8D", "P", (0, (8, 3))),
}
SHORT = {"refit_lgb": "L2 regression", "refit_l1": "L1 regression", "refit_classifier": "Classifier",
         "refit_quantile_q25": "Quantile q=0.25", "refit_weighted": "Class-weighted L2", "refit_ngboost_prob": "NGBoost P(Y<60)"}
WORKFLOW = {   # role -> (fill, border, border style, legend label)
    "data": ("#F2F2F2", "#5A5A5A", "-", None),
    "oof": ("#DCE8F4", "#2E5C8A", "-", "Re-analysis of saved out-of-fold predictions (Sections 4.1-4.4)"),
    "refit": ("#FBE6D2", "#A0522D", (0, (4, 2)), "80/20 calibration-split follow-ups, refitted models"),
    "risk": ("#DDEFE2", "#2E7D4F", (0, (6, 2, 1, 2)), "Risk-oriented decision analysis"),
    "unc": ("#ECDDF1", "#6A3D84", (0, (1, 1.2)), "Uncertainty and probability calibration"),
    "stats": ("#E9EEF3", "#5A5A5A", "-", None),
}


def save(fig, data: pd.DataFrame | None, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)
    if data is not None:
        data.to_csv(out / f"{stem}.csv", index=False, encoding="utf-8-sig")


# ---- Figure 1 (workflow), same text and layout as figures.fig1_flow -------------------------------------------
def figure1_color(counts: dict, out: Path, stem: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(-9, 100)
    ax.axis("off")

    def box(x, y, w, h, text, role, bold=False, fs=7.0, lw=1.1):
        fc, ec, ls, _ = WORKFLOW[role]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=lw, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
                color="#1A1A1A", linespacing=1.18)

    def arrow(x0, y0, x1, y1, role="data"):
        ec, ls = WORKFLOW[role][1], WORKFLOW[role][2]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=ec, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=9))

    steps = [f"Registrar export\n{counts['raw_rows']:,d} enrolments",
             f"Study term 113-2\n{counts['term_rows']:,d} enrolments",
             f"Exclude unexplained\nzero grades\n(n = {counts['zero_rows']:,d})",
             f"Analysis set\n{counts['analysis_rows']:,d} enrolments\n{counts['students']:,d} students"]
    xs = [1, 26, 51, 76]
    for i, (x, s) in enumerate(zip(xs, steps)):
        box(x, 84, 21.5, 12, s, "stats" if i == 3 else "data", bold=i == 3, lw=1.4 if i == 3 else 1.0)
        if i:
            arrow(xs[i - 1] + 22.3, 90, x - 0.8, 90)
    box(1, 59, 52, 14, "Existing out-of-fold predictions of the IR study\n5-fold student-grouped CV, seed 42 (unseen students)\n"
        "registration-time features only\nLightGBM | CatBoost | FT-Transformer | NGBoost | LightGBM-L1", "oof", fs=6.5)
    box(57, 59, 42, 14, "Pre-specified follow-ups (refitted models)\nsame folds; training folds split by student\n"
        "into 80% fit / 20% calibration\nLightGBM L2, L1, classifier, quantile, weighted;\n"
        f"NGBoost; also with zeros (n = {counts['term_rows']:,d})", "refit", fs=6.35)
    arrow(86.5, 83.2, 27, 73.8, "oof")
    arrow(86.5, 83.2, 78, 73.8, "refit")
    box(1, 44, 52, 8, "Alignment audit: data SHA-256, row order, fold partition,\ntargets, registered MAE / RMSE", "oof", fs=6.8)
    box(57, 44, 42, 8, "Checks: student-disjoint fit / calibration / test;\nreproduction of saved predictions", "refit", fs=6.8)
    arrow(27, 58.2, 27, 52.8, "oof")
    arrow(78, 58.2, 78, 52.8, "refit")
    labels = ["RQ1 Overall error\nMAE, RMSE", "RQ1 Risk identification\nP, R, F1, PR-AUC, ROC-AUC\n+ tuned thresholds,\n"
              "objectives, zero grades", "RQ2 Grade-range bias\nsigned error by\nactual grade range",
              "RQ3 Uncertainty\nNLL, coverage, width\n+ conformal, CQR,\nrecalibration"]
    roles = ["oof", "risk", "oof", "unc"]
    for i, (t, role) in enumerate(zip(labels, roles)):
        x = 1 + i * 25
        box(x, 17, 22, 18, t, role, fs=6.6, lw=1.3 if role in ("risk", "unc") else 1.1)
        arrow(27, 43.2, x + 11, 35.8, "oof")
    for i in (1, 3):
        arrow(78, 43.2, 1 + i * 25 + 13, 35.8, "refit")
    box(6, 1, 88, 8.5, "Paired student-cluster bootstrap (5,000 replicates, seed 42); Holm / Bonferroni control; "
        "exploratory, single semester", "stats", fs=6.6)
    for i in range(4):
        arrow(12 + i * 25, 16.2, 50, 10.3)
    handles = [Patch(fc=WORKFLOW[r][0], ec=WORKFLOW[r][1], ls=WORKFLOW[r][2], lw=1.2, label=WORKFLOW[r][3])
               for r in ("oof", "refit", "risk", "unc")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=False, fontsize=6.8,
              handlelength=2.6, handleheight=1.2, columnspacing=1.6)
    data = pd.DataFrame({"step": ["raw", "term", "zeros_excluded", "analysis", "students"],
                         "n": [counts["raw_rows"], counts["term_rows"], counts["zero_rows"], counts["analysis_rows"],
                               counts["students"]]})
    save(fig, data, out, stem)


# ---- Figure 2 (MAE / RMSE) -----------------------------------------------------------------------------------
def figure2_color(fig2: pd.DataFrame, out: Path, stem: str):
    """Dot-and-whisker version: models on the y-axis, colour + marker per model, value labels."""
    labels = list(dict.fromkeys(fig2["model"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True)
    yy = np.arange(len(labels))[::-1]
    for ax, (m, title) in zip(axes, (("MAE", "(a) MAE, lower is better"), ("RMSE", "(b) RMSE, lower is better"))):
        d = fig2[fig2.metric == m].set_index("model").loc[labels]
        for yi, lab in zip(yy, labels):
            col, mk, _ = MODEL_STYLE[lab]
            r = d.loc[lab]
            ax.errorbar(r["estimate"], yi, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]], fmt=mk,
                        ms=6.5, mfc=col, mec="#222222", mew=0.7, ecolor="#333333", elinewidth=1.0, capsize=3)
            ax.text(r["ci_high"] + 0.03 * (d["ci_high"].max() - d["ci_low"].min()), yi, f"{r['estimate']:.3f}",
                    va="center", fontsize=7.3)
        span = d["ci_high"].max() - d["ci_low"].min()
        ax.set_xlim(d["ci_low"].min() - 0.12 * span, d["ci_high"].max() + 0.32 * span)
        ax.set_xlabel(f"{m} (grade points)")
        ax.set_title(title)
        ax.grid(axis="x", color="#E3E3E3", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(yy, labels)
    fig.text(0.5, -0.03, "Markers: point estimates; whiskers: 95% student-cluster bootstrap CI.", ha="center", fontsize=7.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, fig2, out, stem)


def figure2_color_bars(fig2: pd.DataFrame, out: Path, stem: str):
    """Alternative that keeps the original bar layout: colour + hatching per model."""
    labels = list(dict.fromkeys(fig2["model"]))
    hatches = {"LightGBM": "", "CatBoost": "////", "FT-Transformer": "....", "NGBoost": "xxxx", "LightGBM-L1": "\\\\\\\\"}
    plt.rcParams["hatch.linewidth"] = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, m in zip(axes, ["MAE", "RMSE"]):
        d = fig2[fig2.metric == m].set_index("model").loc[labels]
        x = np.arange(len(d))
        err = np.vstack([d["estimate"] - d["ci_low"], d["ci_high"] - d["estimate"]])
        for xi, lab in zip(x, labels):
            ax.bar(xi, d.at[lab, "estimate"], 0.65, color=MODEL_STYLE[lab][0], edgecolor="#333333", lw=0.7,
                   hatch=hatches[lab], alpha=0.85)
        ax.errorbar(x, d["estimate"], yerr=err, fmt="none", ecolor="#222222", capsize=3, lw=1.0)
        for xi, v in zip(x, d["estimate"]):
            ax.text(xi, v + err[1].max() + 0.04, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_ylim(d["ci_low"].min() - (0.8 if m == "MAE" else 1.0), d["ci_high"].max() + 0.35)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel(f"{m} (grade points)")
        ax.set_title(f"({'a' if m == 'MAE' else 'b'}) {m}, lower is better")
    fig.text(0.5, -0.2, "Bars: point estimates; whiskers: 95% student-cluster bootstrap CI. y-axes truncated.", ha="center",
             fontsize=7.5)
    save(fig, fig2, out, stem)


# ---- budget curves -------------------------------------------------------------------------------------------
def _style(key, label, config):
    return REFIT_STYLE[key] if config == "refit_80" else MODEL_STYLE[label]


def _name(key, label, config):
    return SHORT[key] if config == "refit_80" else label


def budget_curves(curve: pd.DataFrame, table: pd.DataFrame, metric: str, out: Path, stem: str, xmax: float = 0.5):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    titles = {"original_oof": "(a) Original out-of-fold predictions", "refit_80": "(b) Refitted on 80% (follow-ups)"}
    for ax, config in zip(axes, titles):
        c = curve[(curve.configuration == config) & (curve.budget_share <= xmax + 1e-9)]
        t = table[table.configuration == config]
        for key in dict.fromkeys(c["score_key"]):
            g, tt = c[c.score_key == key], t[t.score_key == key]
            lab = g["score_label"].iat[0]
            col, mk, ls = _style(key, lab, config)
            ax.plot(g["budget_share"] * 100, g[metric], ls=ls, color=col, lw=1.3)
            ax.errorbar(tt["budget_share"] * 100, tt[metric],
                        yerr=[tt[metric] - tt[f"{metric}_ci_low"], tt[f"{metric}_ci_high"] - tt[metric]], fmt=mk, ms=4.5,
                        mfc=col, mec="#222222", mew=0.5, ecolor=col, elinewidth=0.9, capsize=1.5, ls="none")
        if metric == "recall":
            ax.plot([0, xmax * 100], [0, xmax], color="#8C8C8C", lw=0.9, ls=(0, (2, 2)))
            ax.text(xmax * 100 * 0.97, xmax * 0.9, "random", ha="right", va="top", fontsize=7, color="#6E6E6E")
            ax.set_ylim(0, 1)
        else:
            prev = c["prevalence"].iat[0]
            ax.axhline(prev, color="#8C8C8C", lw=0.9, ls=(0, (2, 2)))
            ax.text(xmax * 100 * 0.97, prev + 0.012, f"random = prevalence ({prev:.3f})", ha="right", fontsize=7, color="#6E6E6E")
            ax.set_ylim(0, 0.8)
        for b in (5, 10, 15, 20, 25):
            ax.axvline(b, color="#EEEEEE", lw=0.6, zorder=0)
        ax.set_xlim(0, xmax * 100)
        ax.set_xlabel("Records reviewed within each test fold (%)")
        ax.set_title(titles[config])
        handles = [Line2D([], [], color=_style(k, n, config)[0], ls=_style(k, n, config)[2], marker=_style(k, n, config)[1],
                          mfc=_style(k, n, config)[0], mec="#222222", mew=0.5, ms=4.5, lw=1.3, label=_name(k, n, config))
                   for k, n in dict(zip(c["score_key"], c["score_label"])).items()]
        leg = ax.legend(handles=handles, frameon=True, fontsize=6.6, loc="lower right" if metric == "recall" else "upper right",
                        handlelength=3.0, framealpha=0.92, edgecolor="none")
        leg.get_frame().set_facecolor("white")
    axes[0].set_ylabel({"recall": "Recall (share of low-grade records found)", "precision": "Precision (share of flagged records < 60)"}[metric])
    fig.text(0.5, -0.04, "Unit: enrolment records. Markers: budgets 5-25% with 95% student-cluster bootstrap CIs. "
             "Panels use different training conditions and are not ranked together.", ha="center", fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(fig, None, out, stem)


# ---- risk strata ---------------------------------------------------------------------------------------------
def strata_plot(st: pd.DataFrame, out: Path, stem: str):
    order = [s for s, *_ in STRATA]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    x = np.arange(len(order))
    rk = st[st.probability_version == "NA (ranking only)"]
    for ax, config, title in ((axes[0], "original_oof", "(a) Ranking: original OOF"),
                              (axes[1], "refit_80", "(b) Ranking: refitted on 80%")):
        d = rk[rk.configuration == config]
        for key in dict.fromkeys(d["score_key"]):
            g = d[d.score_key == key].sort_values("stratum_order")
            lab = g["score_label"].iat[0]
            col, mk, ls = _style(key, lab, config)
            ax.errorbar(x, g["observed_low_rate"], yerr=[g["observed_low_rate"] - g["observed_low_rate_ci_low"],
                                                         g["observed_low_rate_ci_high"] - g["observed_low_rate"]],
                        color=col, ls=ls, marker=mk, ms=4, mfc=col, mec="#222222", mew=0.4, lw=1.1, capsize=1.5,
                        label=_name(key, lab, config))
        first = d[d.score_key == d["score_key"].iat[0]]
        ax.axhline(first["n_low_grade"].sum() / first["n_records"].sum(), color="#8C8C8C", lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks(x, order, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylim(0, 0.65)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=6.0, loc="upper right", handlelength=2.6)
    axes[0].set_ylabel("Observed share with grade < 60")
    ax = axes[2]
    pr = st[(st.configuration == "refit_80") & (st.probability_version.isin(["raw", "isotonic"]))]
    styles = {("refit_ngboost_prob", "raw"): ("#7F3C8D", "P", (0, (8, 3)), "NGBoost raw"),
              ("refit_ngboost_prob", "isotonic"): ("#7F3C8D", "P", "-", "NGBoost isotonic"),
              ("refit_classifier", "raw"): ("#4D4D4D", "X", (0, (8, 3)), "Classifier raw"),
              ("refit_classifier", "isotonic"): ("#4D4D4D", "X", "-", "Classifier isotonic")}
    for (key, v), (col, mk, ls, lab) in styles.items():
        g = pr[(pr.score_key == key) & (pr.probability_version == v)].sort_values("stratum_order")
        ax.plot(g["mean_predicted_probability"], g["observed_low_rate"], ls=ls, marker=mk, color=col, ms=4,
                mfc=col if v == "isotonic" else "white", mec=col, lw=1.1, label=lab)
    ax.plot([0, 0.65], [0, 0.65], color="#8C8C8C", lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("Mean predicted P(grade < 60)")
    ax.set_ylabel("Observed share with grade < 60")
    ax.set_title("(c) Probability calibration by stratum")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    fig.text(0.5, -0.06, "Strata: within-fold risk-score shares (top 5%, 5-10%, 10-20%, 20-40%, 40-100%). Dashed grey: "
             "prevalence (a, b) or perfect calibration (c). Whiskers: 95% student-cluster bootstrap CIs.",
             ha="center", fontsize=6.8)
    fig.tight_layout(w_pad=1.0)
    save(fig, None, out, stem)


INTERVAL_STYLE = {
    "normal_raw": ("#CC79A7", "D", "-", "NGBoost raw"), "normal_rescaled": ("#CC79A7", "d", (0, (5, 2)), "NGBoost rescaled"),
    "cf_absolute": ("#0072B2", "o", (0, (1, 1.5)), "Cross-fold CP, absolute"),
    "cf_normalised": ("#009E73", "^", (0, (6, 2, 1, 2)), "Cross-fold CP, normalised"),
    "cf_mondrian": ("#E69F00", "s", (0, (3, 1, 1, 1)), "Cross-fold Mondrian (category)"),
    "ngb_raw": ("#CC79A7", "D", "-", "NGBoost raw"), "ngb_rescaled": ("#CC79A7", "d", (0, (5, 2)), "NGBoost rescaled"),
    "split_abs": ("#0072B2", "o", (0, (1, 1.5)), "Split CP, absolute"),
    "split_norm": ("#009E73", "^", (0, (6, 2, 1, 2)), "Split CP, normalised"),
    "mondrian_category": ("#E69F00", "s", (0, (3, 1, 1, 1)), "Mondrian, category"),
    "mondrian_predicted_range": ("#56B4E9", "<", (0, (8, 3)), "Mondrian, predicted range"),
    "cqr_lightgbm": ("#4D4D4D", "X", "-", "CQR (LightGBM)"),
    "split_abs_lightgbm": ("#D55E00", "v", (0, (2, 1)), "Split CP around LightGBM"),
}


def interval_plot(iv: pd.DataFrame, out: Path, stem: str):
    order = list(dict.fromkeys(iv.sort_values("stratum_order")["stratum"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    x = np.arange(len(order))
    for ax, config, title in ((axes[0], "original_oof", "(a) Original NGBoost OOF"), (axes[1], "refit_80", "(b) Refitted on 80%")):
        d = iv[iv.configuration == config]
        for m in dict.fromkeys(d["method"]):
            g = d[d.method == m].sort_values("stratum_order")
            col, mk, ls, lab = INTERVAL_STYLE[m]
            ax.plot(x, g["coverage"], ls=ls, marker=mk, color=col, ms=4, mfc=col, mec="#222222", mew=0.4, lw=1.1, label=lab)
        ax.axhline(0.9, color="#8C8C8C", lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks(x, order, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(title)
        ax.set_ylim(0.6, 1.0)
        ax.legend(frameon=False, fontsize=6.0, loc="lower right", ncol=1, handlelength=2.6)
    axes[0].set_ylabel("Empirical coverage of 90% intervals")
    fig.text(0.5, -0.05, "Strata by the risk ranking of the interval's base model (NGBoost P(Y<60); LightGBM-L2 for CQR and "
             "split CP around LightGBM). Dashed grey: nominal 0.90.", ha="center", fontsize=6.8)
    fig.tight_layout(w_pad=1.0)
    save(fig, None, out, stem)
