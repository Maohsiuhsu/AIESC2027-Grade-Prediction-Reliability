"""AIESC 2027 follow-up: calibration-split conformal / CQR intervals and probability recalibration.

Design fixed in research/aiesc2027/calibration_split_spec.json before any model of this follow-up was fitted.

Usage (from the project root; pinned environment, see README):
  VENV=/home/A6000/hdd2/zhe/IR/.venv-aiesc2027/bin/python
  for f in 0 1 2 3 4; do $VENV -B research/aiesc2027/calibration_fit.py --fold $f; done   # ~1.5 min per fold
  $VENV -B research/aiesc2027/run_calibration.py [--n-boot 5000] [--seed 42]

Outputs: outputs/aiesc2027/calibration_split/{validation,metrics,stats,tables,figures,results,logs}/
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.aiesc2027 import calibration_analysis as CA  # noqa: E402
from research.aiesc2027 import config as C  # noqa: E402
from research.aiesc2027 import integrity, io_data  # noqa: E402
from research.aiesc2027 import tables as T  # noqa: E402
from research.aiesc2027.figures import _save  # noqa: E402

log = logging.getLogger("aiesc2027")
BASE = C.OUT / "calibration_split"
DIRS = {k: BASE / k for k in ("validation", "metrics", "stats", "tables", "figures", "results", "logs")}
STYLE = {"ngb_raw": ("#CC79A7", "o-"), "ngb_rescaled": ("#882255", "s--"), "split_abs": ("#56B4E9", "^-"),
         "split_norm": ("#009E73", "D-"), "mondrian_category": ("#E69F00", "v:"),
         "mondrian_predicted_range": ("#D55E00", "P-"), "cqr_lightgbm": ("#0072B2", "X-"),
         "split_abs_lightgbm": ("#666666", "*:")}
SHORT = {"ngb_raw": "NGB raw", "ngb_rescaled": "NGB rescaled", "split_abs": "Split |r|", "split_norm": "Split |r|/σ",
         "mondrian_category": "Mondrian category", "mondrian_predicted_range": "Mondrian pred. range",
         "cqr_lightgbm": "CQR LightGBM", "split_abs_lightgbm": "Split LightGBM"}


def setup_logging():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(DIRS["logs"] / "run_calibration.log", mode="w", encoding="utf-8"),
              logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)


def environment(args) -> dict:
    vers = {"python": platform.python_version(), "executable": sys.executable}
    for m in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "ngboost", "matplotlib"):
        try:
            vers[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            vers[m] = "not importable"
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "versions": vers, "platform": platform.platform(),
            "args": vars(args), "spec": "research/aiesc2027/calibration_split_spec.json"}


# ---- tables -------------------------------------------------------------------------------------------
def _ci(bs, m, s):
    r = bs[(bs.method == m) & (bs.statistic == s)].iloc[0]
    return r["estimate"], f"{T.f3(r['estimate'])} {T.ci(r['ci_low'], r['ci_high'])}"


def table6(iv: pd.DataFrame, bs: pd.DataFrame, out: Path):
    a = iv[iv.nominal == 0.9]
    rows, raw = [], []
    for m, label in CA.METHODS.items():
        g = a[a.method == m]
        al = g[g.group_by == "all"].iloc[0]
        rng = g[g.group_by == "grade_range"].set_index("group")
        cats = g[g.group_by == "course_category"]
        c_all, s_all = _ci(bs, m, "coverage90_all")
        _, s_low = _ci(bs, m, "coverage90_actual_0_59")
        _, s_w = _ci(bs, m, "mean_width90_all")
        rows.append({"Method": label, "Coverage [95% CI]": s_all, "Mean width [95% CI]": s_w,
                     "Median width": T.f3(al["median_width"]), "Coverage, actual 0-59 [95% CI]": s_low,
                     "Coverage, actual 90-100": T.f3(rng.loc["90-100", "coverage"]),
                     "Lowest course-category coverage": T.f3(cats["coverage"].min())})
        raw.append({"method": m, "coverage": al["coverage"], "mean_width": al["mean_width"],
                    "median_width": al["median_width"], "coverage_0_59": rng.loc["0-59", "coverage"],
                    "coverage_90_100": rng.loc["90-100", "coverage"], "min_category_coverage": cats["coverage"].min()})
    return T.save(pd.DataFrame(rows), pd.DataFrame(raw), out, "table6_calibration_split_intervals",
                  "Table 6. Nominal 90% prediction intervals with an in-fold calibration set (student split, n = 83,494)",
                  ["Every interval for test fold k uses only the calibration students of training fold k (20% of its "
                   "students; disjoint from the proper-training students and from test fold k).",
                   "NGBoost and LightGBM were refitted on the remaining 80% of each training fold; their point accuracy "
                   "differs from the original models (Table S6b).",
                   "Split conformal guarantees marginal coverage >= 90% under exchangeable students; there is no "
                   "guarantee conditional on the actual grade range.",
                   "95% CIs: student-cluster bootstrap, 5000 replicates, seed 42. Intervals clipped to [0, 100]."])


def table7(prob: pd.DataFrame, bs: pd.DataFrame, out: Path):
    rows = []
    for _, r in prob.iterrows():
        _, s_b = _ci(bs, r["version"], "brier")
        rows.append({"Version": r["label"], "Mean P(Y<60)": T.f3(r["mean_predicted_probability"]),
                     "Prevalence": T.f3(r["prevalence"]), "Brier [95% CI]": s_b,
                     "Brier skill score": T.f3(r["brier_skill_score"]), "ECE": T.f3(r["ece_10_bins"]),
                     "Log loss": T.f3(r["log_loss"]), "PR-AUC": T.f3(r["pr_auc"]), "ROC-AUC": T.f3(r["roc_auc"])})
    d = bs[bs.statistic == "brier_difference"]
    notes = ["Recalibration maps (isotonic; logistic on logit p) were fitted on the calibration students of each "
             "training fold only.",
             "Brier skill score reference: prevalence of grade < 60 in the training folds (no test label used).",
             "ECE: 10 equal-width bins."]
    notes += [f"Paired Brier difference {r['method']}: {T.f3(r['estimate'])} {T.ci(r['ci_low'], r['ci_high'])}."
              for _, r in d.iterrows()]
    return T.save(pd.DataFrame(rows), prob, out, "table7_probability_recalibration",
                  "Table 7. NGBoost P(Y < 60) before and after recalibration on the in-fold calibration set", notes)


def supplementary(iv, point, repro, scores, out: Path):
    a = iv[iv.group_by == "all"].pivot_table(index="method", columns="nominal", values=["coverage", "mean_width"])
    a = a.loc[list(CA.METHODS)]
    s = pd.DataFrame({"Method": [CA.METHODS[m] for m in a.index]})
    for lv in C.INTERVAL_LEVELS:
        s[f"Coverage {int(lv * 100)}%"] = [T.f3(v) for v in a[("coverage", lv)]]
        s[f"Width {int(lv * 100)}%"] = [T.f3(v) for v in a[("mean_width", lv)]]
    T.save(s, iv, out, "table_s_calibration_split_all_levels",
           "Table S6a. Coverage and mean width at all nominal levels (calibration-split follow-up)", [])
    for by, stem, tab, title in (("grade_range", "table_s_calibration_split_by_actual_range", "S6e", "actual grade range"),
                            ("predicted_range", "table_s_calibration_split_by_predicted_range", "S6f",
                             "predicted grade range (refitted NGBoost mean)"),
                            ("course_category", "table_s_calibration_split_by_category", "S6g", "course category")):
        g = iv[(iv.nominal == 0.9) & (iv.group_by == by)]
        col = "group_en" if by == "course_category" else "group"
        cov = g.pivot_table(index="method", columns=col, values="coverage").loc[list(CA.METHODS)]
        n = g.drop_duplicates(col).set_index(col)["n"]
        t = pd.DataFrame({"Method": [CA.METHODS[m] for m in cov.index]})
        for c in cov.columns:
            t[f"{c} (n={int(n[c]):,d})"] = [T.f3(v) for v in cov[c]]
        T.save(t, g, out, stem, f"Table {tab}. Empirical coverage of nominal 90% intervals by {title}", [])
    p = point.copy()
    for c in ("MAE", "RMSE", "precision", "recall", "f1"):
        p[c] = p[c].map(T.f3)
    T.save(p, point, out, "table_s_point_accuracy",
           "Table S6b. Point accuracy of the original, reproduced and 80% refitted models (risk rule: prediction < 60)",
           ["Reproduction refits use the full saved training folds; 80% refits are the models behind Tables 6-7."])
    sc = scores.copy()
    for c in ("NLL", "CRPS"):
        sc[c] = sc[c].map(T.f3)
    T.save(sc, scores, out, "table_s_ngboost_scores", "Table S6c. NGBoost NLL and CRPS (original vs 80% refit)", [])
    r = repro[["fold", "recomputed_partition_equals_saved", "lightgbm_max_abs_diff", "ngboost_MAE_refit",
               "ngboost_MAE_saved", "ngboost_NLL_refit", "ngboost_NLL_saved", "ngboost_cov90_refit", "ngboost_cov90_saved",
               "partition_fit_students", "partition_cal_students", "partition_fit_rows", "partition_cal_rows"]].copy()
    for c in r.columns:
        if r[c].dtype.kind == "f":
            r[c] = r[c].map(T.f3)
    T.save(r, repro, out, "table_s_reproduction_gate",
           "Table S6d. Reproduction gate (refits on the full saved training folds) and calibration partitions",
           ["recomputed_partition_equals_saved = False: GroupKFold tie order is CPU dependent; the saved partition is used."])


# ---- figure -------------------------------------------------------------------------------------------
def figure6(iv: pd.DataFrame, prob_rel: pd.DataFrame, prob: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    ranges = [b for b, _, _ in C.GRADE_BANDS]
    ax = axes[0]
    g = iv[(iv.nominal == 0.9) & (iv.group_by == "grade_range")]
    for m in CA.METHODS:
        d = g[g.method == m].set_index("group").loc[ranges]
        c, s = STYLE[m]
        ax.plot(range(5), d["coverage"], s, color=c, ms=3.5, lw=1.0, label=SHORT[m])
    ax.axhline(0.9, color="black", lw=0.7, ls="--")
    ax.set_xticks(range(5), ranges)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Actual grade range")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("(a) 90% intervals by actual grade range")
    ax.legend(frameon=False, fontsize=6, loc="lower center", ncol=2)
    ax = axes[1]
    a = iv[(iv.nominal == 0.9)]
    for m in CA.METHODS:
        c, s = STYLE[m]
        al = a[(a.method == m) & (a.group_by == "all")].iloc[0]
        lo = a[(a.method == m) & (a.group_by == "grade_range") & (a.group == "0-59")].iloc[0]
        ax.scatter(al["mean_width"], al["coverage"], color=c, marker=s[0], s=22)
        ax.scatter(lo["mean_width"], lo["coverage"], facecolors="none", edgecolors=c, marker=s[0], s=22)
        ax.plot([al["mean_width"], lo["mean_width"]], [al["coverage"], lo["coverage"]], color=c, lw=0.5, alpha=0.6)
    ax.axhline(0.9, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Mean interval width (grade points)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("(b) Width vs coverage (filled: all; open: actual 0-59)")
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=":")
    sty = {"original_full_fold": ("#CC79A7", "o--", "Original, raw"), "refit_raw": ("#882255", "s-", "Refit, raw"),
           "refit_isotonic": ("#009E73", "D-", "Refit, isotonic"), "refit_platt": ("#0072B2", "^-", "Refit, Platt")}
    for k, (c, s, name) in sty.items():
        r = prob_rel[(prob_rel.version == k) & (prob_rel.n > 0)]
        ece = prob.set_index("version").loc[k, "ece_10_bins"]
        ax.plot(r["mean_predicted"], r["observed_rate"], s, color=c, ms=3, lw=1.0, label=f"{name} (ECE {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(Y < 60)")
    ax.set_ylabel("Observed share with grade < 60")
    ax.set_title("(c) Reliability of P(Y<60)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.text(0.5, -0.04, "All conformal quantiles and recalibration maps come from an in-fold calibration set "
             "(20% of the training-fold students); NGB = NGBoost refitted on the other 80%.", ha="center", fontsize=7)
    _save(fig, pd.concat([iv[iv.nominal == 0.9].assign(panel="a-b"), prob_rel.assign(panel="c")], ignore_index=True),
          out, "figure6_calibration_split")


def not_available() -> pd.DataFrame:
    return pd.DataFrame([
        ("Conditional (per actual grade range) coverage guarantee", "Not Available",
         "The actual grade is unknown at prediction time; no conformal method can condition on it. Mondrian by "
         "predicted range conditions on what is known, which is a different target."),
        ("CQR around NGBoost or FT-Transformer quantiles", "Not run",
         "Not in the pre-specified design; NGBoost's Normal quantiles are covered by the normalised split method."),
        ("Calibration-split results for the original full-training-fold models", "Not Available",
         "Those models used every training row; a calibration set has to be held out, so the follow-up models are refits."),
    ], columns=["item", "status", "reason"])


# ---- main ---------------------------------------------------------------------------------------------
def run(args) -> float:
    t0 = time.time()
    before = integrity.snapshot()
    log.info("integrity snapshot: %d protected files", len(before))
    d = io_data.load_all()
    log.info("main alignment checks: %d PASS", len(d.checks))
    jobs, infos = CA.load_jobs()
    y, fold = d.y, d.fold
    students = d.frame["student_id"].astype(str).to_numpy()
    cat = d.frame["course_category"].astype(str).to_numpy()
    chk = CA.check_jobs(jobs, infos, fold, students)
    chk.to_csv(DIRS["validation"] / "calibration_split_checks.csv", index=False, encoding="utf-8-sig")
    if (chk.status != "PASS").any():
        raise io_data.ValidationError(chk[chk.status != "PASS"].to_string())
    log.info("calibration-split checks: %d PASS", len(chk))

    repro = CA.reproduction_table(infos)
    iv, bounds = CA.interval_table(jobs, infos, y, cat, fold)
    scores = CA.scoring_table(jobs, infos, y, d.ngb_loc, d.ngb_scale, d.ngb_scale_mult)
    point = CA.point_table(jobs, infos, y, d.preds)
    probs = CA.recalibrated_probabilities(jobs, y, d.ngb_loc, d.ngb_scale)
    prob, rel = CA.probability_table(probs, y, fold)
    M_ = DIRS["metrics"]
    for name, df in {"reproduction_gate": repro, "intervals": iv, "ngboost_scores": scores, "point_accuracy": point,
                     "probability_recalibration": prob, "probability_reliability": rel,
                     "not_available": not_available()}.items():
        df.to_csv(M_ / f"{name}.csv", index=False, encoding="utf-8-sig")
    log.info("metrics written")

    log.info("bootstrap: %d replicates, seed %d", args.n_boot, args.seed)
    bs = CA.bootstrap(bounds[0.9], probs, y, students, args.n_boot, args.seed)
    bs.to_csv(DIRS["stats"] / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    for m in CA.METHODS:            # bootstrap point estimates must equal the direct computations
        r = iv[(iv.method == m) & (iv.nominal == 0.9) & (iv.group_by == "all")].iloc[0]
        e = bs[(bs.method == m) & (bs.statistic == "coverage90_all")]["estimate"].iat[0]
        assert abs(e - r["coverage"]) < 1e-12, m

    table6(iv, bs, DIRS["tables"])
    table7(prob, bs, DIRS["tables"])
    supplementary(iv, point, repro, scores, DIRS["tables"])
    figure6(iv, rel, prob, DIRS["figures"])
    log.info("tables and figure written")

    if not args.skip_text:
        from research.aiesc2027 import calibration_text
        calibration_text.write(DIRS["results"], iv=iv, bs=bs, prob=prob, point=point, repro=repro, scores=scores)
        log.info("results draft written")

    from ir_pipeline.privacy import privacy_gate, raw_identifiers
    ids, names = raw_identifiers([pd.read_csv(C.RAW_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")])
    gate = privacy_gate([BASE], ids, names)
    (DIRS["validation"] / "privacy_scan.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("privacy gate: %s (%d findings)", gate["status"], gate["n_findings"])
    if gate["status"] != "PASS":
        raise RuntimeError("privacy gate failed")
    cmp_ = integrity.compare(before, integrity.snapshot())
    (DIRS["validation"] / "integrity_check.json").write_text(json.dumps(cmp_, indent=2), encoding="utf-8")
    log.info("integrity of pre-existing artefacts: %s (%d files)", cmp_["status"], cmp_["files_checked"])
    if cmp_["status"] != "PASS":
        raise RuntimeError(f"pre-existing artefacts changed: {cmp_}")
    return round(time.time() - t0, 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--skip-text", action="store_true")
    args = p.parse_args(argv)
    setup_logging()
    env = environment(args)
    (DIRS["logs"] / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        sec = run(args)
        status = {"status": "complete", "seconds": sec, **env}
        log.info("done in %.1f s", sec)
    except Exception as e:  # noqa: BLE001
        log.error("FAILED: %s\n%s", e, traceback.format_exc())
        status = {"status": "failed", "error": f"{type(e).__name__}: {e}", **env}
        (DIRS["logs"] / "run_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(1)
    (DIRS["logs"] / "run_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
