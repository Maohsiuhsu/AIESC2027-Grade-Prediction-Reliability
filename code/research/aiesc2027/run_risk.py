"""AIESC 2027 follow-up 2: risk-oriented decision rules / objectives and the zero-grade sensitivity analysis.

Design fixed in research/aiesc2027/risk_methods_spec.json before any model of this follow-up was fitted.

Usage (from the project root; pinned environment, see README section 12):
  VENV=/home/A6000/hdd2/zhe/IR/.venv-aiesc2027/bin/python
  for f in 0 1 2 3 4; do for p in A B; do $VENV -B research/aiesc2027/risk_fit.py --population $p --fold $f; done; done
  $VENV -B research/aiesc2027/run_risk.py [--n-boot 5000] [--seed 42]

Outputs: outputs/aiesc2027/risk_methods/{validation,metrics,stats,tables,figures,results,logs}/
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

from research.aiesc2027 import config as C  # noqa: E402
from research.aiesc2027 import integrity, io_data  # noqa: E402
from research.aiesc2027 import risk_analysis as R  # noqa: E402
from research.aiesc2027 import tables as T  # noqa: E402
from research.aiesc2027.figures import _save  # noqa: E402

log = logging.getLogger("aiesc2027")
BASE = C.OUT / "risk_methods"
DIRS = {k: BASE / k for k in ("validation", "metrics", "stats", "tables", "figures", "results", "logs")}
COLORS = {"lgb": "#0072B2", "l1": "#D55E00", "classifier": "#009E73", "quantile_q25": "#E69F00",
          "weighted": "#56B4E9", "ngboost_prob": "#CC79A7"}
SHORT = {"lgb": "LightGBM-L2", "l1": "LightGBM-L1", "classifier": "Classifier", "quantile_q25": "Quantile q0.25",
         "weighted": "Class-weighted L2", "ngboost_prob": "NGBoost P(Y<60)"}


def setup_logging():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(DIRS["logs"] / "run_risk.log", mode="w", encoding="utf-8"), logging.StreamHandler()):
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
            "args": vars(args), "spec": "research/aiesc2027/risk_methods_spec.json"}


def regression_preds(jobs, n) -> dict:
    out = {}
    for m, key in (("lgb", "lgb"), ("l1", "l1"), ("weighted", "weighted")):
        v = np.full(n, np.nan)
        for j in jobs.values():
            v[j["test_idx"]] = np.clip(j[f"{key}_te"], 0, 100)
        out[m] = v
    return out


def evaluate(y, students, jobs, args, mask=None):
    scores, flags, probs, thr = R.assemble(jobs, y)
    reg = regression_preds(jobs, len(y))
    met = R.metrics_table(y, scores, flags, reg)
    s = np.ones(len(y), bool) if mask is None else mask
    bs = R.RiskBootstrap(students[s], y[s] < C.RISK_THRESHOLD, {m: v[s] for m, v in scores.items()},
                         {k: v[s] for k, v in flags.items()})
    point = bs.point()
    for _, r in met.iterrows():          # bootstrap point estimates must equal the direct computations
        if mask is None:
            assert abs(point[r["method"]]["pr_auc"] - r["pr_auc"]) < 1e-9
            assert abs(point[r["method"]][f"recall_{r['operating_point']}"] - r["recall"]) < 1e-9
    log.info("bootstrap: %d replicates, seed %d, %d students", args.n_boot, args.seed, bs.G)
    reps = bs.run(args.n_boot, args.seed)
    return {"scores": scores, "flags": flags, "probs": probs, "thr": thr, "reg": reg, "metrics": met,
            "point": point, "reps": reps, "ci": R.per_method_ci(point, reps)}


# ---- tables -------------------------------------------------------------------------------------------
def _c(ci, m, k):
    r = ci[(ci.method == m) & (ci.metric == k)].iloc[0]
    return f"{T.f3(r['estimate'])} {T.ci(r['ci_low'], r['ci_high'])}"


def table8(ev, out: Path):
    met, ci = ev["metrics"].set_index(["method", "operating_point"]), ev["ci"]
    rows = []
    for m in R.METHODS:
        nat, o1, o2 = met.loc[(m, "native")], met.loc[(m, "OP1")], met.loc[(m, "OP2")]
        rows.append({"Method": R.METHODS[m], "PR-AUC [95% CI]": _c(ci, m, "pr_auc"), "ROC-AUC": T.f3(nat["roc_auc"]),
                     "Native: recall / precision": f"{T.f3(nat['recall'])} / {T.f3(nat['precision'])}",
                     "Native: flag rate": T.f3(nat["flag_rate"]),
                     "OP1: F1 [95% CI]": _c(ci, m, "f1_OP1"),
                     "OP1: recall / precision": f"{T.f3(o1['recall'])} / {T.f3(o1['precision'])}",
                     "OP2: recall / precision [95% CI]": f"{T.f3(o2['recall'])} / {_c(ci, m, 'precision_OP2')}",
                     "OP2: flag rate": T.f3(o2["flag_rate"])})
    return T.save(pd.DataFrame(rows), ev["metrics"], out, "table8_risk_methods",
                  "Table 8. Risk-oriented decision rules and objectives (student split, n = 83,494, grade < 60)",
                  ["All models fitted on 80% of each training fold's students; thresholds (OP1, OP2) and isotonic maps "
                   "chosen on the other 20% (calibration students) and applied unchanged to the test fold.",
                   "Native rule: predicted grade < 60 (regressions, quantile) or recalibrated probability >= 0.5.",
                   "OP1: flag budget equal to the calibration prevalence. OP2: calibration recall >= 0.50.",
                   "95% CIs: student-cluster bootstrap, 5000 replicates, seed 42."])


def table_contrasts(con: pd.DataFrame, out: Path):
    t = con.copy()
    t["Difference [95% CI]"] = [f"{r.difference:+.3f} [{r.ci_low_95:+.3f}, {r.ci_high_95:+.3f}]" for r in t.itertuples()]
    t["Holm p"] = t["p_holm"].map(T.fp)
    show = t[["family", "contrast", "candidate_value", "reference_value", "Difference [95% CI]", "Holm p",
              "bonf_ci_excludes_0", "practically_important", "improves"]].copy()
    for c in ("candidate_value", "reference_value"):
        show[c] = show[c].map(T.f3)
    return T.save(show, con, out, "table_s7a_risk_contrasts",
                  "Table S7a. Paired contrasts (primary family: 20 contrasts vs tuned LightGBM-L2; threshold family: 8)",
                  ["'improves' = Holm-adjusted p < 0.05 AND |difference| >= 0.05 AND difference > 0.",
                   "Bootstrap p-values have a floor of about 0.0004 (5000 replicates)."])


def table_thresholds(thr: pd.DataFrame, out: Path):
    g = thr.groupby("method").agg(cut1_min=("grade_cutoff_OP1", "min"), cut1_max=("grade_cutoff_OP1", "max"),
                                  cut2_min=("grade_cutoff_OP2", "min"), cut2_max=("grade_cutoff_OP2", "max"),
                                  thr1_min=("threshold_OP1", "min"), thr1_max=("threshold_OP1", "max"),
                                  thr2_min=("threshold_OP2", "min"), thr2_max=("threshold_OP2", "max")).loc[list(R.METHODS)]
    rows = []
    for m, r in g.iterrows():
        reg = m in R.REGRESSIONS
        f = (lambda a, b: f"{a:.3f}-{b:.3f}")  # noqa: E731
        rows.append({"Method": R.METHODS[m],
                     "OP1 rule (5 folds)": f"pred < {f(r.cut1_min, r.cut1_max)}" if reg else f"score >= {f(r.thr1_min, r.thr1_max)}",
                     "OP2 rule (5 folds)": f"pred < {f(r.cut2_min, r.cut2_max)}" if reg else f"score >= {f(r.thr2_min, r.thr2_max)}"})
    return T.save(pd.DataFrame(rows), thr, out, "table_s7b_thresholds",
                  "Table S7b. Thresholds chosen on the calibration students (range over the 5 folds)",
                  ["For regressions and the quantile model the rule is 'flag if predicted value < cutoff' "
                   "(predicted value >= cutoff counts as not at risk; ties at the cutoff are flagged).",
                   "Classifier and NGBoost scores are raw probabilities before isotonic recalibration."])


def table_regression(met: pd.DataFrame, prob: pd.DataFrame, out: Path):
    r = met[(met.operating_point == "native") & met["MAE"].notna()][["label", "MAE", "RMSE"]].copy()
    for c in ("MAE", "RMSE"):
        r[c] = r[c].map(T.f3)
    p = prob.copy()
    for c in ("mean_probability", "prevalence", "brier", "brier_skill_score", "ece_10_bins"):
        p[c] = p[c].map(T.f3)
    T.save(r.rename(columns={"label": "Model"}), None, out, "table_s7c_regression_accuracy",
           "Table S7c. Point accuracy of the regression variants (80% refits)", [])
    T.save(p.drop(columns=["method"]).rename(columns={"label": "Model"}), None, out, "table_s7d_probabilities",
           "Table S7d. Calibration of the recalibrated risk probabilities", [])


def table9(evA, evB, zero_share, nonzero_met, out: Path):
    a = evA["metrics"].set_index(["method", "operating_point"])
    b = evB["metrics"].set_index(["method", "operating_point"])
    nz = nonzero_met.set_index(["method", "operating_point"])
    rows = []
    for m in R.METHODS:
        rows.append({"Method": R.METHODS[m],
                     "PR-AUC, all 85,567 [95% CI]": _c(evB["ci"], m, "pr_auc"),
                     "OP1 recall, all": T.f3(b.loc[(m, "OP1"), "recall"]),
                     "Native recall, all": T.f3(b.loc[(m, "native"), "recall"]),
                     "Zero grades flagged: native / OP1": f"{T.f3(zero_share[(m, 'native')])} / {T.f3(zero_share[(m, 'OP1')])}",
                     "Non-zero rows: PR-AUC (B / A)": f"{T.f3(nz.loc[(m, 'OP1'), 'pr_auc'])} / {T.f3(a.loc[(m, 'OP1'), 'pr_auc'])}",
                     "Non-zero rows: ROC-AUC (B / A)": f"{T.f3(nz.loc[(m, 'OP1'), 'roc_auc'])} / {T.f3(a.loc[(m, 'OP1'), 'roc_auc'])}"})
    return T.save(pd.DataFrame(rows), evB["metrics"], out, "table9_zero_grade_sensitivity",
                  "Table 9. Sensitivity analysis: 2,073 unexplained zero grades included (n = 85,567)",
                  ["Population B: zero grades included in training, calibration and evaluation; every student keeps "
                   "the test fold and calibration role of the main analysis (50 zero-only students assigned with seed 42).",
                   "'B / A': models trained with zeros (B) vs without (A), both evaluated on the same 83,494 non-zero rows "
                   "(threshold-free metrics only, because the OP1 budget of B includes the zero grades).",
                   f"Prevalence of grade < 60 including zeros: {float(evB['metrics']['prevalence'].iat[0]):.3f}.",
                   "95% CIs: student-cluster bootstrap, 5000 replicates, seed 42."])


# ---- figure -------------------------------------------------------------------------------------------
def figure7(curves, evA, evB, zero_share, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    ax = axes[0]
    met = evA["metrics"].set_index(["method", "operating_point"])
    for m in R.METHODS:
        d = curves[curves.method == m]
        ax.plot(d["flag_rate"], d["recall"], color=COLORS[m], lw=1.0, label=SHORT[m])
    for m, mk in (("lgb", "o"), ("l1", "s")):
        for op, fc in (("native", "white"), ("OP1", COLORS[m]), ("OP2", COLORS[m])):
            r = met.loc[(m, op)]
            ax.scatter(r["flag_rate"], r["recall"], marker=mk if op != "OP2" else "^", s=26, facecolors=fc,
                       edgecolors=COLORS[m], zorder=3)
    prev = float(met["prevalence"].iat[0])
    ax.axvline(prev, color="#999999", lw=0.7, ls=":")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of enrolments flagged")
    ax.set_ylabel("Recall (grade < 60)")
    ax.set_title("(a) Recall vs flag budget")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.text(0.01, 0.97, "open: native rule; filled: OP1; triangle: OP2\ndotted line: prevalence", fontsize=5.8,
            va="top", transform=ax.transAxes)
    ax = axes[1]
    w = 0.26
    x = np.arange(len(R.METHODS))
    for i, (op, c) in enumerate((("native", "#BBBBBB"), ("OP1", "#0072B2"), ("OP2", "#009E73"))):
        ax.bar(x + (i - 1) * w, [met.loc[(m, op), "recall"] for m in R.METHODS], w, color=c, label=R.OPS[op].split(":")[0])
    ax.set_xticks(x, [SHORT[m] for m in R.METHODS], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall (grade < 60)")
    ax.set_title("(b) Recall by decision rule (n = 83,494)")
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[2]
    b = evB["metrics"].set_index(["method", "operating_point"])
    ax.bar(x - w / 2, [b.loc[(m, "OP1"), "recall"] for m in R.METHODS], w, color="#0072B2", label="All low grades (incl. zeros)")
    ax.bar(x + w / 2, [zero_share[(m, "OP1")] for m in R.METHODS], w, color="#D55E00", label="Zero grades only")
    ax.set_xticks(x, [SHORT[m] for m in R.METHODS], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share flagged at OP1")
    ax.set_title("(c) Zero grades included (n = 85,567)")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    data = pd.concat([curves.assign(panel="a"), evA["metrics"].assign(panel="b"), evB["metrics"].assign(panel="c"),
                      pd.DataFrame([{"method": m, "operating_point": op, "zero_rows_flagged": v, "panel": "c"}
                                    for (m, op), v in zero_share.items()])], ignore_index=True)
    _save(fig, data, out, "figure7_risk_methods")


# ---- main ---------------------------------------------------------------------------------------------
def run(args) -> float:
    t0 = time.time()
    before = integrity.snapshot()
    log.info("integrity snapshot: %d protected files", len(before))

    d = io_data.load_all()
    yA, sA, fA = d.y, d.frame["student_id"].astype(str).to_numpy(), d.fold
    jobsA, infosA = R.load_jobs("A")
    chkA = R.check_jobs(jobsA, yA, sA, fA).assign(population="A")
    for f, j in jobsA.items():           # population A reuses the calibration-split partition exactly
        cj = np.load(C.OUT / "calibration_split" / "jobs" / f"fold{f}.npz")
        ok = all(np.array_equal(j[k], cj[k]) for k in ("test_idx", "cal_idx", "fit_idx"))
        chkA.loc[len(chkA)] = {"check": f"fold {f}: partition identical to follow-up 1", "status": "PASS" if ok else "FAIL",
                               "detail": "", "population": "A"}

    from research.advanced.common import load_frames
    from research.config import ResearchConfig, TARGET
    _, fb, _ = load_frames(ResearchConfig())
    yB, sB = fb[TARGET].to_numpy(float), fb["student_id"].astype(str).to_numpy()
    jobsB, infosB = R.load_jobs("B")
    chkB = R.check_jobs(jobsB, yB, sB, None).assign(population="B")
    foldB = np.full(len(yB), -1)
    for f, j in jobsB.items():
        foldB[j["test_idx"]] = f
    rid_fold = dict(zip(d.frame["row_id"].to_numpy(), fA))
    inA = fb["row_id"].isin(rid_fold).to_numpy()
    same = bool((foldB[inA] == fb.loc[inA, "row_id"].map(rid_fold).to_numpy()).all())
    zero = yB == 0
    rows_ok = int(inA.sum()) == len(yA) and int(zero.sum()) == C.EXPECTED_ZERO_ROWS and bool((inA | zero).all())
    chkB.loc[len(chkB)] = {"check": "non-zero rows keep their main-analysis test fold", "status": "PASS" if same else "FAIL",
                           "detail": "", "population": "B"}
    chkB.loc[len(chkB)] = {"check": "population B = 83,494 non-zero rows + 2,073 zero rows", "status": "PASS" if rows_ok else "FAIL",
                           "detail": f"{int(inA.sum())} + {int(zero.sum())}", "population": "B"}
    chk = pd.concat([chkA, chkB], ignore_index=True)
    chk.to_csv(DIRS["validation"] / "risk_methods_checks.csv", index=False, encoding="utf-8-sig")
    if (chk.status != "PASS").any():
        raise io_data.ValidationError(chk[chk.status != "PASS"].to_string())
    log.info("risk-method checks: %d PASS", len(chk))

    evA = evaluate(yA, sA, jobsA, args)
    con = R.contrasts(evA["point"], evA["reps"])
    probA = R.probability_table(yA, evA["probs"], fA)
    curves = R.recall_flag_curves(yA, evA["scores"])
    evB = evaluate(yB, sB, jobsB, args)
    nonzero = R.metrics_table(yB, evB["scores"], evB["flags"], evB["reg"], rows_mask=~zero)
    zero_share = {(m, op): float(evB["flags"][(m, op)][zero].mean()) for m in R.METHODS for op in R.OPS}
    probB = R.probability_table(yB, evB["probs"], foldB)

    M_ = DIRS["metrics"]
    for name, df in {"A_metrics": evA["metrics"], "A_thresholds": evA["thr"], "A_probabilities": probA,
                     "A_recall_flag_curves": curves, "B_metrics_all_rows": evB["metrics"], "B_metrics_nonzero_rows": nonzero,
                     "B_thresholds": evB["thr"], "B_probabilities": probB,
                     "B_zero_rows_flagged": pd.DataFrame([{"method": m, "operating_point": op, "share_flagged": v,
                                                           "n_zero_rows": int(zero.sum())} for (m, op), v in zero_share.items()]),
                     "fit_info": pd.DataFrame(infosA + infosB)}.items():
        df.to_csv(M_ / f"{name}.csv", index=False, encoding="utf-8-sig")
    evA["ci"].to_csv(DIRS["stats"] / "A_per_method_ci.csv", index=False, encoding="utf-8-sig")
    evB["ci"].to_csv(DIRS["stats"] / "B_per_method_ci.csv", index=False, encoding="utf-8-sig")
    con.to_csv(DIRS["stats"] / "A_contrasts.csv", index=False, encoding="utf-8-sig")
    log.info("metrics and statistics written")

    O = DIRS["tables"]
    table8(evA, O)
    table_contrasts(con, O)
    table_thresholds(evA["thr"], O)
    table_regression(evA["metrics"], probA, O)
    table9(evA, evB, zero_share, nonzero, O)
    figure7(curves, evA, evB, zero_share, DIRS["figures"])
    log.info("tables and figure written")

    if not args.skip_text:
        from research.aiesc2027 import risk_text
        risk_text.write(DIRS["results"], evA=evA, evB=evB, con=con, probA=probA, nonzero=nonzero,
                        zero_share=zero_share, n_zero=int(zero.sum()))
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
