"""AIESC 2027 supplementary analysis - full, reproducible run (read-only w.r.t. all earlier artefacts).

Usage (from the project root):
  PY=/home/A6000/hdd1/conda/envs/zhe/bin/python      # see README: the project .venv is not usable here
  $PY -B -m pytest -q -p no:cacheprovider research/aiesc2027/tests
  $PY -B research/aiesc2027/run_all.py [--n-boot 5000] [--seed 42] [--skip-text]

Inputs : see research/aiesc2027/config.py (cleaned data, saved OOF predictions, IR registries)
Outputs: outputs/aiesc2027/{inventory,validation,metrics,stats,tables,figures,results,logs}/
Steps  : integrity snapshot -> inventory -> load + validate + align -> metrics -> bootstrap -> tables ->
         figures -> English results draft -> privacy gate -> integrity re-check.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.aiesc2027 import analysis as A  # noqa: E402
from research.aiesc2027 import bootstrap as B  # noqa: E402
from research.aiesc2027 import config as C  # noqa: E402
from research.aiesc2027 import figures as F  # noqa: E402
from research.aiesc2027 import integrity, inventory, io_data  # noqa: E402
from research.aiesc2027 import tables as T  # noqa: E402
from research.aiesc2027 import uncertainty as U  # noqa: E402

log = logging.getLogger("aiesc2027")
DIRS = {k: C.OUT / k for k in ("inventory", "validation", "metrics", "stats", "tables", "figures", "results", "logs")}


def setup_logging():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(DIRS["logs"] / "run.log", mode="w", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)


def environment(args) -> dict:
    vers = {"python": platform.python_version(), "executable": sys.executable}
    for m in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            vers[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            vers[m] = "not importable"
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "versions": vers, "platform": platform.platform(),
            "cpu": cpu, "args": vars(args), "seed": args.seed, "n_boot": args.n_boot,
            "original_ir_environment": "python 3.13.9, numpy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2 "
                                       "(outputs/research/final/reproducibility_manifest.json)",
            "note": "No model is trained. The project .venv points to a base interpreter that does not exist on this "
                    "host; the analysis uses only numpy / pandas / scipy / scikit-learn metrics / matplotlib. The fold "
                    "partition is read from the saved test indices, so it does not depend on package versions or on the "
                    "CPU-dependent tie order of GroupKFold."}


def not_available() -> pd.DataFrame:
    return pd.DataFrame([
        ("Risk probability for LightGBM, CatBoost, FT-Transformer, LightGBM-L1", "Not Available",
         "Point-prediction regressors; no predictive distribution was saved or defined. Scores are used only as rankings."),
        ("Zero-inclusive (85,567) risk analysis on the student split", "Available in follow-up 2 (partly)",
         "Not computable from the saved predictions (the IR study stored metrics only). Run with refitted LightGBM "
         "variants and NGBoost in research/aiesc2027/run_risk.py (outputs/aiesc2027/risk_methods/); CatBoost and "
         "FT-Transformer were not refitted."),
        ("CQR on the student split", "Available in follow-up",
         "Not computable from the saved predictions; run with refitted models in the calibration-split follow-up "
         "(research/aiesc2027/run_calibration.py, outputs/aiesc2027/calibration_split/)."),
        ("Split / Mondrian conformal with an independent in-fold calibration set (student split)", "Available in follow-up",
         "Not computable from the saved predictions (post-hoc cross-fold conformal is reported here instead); run with "
         "NGBoost refitted on an 80/20 student-grouped fit/calibration split in the calibration-split follow-up."),
        ("Family-split conformal results (research/family)", "Not comparable",
         "Different split, different point model (CatBoost + course text), different population of folds; reported as "
         "context only, never pooled."),
        ("TabM", "Not Available", "Student split not completed in the IR study (budget exhausted)."),
        ("Cross-semester / prospective validation", "Not Available", "Only one complete semester (113-2) exists."),
    ], columns=["item", "status", "reason"])


def run(args):
    t0 = time.time()
    before = integrity.snapshot()
    log.info("integrity snapshot: %d protected files", len(before))

    inv = inventory.build()
    inv.to_csv(DIRS["inventory"] / "file_inventory.csv", index=False, encoding="utf-8-sig")
    (DIRS["inventory"] / "file_inventory.md").write_text(
        "**File inventory (inputs of the AIESC 2027 analysis)**\n\n" + T.to_markdown(inv.fillna("")) + "\n", encoding="utf-8")
    missing = inv[~inv["exists"]]["path"].tolist()
    log.info("inventory: %d items, missing: %s", len(inv), missing or "none")

    d = io_data.load_all()
    pd.DataFrame(d.checks).to_csv(DIRS["validation"] / "alignment_checks.csv", index=False, encoding="utf-8-sig")
    (DIRS["validation"] / "data_counts.json").write_text(json.dumps(d.counts, indent=2, default=float), encoding="utf-8")
    log.info("validation: %d checks, all PASS", len(d.checks))

    M = DIRS["metrics"]
    t1, folds = A.dataset_table(d)
    overall = A.overall_table(d)
    per_fold = A.per_fold_table(d)
    risk60 = A.risk_table(d)
    risk_off = A.risk_table(d, "official")
    ngb_prob, rel = A.ngboost_probability_table(d)
    pr = A.pr_curves(d)
    bias = A.bias_table(d)
    low = A.low_grade_detail(d)
    unc = A.uncertainty_tables(d)
    ir_check = A.check_against_ir(unc)
    fam_check = A.verify_family_t1(d)
    for name, df in {"dataset": t1, "folds": folds, "overall": overall, "per_fold": per_fold, "risk_threshold60": risk60,
                     "risk_official_threshold": risk_off, "ngboost_probability": ngb_prob, "ngboost_reliability": rel,
                     "pr_curves": pr, "grade_range_bias": bias, "low_high_grade_detail": low,
                     "ngboost_intervals": unc["intervals"], "ngboost_scores": unc["scores"], "ngboost_pit": unc["pit"],
                     "crosscheck_ir_student_T3": ir_check, "crosscheck_ir_family_T1": fam_check,
                     "not_available": not_available()}.items():
        df.to_csv(M / f"{name}.csv", index=False, encoding="utf-8-sig")
    if ir_check["abs_diff"].max() > 1e-3 or fam_check["abs_diff"].max() > 1e-3:
        raise io_data.ValidationError("recomputed NGBoost interval statistics do not reproduce the IR outputs")
    log.info("metrics written; NGBoost T3/T1 interval statistics reproduce the IR outputs (max diff %.1e)",
             max(ir_check["abs_diff"].max(), fam_check["abs_diff"].max()))

    prob60 = U.prob_below(C.RISK_THRESHOLD, d.ngb_loc, d.ngb_scale)
    bs = B.PairedClusterBootstrap(d.frame["student_id"].to_numpy(), d.y, d.preds, C.RISK_THRESHOLD,
                                  extra_scores={"ngboost_prob": prob60})
    point = bs.point()
    for k in C.MODEL_ORDER:            # bootstrap point estimates must equal the direct computations
        r = risk60[risk60.key == k].iloc[0]
        o = overall[overall.key == k].iloc[0]
        for m in ("precision", "recall", "f1", "specificity", "balanced_accuracy", "pr_auc", "roc_auc"):
            assert abs(point[k][m] - r[m]) < 1e-9, (k, m)
        assert abs(point[k]["MAE"] - o["MAE"]) < 1e-9 and abs(point[k]["RMSE"] - o["RMSE"]) < 1e-9
    log.info("bootstrap: %d replicates, seed %d, %d student clusters", args.n_boot, args.seed, bs.G)
    reps = bs.run(args.n_boot, args.seed)
    per_model, diff = B.summarise(point, reps, C.REFERENCE_MODEL, C.FAMILY_METRICS, C.LOWER_IS_BETTER, C.ALPHA,
                                  C.PRACTICAL_THRESHOLDS)
    per_model.to_csv(DIRS["stats"] / "per_model_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(DIRS["stats"] / "paired_vs_lightgbm.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(DIRS["stats"] / "bootstrap_replicates.npz",
                        **{f"{k}|{m}": v[m].to_numpy() for k, v in reps.items() for m in v.columns})

    O = DIRS["tables"]
    T.table1(t1, folds, O)
    T.table2(overall, per_model, diff, O)
    T.table3(risk60, per_model, ngb_prob, O)
    T.table4(bias, O)
    T.table5(unc, O)
    T.table_stats(diff, O)
    log.info("tables written")

    G = DIRS["figures"]
    F.fig1_flow(d.counts, G)
    F.fig2_error(per_model, G)
    F.fig3_risk(per_model, pr, float(risk60["prevalence"].iat[0]), G)
    F.fig4_bias(bias, G)
    F.fig5_calibration(unc["intervals"], rel, G)
    F.figS1_pit(unc["pit"], G)
    F.figS2_category(unc["intervals"], G)
    log.info("figures written")

    if not args.skip_text:
        from research.aiesc2027 import report_text
        report_text.write(DIRS["results"], d=d, overall=overall, risk60=risk60, risk_off=risk_off, ngb_prob=ngb_prob,
                          bias=bias, low=low, unc=unc, per_model=per_model, diff=diff, fam_check=fam_check)
        log.info("results draft written")

    from ir_pipeline.privacy import privacy_gate, raw_identifiers
    ids, names = raw_identifiers([pd.read_csv(C.RAW_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")])
    gate = privacy_gate([C.OUT], ids, names)
    (DIRS["validation"] / "privacy_scan.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("privacy gate: %s (%d findings)", gate["status"], gate["n_findings"])
    if gate["status"] != "PASS":
        raise RuntimeError("privacy gate failed")

    after = integrity.snapshot()
    cmp_ = integrity.compare(before, after)
    (DIRS["validation"] / "integrity_check.json").write_text(json.dumps(cmp_, indent=2), encoding="utf-8")
    log.info("integrity of pre-existing artefacts: %s (%d files)", cmp_["status"], cmp_["files_checked"])
    if cmp_["status"] != "PASS":
        raise RuntimeError(f"pre-existing artefacts changed: {cmp_}")
    return round(time.time() - t0, 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--skip-text", action="store_true", help="do not write the English results draft")
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
