"""Amendment 1 runner: feasibility folds, budgeted full evaluation, analysis, report.

Every fit runs in its own subprocess (own session / process group) and is killed with SIGKILL
when its timeout expires. Timeout = min(per-fit limit, model share left, global budget left).

Usage: .venv/bin/python research/exploratory/amendment1/run.py [--quick] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd  # noqa: I001
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.advanced.common import SPLIT_GROUP, Registry, cross_validate, get_folds, load_frames  # noqa: E402
from research.advanced.course_repr import RepModel  # noqa: E402
from research.config import TARGET, ResearchConfig  # noqa: E402
from research.exploratory.run import paired_bootstrap  # noqa: E402
from research.family.groups import size_bins  # noqa: E402

HERE = Path(__file__).resolve().parent
AMEND = HERE / "amendment.json"
ORIG_PREREG = ROOT / "research" / "exploratory" / "prereg.json"
ORIG_OUT = ROOT / "outputs" / "research" / "exploratory"
OUT = ROOT / "outputs" / "research" / "exploratory_amendment1"
MODELS = ["M1_tabm", "M2_ft_transformer", "M3_ngboost"]
UNITS = [("T1", "family", "primary"), ("T2", "family", "zeros"), ("T3", "student", "primary"),
         ("T4", "section", "primary"), ("T5", "course", "primary")]
LEVELS = (0.5, 0.8, 0.9, 0.95)


def check_locks(out: Path, quick: bool) -> dict:
    a = json.loads(AMEND.read_text(encoding="utf-8"))
    if quick:
        return a
    lock = json.loads((out / "amendment.lock.json").read_text(encoding="utf-8"))
    h = hashlib.sha256(AMEND.read_bytes()).hexdigest()
    if h != lock["amendment_sha256"]:
        raise SystemExit("amendment.json changed after it was locked; refusing to run")
    orig = json.loads((ORIG_OUT / "prereg.lock.json").read_text(encoding="utf-8"))["prereg_sha256"]
    if hashlib.sha256(ORIG_PREREG.read_bytes()).hexdigest() != orig:
        raise SystemExit("the original pre-registration was modified; refusing to run")
    return a


def run_job(job: dict, timeout_s: float, python=sys.executable) -> dict:
    """Run one fit in a subprocess; SIGKILL the whole process group at the timeout."""
    jp = Path(job["out"] + ".job.json")
    jp.write_text(json.dumps(job), encoding="utf-8")
    env = {**os.environ, "LD_LIBRARY_PATH": str(Path.home() / "anaconda3" / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    t0 = time.time()
    proc = subprocess.Popen([python, str(HERE / "worker.py"), str(jp)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True, env=env)
    try:
        _, err = proc.communicate(timeout=max(1.0, timeout_s))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return {"status": "timeout", "seconds": round(time.time() - t0, 1), "timeout_s": round(timeout_s, 1)}
    if proc.returncode != 0 or not Path(job["out"] + ".npz").exists():
        return {"status": "failed", "seconds": round(time.time() - t0, 1),
                "error": err.decode("utf-8", "ignore")[-800:], "returncode": proc.returncode}
    info = json.loads(Path(job["out"] + ".json").read_text(encoding="utf-8"))
    return {"status": "complete", **info, "wall_seconds": round(time.time() - t0, 1)}


def execute(a, frames_paths, cfg, reg, out: Path, quick: bool) -> pd.DataFrame:
    b = a["budget"]
    t_start = time.time()
    g_deadline = t_start + (0.5 if quick else b["global_wall_clock_hours"]) * 3600   # quick = smoke test only
    share = (8.0 if quick else b["per_model_share_minutes"]) * 60
    per_fit = (2.0 if quick else b["per_fit_limit_minutes"]) * 60
    n_folds = cfg.n_folds
    used = {m: 0.0 for m in MODELS}
    dead = {m: None for m in MODELS}
    rows = []
    jobdir = out / "jobs"
    jobdir.mkdir(parents=True, exist_ok=True)

    def attempt(model, unit, split, frame_kind, fold):
        if dead[model]:
            rows.append({"model": model, "unit": unit, "fold": fold, "status": "not_started", "reason": dead[model]})
            return False
        remaining_g, remaining_m = g_deadline - time.time(), share - used[model]
        if remaining_g <= 5 or remaining_m <= 5:
            dead[model] = "model share exhausted" if remaining_m <= 5 else "global budget exhausted"
            rows.append({"model": model, "unit": unit, "fold": fold, "status": "not_started", "reason": dead[model]})
            return False
        timeout = min(per_fit, remaining_m, remaining_g)
        job = {"model": model, "split": split, "fold": fold, "seed": 42, "quick": quick,
               "frame_path": str(frames_paths[frame_kind]), "out": str(jobdir / f"{model}__{unit}__f{fold}")}
        r = run_job(job, timeout)
        if r["status"] == "failed":                    # one retry for non-timeout failures (pre-registered)
            r = run_job(job, min(per_fit, share - used[model] - r["seconds"], g_deadline - time.time()))
            r["retried"] = True
        used[model] += r.get("wall_seconds", r.get("seconds", 0))
        cause = ("per-fit limit" if timeout >= per_fit - 1 else "model share / global budget limit")
        rows.append({"model": model, "unit": unit, "split": split, "frame": frame_kind, "fold": fold,
                     **{k: v for k, v in r.items() if k != "error"}, "error": r.get("error"),
                     "timeout_cause": cause if r["status"] == "timeout" else None})
        reg.log(phase="amendment1", experiment=model, unit=unit, fold=fold, split=split, seed=42, **{k: v for k, v in r.items()})
        print(f"  {model:18s} {unit} fold {fold}: {r['status']:9s} {r.get('wall_seconds', r.get('seconds'))}s "
              f"(model used {used[model] / 60:.1f}/{share / 60:.0f} min)")
        if r["status"] != "complete":
            dead[model] = f"{r['status']} at {unit} fold {fold} ({cause if r['status'] == 'timeout' else 'error'})"
            return False
        return True

    print("feasibility folds")
    for m in MODELS:
        attempt(m, "T1", "family", "primary", 0)
    for m in MODELS:
        for unit, split, kind in UNITS:
            for fold in range(n_folds):
                if unit == "T1" and fold == 0:
                    continue
                attempt(m, unit, split, kind, fold)
    return pd.DataFrame(rows)


def collect(status: pd.DataFrame, frames: dict, out: Path, n_folds: int = 5) -> dict:
    """OOF arrays per (model, unit) for units whose folds ALL completed."""
    res = {}
    for (m, unit), g in status[status["status"] == "complete"].groupby(["model", "unit"]):
        if g["fold"].nunique() < n_folds:
            continue
        kind = g["frame"].iat[0]
        n = len(frames[kind])
        pred, loc, scale = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for f in range(n_folds):
            z = np.load(out / "jobs" / f"{m}__{unit}__f{f}.npz")
            pred[z["test_idx"]] = z["pred"]
            if "loc" in z.files:
                loc[z["test_idx"]], scale[z["test_idx"]] = z["loc"], z["scale"]
        mult = g.set_index("fold")["scale_multiplier"].to_dict() if "scale_multiplier" in g else {}
        res[(m, unit)] = {"pred": pred, "loc": loc, "scale": scale, "mult": mult}
    return res


def uncertainty(frame, folds, r) -> list[dict]:
    from scipy.stats import norm
    y = frame[TARGET].to_numpy()
    rows = []
    mult = np.full(len(y), np.nan)
    for f, (_, te) in enumerate(folds):
        mult[te] = r["mult"].get(f, np.nan)
    single = (size_bins(frame)["family_courses_bin"] == "1").to_numpy()
    for version, sc in (("raw", r["scale"]), ("rescaled_in_fold", r["scale"] * mult)):
        for pop, m in (("all", np.ones(len(y), bool)), ("single-course families", single)):
            nll = float(-np.mean(norm.logpdf(y[m], r["loc"][m], sc[m])))
            rec = {"version": version, "population": pop, "n": int(m.sum()), "NLL": round(nll, 4)}
            errs = []
            for lv in LEVELS:
                z = norm.ppf(0.5 + lv / 2)
                lo, hi = np.clip(r["loc"][m] - z * sc[m], 0, 100), np.clip(r["loc"][m] + z * sc[m], 0, 100)
                cov = float(np.mean((y[m] >= lo) & (y[m] <= hi)))
                rec[f"coverage_{int(lv * 100)}"] = round(cov, 4)
                rec[f"width_{int(lv * 100)}"] = round(float(np.mean(hi - lo)), 3)
                errs.append(abs(cov - lv))
            rec["mean_abs_coverage_error"] = round(float(np.mean(errs)), 4)
            rows.append(rec)
    return rows


def analyse(a, status, frames, cfg, reg, out, quick):
    res = collect(status, frames, out, cfg.n_folds)
    oof_orig = np.load(ORIG_OUT / "oof.npz") if (ORIG_OUT / "oof.npz").exists() and not quick else None
    nb = 400 if quick else 5000
    alpha_adj = 0.05 / 15
    rows, cis, unc = [], [], []
    base_cache = {}
    for unit, split, kind in UNITS:
        frame = frames[kind]
        y = frame[TARGET].to_numpy()
        # frozen baselines on identical data and folds
        bases = {}
        for learner in ("catboost", "lightgbm"):
            key = f"{split}|frozen:{learner}:cat"
            if kind == "primary" and oof_orig is not None and key in oof_orig.files:
                bases[f"frozen:{learner}:cat"] = oof_orig[key]
            else:
                ck = (kind, split, learner)
                if ck not in base_cache:
                    r = cross_validate(frame, split, lambda tr, l=learner: RepModel(l, (), True, cfg=cfg), cfg, reg,
                                       "baselines", f"frozen:{learner}:cat", extra={"frame": kind})
                    base_cache[ck] = r["oof"]
                bases[f"frozen:{learner}:cat"] = base_cache[ck]
        comp = min(bases, key=lambda b: np.mean(np.abs(y - bases[b])))
        systems = {**bases, **{m: res[(m, unit)]["pred"] for m in MODELS if (m, unit) in res}}
        pops = {"all": np.ones(len(y), bool)}
        if split == "family":
            pops["single-course families"] = (size_bins(frame)["family_courses_bin"] == "1").to_numpy()
        if kind == "zeros":
            pops["zero-grade rows"] = y == 0
            pops["non-zero rows"] = y != 0
        for name, o in systems.items():
            for pop, msk in pops.items():
                rows.append({"unit": unit, "split": split, "frame": kind, "system": name, "population": pop,
                             "n": int(msk.sum()), "MAE": round(float(np.mean(np.abs(y[msk] - o[msk]))), 4),
                             "RMSE": round(float(np.sqrt(np.mean((y[msk] - o[msk]) ** 2))), 4)})
                if name in MODELS and pop in ("all", "single-course families"):
                    sub = frame[msk].reset_index(drop=True)
                    c = paired_bootstrap(sub, bases[comp][msk], o[msk], SPLIT_GROUP[split], nb, (0.05, alpha_adj), cfg.seed)
                    cis.append({"unit": unit, "split": split, "frame": kind, "population": pop, "comparator": comp,
                                "candidate": name, **c})
        if ("M3_ngboost", unit) in res:
            for u in uncertainty(frame, get_folds(frame, split, cfg), res[("M3_ngboost", unit)]):
                unc.append({"unit": unit, "split": split, "frame": kind, **u})
    return pd.DataFrame(rows), pd.DataFrame(cis), pd.DataFrame(unc)


def main(argv=None):
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=OUT)
    args = p.parse_args(argv)
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    a = check_locks(OUT, args.quick)
    cfg = ResearchConfig(quick=args.quick)
    reg = Registry(OUT / "registry.jsonl")
    t0 = time.time()
    reg.log(phase="start", experiment="amendment1", status="done", seed=42,
            amendment_sha256=hashlib.sha256(AMEND.read_bytes()).hexdigest(), quick=args.quick)
    primary, with_zeros, _ = load_frames(cfg)
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    paths = {"primary": cache / "frame_primary.pkl", "zeros": cache / "frame_zeros.pkl"}
    primary.to_pickle(paths["primary"])
    with_zeros.to_pickle(paths["zeros"])
    status = execute(a, paths, cfg, reg, OUT, args.quick)
    status.to_csv(OUT / "fit_status.csv", index=False)
    metrics, cis, unc = analyse(a, status, {"primary": primary, "zeros": with_zeros}, cfg, reg, OUT, args.quick)
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    cis.to_csv(OUT / "contrasts.csv", index=False, encoding="utf-8-sig")
    unc.to_csv(OUT / "ngboost_uncertainty.csv", index=False, encoding="utf-8-sig")
    for f in paths.values():                         # remove the temporary data copies
        f.unlink(missing_ok=True)
    for f in (OUT / "jobs").glob("*.job.json"):
        f.unlink()
    total = round(time.time() - t0, 1)
    reg.log(phase="end", experiment="amendment1", status="done", seed=42, seconds=total)
    (OUT / "summary.json").write_text(json.dumps({"seconds_total": total,
                                                  "budget_hours": a["budget"]["global_wall_clock_hours"],
                                                  "quick": args.quick}, indent=2), encoding="utf-8")
    from research.exploratory.amendment1.report_zh import write_report
    write_report(OUT)
    from ir_pipeline.privacy import privacy_gate, raw_identifiers
    raw = ROOT / "IR.csv"
    ids, names = raw_identifiers([pd.read_csv(raw, dtype=str, keep_default_na=False, encoding="utf-8-sig")]) \
        if raw.exists() else (set(), set())
    gate = privacy_gate([OUT], ids, names, exclude=[ROOT / "private"])
    (OUT / "privacy_scan.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done in", total, "s | privacy:", gate["status"])
    if gate["status"] == "FAIL":
        raise SystemExit("PRIVACY FAIL")


if __name__ == "__main__":
    main()
