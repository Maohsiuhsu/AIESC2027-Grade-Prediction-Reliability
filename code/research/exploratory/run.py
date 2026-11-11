"""Pre-registered exploratory study runner (see research/exploratory/prereg.json).

Usage: .venv/bin/python research/exploratory/run.py [--quick] [--output-dir DIR]
Resumable: finished (candidate, split) OOFs are kept in oof.npz and skipped on restart.
All results are EXPLORATORY: IR.csv has been used repeatedly for model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd  # noqa: I001  (pandas/pyarrow must load before torch on this machine)
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.advanced.common import SPLIT_GROUP, Registry, cross_validate, get_folds, load_frames  # noqa: E402
from research.advanced.course_repr import RepModel  # noqa: E402
from research.config import TARGET, ResearchConfig  # noqa: E402
from research.family.groups import size_bins  # noqa: E402

PREREG = Path(__file__).with_name("prereg.json")
OUT = ROOT / "outputs" / "research" / "exploratory"
ADV_OOF = ROOT / "outputs" / "research" / "advanced" / "phase1_oof.npz"
SPLITS = ["student", "section", "course", "family"]
BASELINES = ["frozen:catboost:cat", "frozen:lightgbm:cat"]


# ---- pre-registration lock ---------------------------------------------------------------------------
def check_prereg(out: Path) -> dict:
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    h = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    lock = out / "prereg.lock.json"
    if not lock.exists():
        lock.write_text(json.dumps({"prereg_sha256": h, "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))
    locked = json.loads(lock.read_text())["prereg_sha256"]
    if locked != h:
        raise SystemExit(f"prereg.json changed after it was locked ({locked[:12]} -> {h[:12]}); refusing to run")
    return pre


# ---- statistics ----------------------------------------------------------------------------------------
def paired_bootstrap(frame, base, cand, group_col, n_boot, alphas=(0.05,), seed=42) -> dict:
    y = frame[TARGET].to_numpy()
    ok = ~(np.isnan(base) | np.isnan(cand))
    ea, eb = np.abs(y[ok] - base[ok]), np.abs(y[ok] - cand[ok])
    g = frame[group_col].astype(str).to_numpy()[ok]
    _, inv = np.unique(g, return_inverse=True)
    sa, sb, cnt = np.bincount(inv, ea), np.bincount(inv, eb), np.bincount(inv).astype(float)
    n = len(cnt)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        diffs[i] = (w @ sa - w @ sb) / (w @ cnt)
    out = {"mae_diff": round(float((sa.sum() - sb.sum()) / cnt.sum()), 4), "groups": int(n)}
    for a in alphas:
        lo, hi = np.percentile(diffs, [100 * a / 2, 100 * (1 - a / 2)])
        tag = "95" if abs(a - 0.05) < 1e-12 else "adj"
        out[f"ci_low_{tag}"], out[f"ci_high_{tag}"] = round(float(lo), 4), round(float(hi), 4)
    return out


def mae_rmse(y, p):
    return round(float(np.mean(np.abs(y - p))), 4), round(float(np.sqrt(np.mean((y - p) ** 2))), 4)


# ---- stages ---------------------------------------------------------------------------------------------
def frozen_baselines(frame, cfg, reg, oofs, quick):
    reuse = {}
    if ADV_OOF.exists() and not quick:
        z = np.load(ADV_OOF)
        for split in SPLITS:
            for learner in ("catboost", "lightgbm"):
                k = f"{split}|{learner}|cat"
                if k in z.files:
                    reuse[(split, f"frozen:{learner}:cat")] = z[k]
        r = cross_validate(frame, "student", lambda tr: RepModel("catboost", (), True, cfg=cfg), cfg, reg,
                           "frozen", "catboost:cat[repro-check]")
        dev = float(np.nanmax(np.abs(r["oof"] - reuse[("student", "frozen:catboost:cat")])))
        reg.log(phase="frozen", experiment="reproducibility_check", status="success", max_abs_diff=dev, seed=cfg.seed)
        print(f"reproducibility check (CatBoost, student split): max |diff| = {dev}")
        if dev > 1e-9:
            reuse = {}
    for split in SPLITS:
        for learner in ("catboost", "lightgbm"):
            key = (split, f"frozen:{learner}:cat")
            if key in oofs:
                continue
            if key in reuse:
                oofs[key] = reuse[key]
                reg.log(phase="frozen", experiment=key[1], split=split, status="reused_identical_folds", seed=cfg.seed)
                continue
            r = cross_validate(frame, split, lambda tr, l=learner: RepModel(l, (), True, cfg=cfg), cfg, reg, "frozen", key[1])
            oofs[key] = r["oof"]


def run_candidates(frame, cfg, reg, pre, oofs, out, t_budget_start, prior_seconds):
    from research.exploratory.candidates import factories
    fac = factories(cfg)
    order = pre["budget_and_stopping"]["execution_order"]
    cap_total = pre["budget_and_stopping"]["total_wall_clock_cap_hours"] * 3600
    cap_fit = pre["budget_and_stopping"]["per_fit_cap_minutes"] * 60
    status = {}
    for name in order:
        for split in SPLITS:
            if (split, name) in oofs:
                continue
            elapsed = prior_seconds + time.time() - t_budget_start
            if elapsed > cap_total:
                reg.log(phase="candidates", experiment=name, split=split, status="skipped_total_budget", seed=cfg.seed)
                status[(name, split)] = "skipped_total_budget"
                continue
            r = cross_validate(frame, split, lambda tr, n=name, s=split: fac[n](s), cfg, reg, "candidates", name)
            if r["status"] != "success":        # one automatic retry for transient errors (pre-registered)
                r = cross_validate(frame, split, lambda tr, n=name, s=split: fac[n](s), cfg, reg, "candidates", name + "[retry]")
            if r["status"] != "success":
                status[(name, split)] = "failed"
                continue
            rec = [json.loads(l) for l in (out / "registry.jsonl").read_text(encoding="utf-8").splitlines()][-1]
            fi = rec.get("fold_info", [])
            over = any(f.get("seconds", 0) > cap_fit or f.get("budget_exceeded") for f in fi)
            status[(name, split)] = "budget_exceeded" if over else "complete"
            oofs[(split, name)] = r["oof"]
            np.savez_compressed(out / "oof.npz", **{"|".join(k): v for k, v in oofs.items()})
            print(f"  {name:26s} {split:8s} MAE {r['MAE']:.3f}  {r['seconds']:.0f}s  {status[(name, split)]}")
    return status


def analyse(frame, oofs, pre, cfg, status):
    n_cand = len(pre["candidates"])
    alpha_adj = 0.05 / (n_cand * 5)
    nb = 400 if cfg.quick else pre["endpoints"]["bootstrap"]["n_boot"]
    y = frame[TARGET].to_numpy()
    single = (size_bins(frame)["family_courses_bin"] == "1").to_numpy()
    rows, cis, groups = [], [], []
    for split in SPLITS:
        sys_ = {n: o for (s, n), o in oofs.items() if s == split}
        for n, o in sys_.items():
            mae, rmse = mae_rmse(y, o)
            rows.append({"split": split, "system": n, "population": "all", "MAE": mae, "RMSE": rmse})
            if split == "family":
                mae, rmse = mae_rmse(y[single], o[single])
                rows.append({"split": split, "system": n, "population": "single-course families", "MAE": mae, "RMSE": rmse})
            for col in ("course_category", "student_college", "program"):
                for gv, idx in frame.groupby(col).indices.items():
                    if len(idx) >= 200:
                        groups.append({"split": split, "system": n, "group_by": col, "group": gv, "n": len(idx),
                                       "MAE": round(float(np.mean(np.abs(y[idx] - o[idx]))), 4)})
        base_name = min((b for b in BASELINES if b in sys_), key=lambda b: np.mean(np.abs(y - sys_[b])))
        for n, o in sys_.items():
            if n in BASELINES:
                continue
            gcol = SPLIT_GROUP[split]
            c = paired_bootstrap(frame, sys_[base_name], o, gcol, nb, (0.05, alpha_adj), cfg.seed)
            cis.append({"split": split, "population": "all", "comparator": base_name, "candidate": n,
                        "status": status.get((n, split), "complete"), **c})
            if split == "family":
                sub = frame[single].reset_index(drop=True)
                c = paired_bootstrap(sub, sys_[base_name][single], o[single], gcol, nb, (0.05, alpha_adj), cfg.seed)
                cis.append({"split": split, "population": "single-course families", "comparator": base_name,
                            "candidate": n, "status": status.get((n, split), "complete"), **c})
    metrics, cis, groups = pd.DataFrame(rows), pd.DataFrame(cis), pd.DataFrame(groups)
    # worst group per system/split
    worst = (groups.sort_values("MAE", ascending=False).groupby(["split", "system"]).head(1)
             .rename(columns={"MAE": "worst_group_MAE"}))
    # pre-registered decision
    dec = []
    for cand in pre["candidates"]:
        c = cis[cis["candidate"] == cand]
        if not len(c):
            dec.append({"candidate": cand, "evaluated": False, "exploratory_signal": False, "reason": "not evaluated"})
            continue
        prot = c[c["population"] == "all"]
        incomplete = bool((c["status"] != "complete").any()) or len(prot) < len(SPLITS)
        n_adj = int(((prot["ci_low_adj"] > 0) & (prot["mae_diff"] >= 0.05)).sum())
        worse = bool((c["ci_high_95"] < 0).any())
        signal = (not incomplete) and n_adj >= 2 and not worse
        dec.append({"candidate": cand, "evaluated": True, "incomplete": incomplete,
                    "protocols_with_adjusted_gain": n_adj, "significantly_worse_somewhere": worse,
                    "mean_diff_4_protocols": round(float(prot["mae_diff"].mean()), 4),
                    "exploratory_signal": bool(signal)})
    return metrics, cis, groups, worst, pd.DataFrame(dec), alpha_adj


def zero_sensitivity(with_zeros, cfg, reg, dec, pre):
    from research.exploratory.candidates import factories
    fac = factories(cfg)
    d = dec[dec["evaluated"]].copy()
    best_a = d[d["candidate"].str.startswith("A")].sort_values("mean_diff_4_protocols").tail(1)["candidate"].tolist()
    best_b = d[d["candidate"].str.startswith("B")].sort_values("mean_diff_4_protocols").tail(1)["candidate"].tolist()
    systems = {"frozen:catboost:cat": lambda s: RepModel("catboost", (), True, cfg=cfg),
               "frozen:lightgbm:cat": lambda s: RepModel("lightgbm", (), True, cfg=cfg)}
    for n in ["C3_catboost_mae", "C4_catboost_huber", "C5_lightgbm_l1", "C6_lightgbm_huber"] + best_a + best_b:
        systems[n] = fac[n]
    y = with_zeros[TARGET].to_numpy()
    z = y == 0
    rows = []
    for split in ("student", "family"):
        for n, f in systems.items():
            r = cross_validate(with_zeros, split, lambda tr, f=f, s=split: f(s), cfg, reg, "zeros", n)
            if r["status"] != "success":
                continue
            o = r["oof"]
            rows.append({"split": split, "system": n, "n": len(y), "zeros": int(z.sum()), "MAE_all": round(r["MAE"], 4),
                         "MAE_nonzero": round(float(np.mean(np.abs(y[~z] - o[~z]))), 4),
                         "MAE_zero": round(float(np.mean(np.abs(y[z] - o[z]))), 4), "seconds": r["seconds"]})
    return pd.DataFrame(rows)


def costs(out: Path) -> pd.DataFrame:
    recs = [json.loads(l) for l in (out / "registry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame([r for r in recs if r.get("status") in ("success", "failed")])
    if df.empty:
        return df
    df["max_fold_seconds"] = df["fold_info"].map(lambda fi: max((f.get("seconds", 0) for f in fi), default=0)
                                                 if isinstance(fi, list) else 0)
    return (df.groupby(["phase", "experiment"]).agg(runs=("split", "size"), total_seconds=("seconds", "sum"),
                                                     max_fold_seconds=("max_fold_seconds", "max")).reset_index())


def main(argv=None):
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=OUT)
    a = p.parse_args(argv)
    OUT = a.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    pre = check_prereg(OUT) if not a.quick else json.loads(PREREG.read_text(encoding="utf-8"))
    cfg = ResearchConfig(quick=a.quick)
    reg = Registry(OUT / "registry.jsonl")
    prior = sum(r.get("seconds", 0) or 0 for r in map(json.loads, (OUT / "registry.jsonl").read_text().splitlines())
                if r.get("experiment") == "session_total") if (OUT / "registry.jsonl").exists() else 0.0
    t0 = time.time()
    reg.log(phase="start", experiment="session_start", status="done", seed=cfg.seed,
            prereg_sha256=hashlib.sha256(PREREG.read_bytes()).hexdigest(), quick=a.quick, budget_used_before_s=prior)
    frame, with_zeros, _ = load_frames(cfg)
    oofs = {}
    if (OUT / "oof.npz").exists():
        z = np.load(OUT / "oof.npz")
        oofs = {tuple(k.split("|")): z[k] for k in z.files}
    frozen_baselines(frame, cfg, reg, oofs, a.quick)
    np.savez_compressed(OUT / "oof.npz", **{"|".join(k): v for k, v in oofs.items()})
    status = run_candidates(frame, cfg, reg, pre, oofs, OUT, t0, prior)
    (OUT / "run_status.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in status.items()}, indent=2))
    metrics, cis, groups, worst, dec, alpha_adj = analyse(frame, oofs, pre, cfg, status)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    cis.to_csv(OUT / "contrasts.csv", index=False)
    groups.to_csv(OUT / "groups.csv", index=False, encoding="utf-8-sig")
    worst.to_csv(OUT / "worst_groups.csv", index=False, encoding="utf-8-sig")
    dec.to_csv(OUT / "decision.csv", index=False)
    zs = zero_sensitivity(with_zeros, cfg, reg, dec, pre)
    zs.to_csv(OUT / "zero_sensitivity.csv", index=False)
    reg.log(phase="end", experiment="session_total", status="done", seed=cfg.seed, seconds=round(time.time() - t0, 1))
    costs(OUT).to_csv(OUT / "costs.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "alpha_adjusted": alpha_adj, "any_exploratory_signal": bool(dec["exploratory_signal"].any()),
        "signals": dec.loc[dec["exploratory_signal"], "candidate"].tolist(),
        "session_seconds": round(time.time() - t0, 1), "budget_used_s": round(prior + time.time() - t0, 1)},
        indent=2), encoding="utf-8")
    from research.exploratory.report_zh import write_report
    write_report(OUT)
    from ir_pipeline.privacy import privacy_gate, raw_identifiers
    raw = ROOT / "IR.csv"
    ids, names = raw_identifiers([pd.read_csv(raw, dtype=str, keep_default_na=False, encoding="utf-8-sig")]) \
        if raw.exists() else (set(), set())
    gate = privacy_gate([OUT], ids, names, exclude=[ROOT / "private"])
    (OUT / "privacy_scan.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print("signals:", dec.loc[dec["exploratory_signal"], "candidate"].tolist(), "| privacy:", gate["status"])
    if gate["status"] == "FAIL":
        raise SystemExit("PRIVACY FAIL")


if __name__ == "__main__":
    main()
