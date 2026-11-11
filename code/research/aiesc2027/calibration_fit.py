"""Model fits of the calibration-split follow-up (see calibration_split_spec.json). One outer fold per call.

Usage (pinned environment, from the project root):
  VENV=/home/A6000/hdd2/zhe/IR/.venv-aiesc2027/bin/python
  $VENV -B research/aiesc2027/calibration_fit.py --fold 0 [--quick]

Per outer fold k (saved student-split test indices; training fold = all other rows):
  1. reproduction gate: original NGBoost (Amendment 1 NGBU) and frozen LightGBM refitted on the FULL training
     fold, compared with the saved OOF predictions;
  2. student-grouped 80/20 proper-training / calibration partition of the training fold;
  3. NGBoost, LightGBM-L2 and 8 LightGBM quantile models fitted on the proper-training part only;
     predictions for the calibration rows and the test rows.
Writes outputs/aiesc2027/calibration_split/jobs/fold{k}.npz (+ .json). No student identifier is written.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd  # noqa: I001
import numpy as np
from scipy.stats import norm

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.advanced.common import get_folds, load_frames  # noqa: E402
from research.aiesc2027 import config as C  # noqa: E402
from research.config import TARGET, ResearchConfig  # noqa: E402
from research.exploratory.amendment1.worker import NGBU  # noqa: E402
from research.exploratory.candidates import GBM  # noqa: E402

JOBS = C.OUT / "calibration_split" / "jobs"
CALIB_FRACTION = 0.2
QUANTILES = (0.025, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.975)


def calibration_partition(students: np.ndarray, fold: int, fraction: float = CALIB_FRACTION) -> np.ndarray:
    """Boolean mask over the training rows: True = calibration row (whole students only)."""
    u = np.unique(students)
    perm = np.random.default_rng(C.SEED + fold).permutation(len(u))
    cal_students = u[perm[: int(round(fraction * len(u)))]]
    return np.isin(students, cal_students)


def _saved(fold: int) -> dict:
    z = np.load(C.AMENDMENT_JOBS / f"M3_ngboost__{C.AMENDMENT_UNIT}__f{fold}.npz")
    info = json.loads((C.AMENDMENT_JOBS / f"M3_ngboost__{C.AMENDMENT_UNIT}__f{fold}.json").read_text())
    return {"test_idx": z["test_idx"], "loc": z["loc"], "scale": z["scale"], "mult": info["scale_multiplier"],
            "lgb_oof": np.load(C.EXPLORATORY_OOF)[C.MODELS["lightgbm"]["key"]]}


def run_fold(fold: int, quick: bool = False) -> dict:
    t0 = time.time()
    cfg = ResearchConfig(quick=quick)
    frame, _, _ = load_frames(ResearchConfig())
    y = frame[TARGET].to_numpy(float)
    saved = _saved(fold)
    te = saved["test_idx"]
    tr = np.setdiff1d(np.arange(len(frame)), te)
    st = frame["student_id"].astype(str)
    if set(st.iloc[tr]) & set(st.iloc[te]):
        raise RuntimeError(f"fold {fold}: saved test fold shares students with its training fold")
    # GroupKFold orders equal-sized groups with numpy's unstable argsort, whose tie order depends on the
    # CPU's SIMD sort kernel; the saved partition is therefore authoritative and only compared here.
    recomputed = get_folds(frame, "student", ResearchConfig())[fold][1]
    info = {"fold": fold, "quick": quick, "n_train": int(len(tr)), "n_test": int(len(te)),
            "recomputed_test_idx_equals_saved": bool(np.array_equal(np.sort(recomputed), np.sort(te)))}
    n_est = 40 if quick else 400
    out = {"test_idx": te}
    times = {}

    # 1. reproduction gate (full training fold, exactly as in the IR study)
    s = time.time()
    m = NGBU("student", C.SEED, n_est).fit(frame.iloc[tr], y[tr])
    loc, scale = m._dist(frame.iloc[te])
    out["repro_ngb_loc"], out["repro_ngb_scale"] = loc, scale

    def summary(mu, sd):
        yt = y[te]
        return {"MAE": float(np.mean(np.abs(np.clip(mu, 0, 100) - yt))),
                "NLL": float(-np.mean(norm.logpdf(yt, mu, sd))),
                "coverage90": float(np.mean(np.abs(yt - mu) <= norm.ppf(0.95) * sd))}

    info["repro_ngboost"] = {"refit": summary(loc, scale), "saved": summary(saved["loc"], saved["scale"]),
                             "max_abs_diff_loc": float(np.max(np.abs(loc - saved["loc"]))),
                             "mean_abs_diff_loc": float(np.mean(np.abs(loc - saved["loc"]))),
                             "max_abs_diff_scale": float(np.max(np.abs(scale - saved["scale"]))),
                             "scale_multiplier": round(m.scale_mult_, 4), "saved_scale_multiplier": saved["mult"],
                             "best_iteration": m.fit_info_.get("best_iteration")}
    times["repro_ngboost"] = round(time.time() - s, 1)
    s = time.time()
    g = GBM("lightgbm", None, cfg).fit(frame.iloc[tr], y[tr])
    p = np.clip(g.predict(frame.iloc[te]), 0, 100)
    out["repro_lgb"] = p
    info["repro_lightgbm"] = {"max_abs_diff": float(np.max(np.abs(p - saved["lgb_oof"][te]))),
                              "mae_test_fold": float(np.mean(np.abs(p - y[te]))),
                              "mae_test_fold_saved": float(np.mean(np.abs(saved["lgb_oof"][te] - y[te])))}
    times["repro_lightgbm"] = round(time.time() - s, 1)

    # 2. calibration partition of the training fold (whole students)
    cal_mask = calibration_partition(st.to_numpy()[tr], fold)
    fit_idx, cal_idx = tr[~cal_mask], tr[cal_mask]
    out["fit_idx"], out["cal_idx"] = fit_idx, cal_idx
    info["partition"] = {"fit_rows": int(len(fit_idx)), "cal_rows": int(len(cal_idx)),
                         "fit_students": int(st.iloc[fit_idx].nunique()), "cal_students": int(st.iloc[cal_idx].nunique()),
                         "students_shared_fit_cal": int(len(set(st.iloc[fit_idx]) & set(st.iloc[cal_idx]))),
                         "students_shared_cal_test": int(len(set(st.iloc[cal_idx]) & set(st.iloc[te])))}
    if info["partition"]["students_shared_fit_cal"] or info["partition"]["students_shared_cal_test"]:
        raise RuntimeError("calibration partition is not student-disjoint")
    Xf, yf = frame.iloc[fit_idx], y[fit_idx]
    Xc, Xt = frame.iloc[cal_idx], frame.iloc[te]

    # 3. models on the proper-training part only
    s = time.time()
    m = NGBU("student", C.SEED, n_est).fit(Xf, yf)
    out["ngb_loc_cal"], out["ngb_scale_cal"] = m._dist(Xc)
    out["ngb_loc_te"], out["ngb_scale_te"] = m._dist(Xt)
    info["ngboost_fit"] = {"scale_multiplier": round(m.scale_mult_, 4), "best_iteration": m.fit_info_.get("best_iteration"),
                           "inner_val_rows": m.fit_info_.get("val_rows")}
    times["ngboost_fit"] = round(time.time() - s, 1)
    s = time.time()
    g = GBM("lightgbm", None, cfg).fit(Xf, yf)
    out["lgb_cal"], out["lgb_te"] = g.predict(Xc), g.predict(Xt)
    times["lightgbm_fit"] = round(time.time() - s, 1)
    for q in QUANTILES:
        s = time.time()
        g = GBM("lightgbm", {"objective": "quantile", "alpha": q}, cfg).fit(Xf, yf)
        out[f"q{q}_cal"], out[f"q{q}_te"] = g.predict(Xc), g.predict(Xt)
        times[f"lightgbm_q{q}"] = round(time.time() - s, 1)

    JOBS.mkdir(parents=True, exist_ok=True)
    stem = JOBS / f"fold{fold}{'_quick' if quick else ''}"
    np.savez_compressed(f"{stem}.npz", **out)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    info.update({"seconds": round(time.time() - t0, 1), "step_seconds": times,
                 "max_rss_mb": round(ru.ru_maxrss / 1024, 1), "quantiles": list(QUANTILES),
                 "calibration_fraction": CALIB_FRACTION})
    Path(f"{stem}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--quick", action="store_true", help="40 NGBoost / 60 LightGBM iterations (smoke test only)")
    a = p.parse_args(argv)
    print(json.dumps(run_fold(a.fold, a.quick), indent=2))


if __name__ == "__main__":
    main()
