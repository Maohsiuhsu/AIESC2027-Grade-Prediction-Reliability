"""Model fits of follow-up 2 (design: risk_methods_spec.json). One outer fold and one population per call.

Usage (pinned environment, from the project root; run folds one after another, not in parallel):
  VENV=/home/A6000/hdd2/zhe/IR/.venv-aiesc2027/bin/python
  $VENV -B research/aiesc2027/risk_fit.py --population A --fold 0     # needs calibration_split/jobs/fold0.npz
  $VENV -B research/aiesc2027/risk_fit.py --population B --fold 0

Population A (83,494 rows): reuses the follow-up-1 fits (LightGBM-L2, LightGBM q0.25, NGBoost on the same 80% part)
and adds LightGBM-L1, the LightGBM classifier and the class-weighted LightGBM.
Population B (85,567 rows, zero grades included): every model is fitted on the zero-inclusive frame with the
student roles of population A (see risk_methods_spec.json).
Writes outputs/aiesc2027/risk_methods/jobs/{A,B}_fold{k}.npz (+ .json); row indices refer to the population's frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd  # noqa: I001
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lightgbm import LGBMRegressor  # noqa: E402

from research.advanced.common import load_frames  # noqa: E402
from research.aiesc2027 import config as C  # noqa: E402
from research.config import TARGET, ResearchConfig  # noqa: E402
from research.exploratory.amendment1.worker import NGBU  # noqa: E402
from research.exploratory.candidates import BASE_CAT, BASE_NUM, GBM  # noqa: E402
from research.models import LightGBMFrame  # noqa: E402

CAL_JOBS = C.OUT / "calibration_split" / "jobs"
JOBS = C.OUT / "risk_methods" / "jobs"


class WeightedLightGBM(LightGBMFrame):
    """The frozen LightGBM (research.models.LightGBMFrame) with per-row sample weights."""

    def fit(self, X, y, sample_weight=None):
        self.categories_ = {c: sorted(X[c].astype("string").fillna("__MISSING__").astype(str).unique())
                            for c in self.categorical}
        self.model_ = LGBMRegressor(random_state=self.seed, **self.params)
        self.model_.fit(self._prep(X), np.asarray(y, float), sample_weight=sample_weight,
                        categorical_feature=self.categorical)
        return self


def class_weights(y: np.ndarray) -> tuple[np.ndarray, float]:
    low = y < C.RISK_THRESHOLD
    p = float(low.mean())
    return np.where(low, (1 - p) / p, 1.0), (1 - p) / p


def zero_only_assignment(students_b: np.ndarray, students_a: set) -> pd.DataFrame:
    """Test fold and calibration role of the students that exist only in population B."""
    only = np.array(sorted(set(students_b) - students_a))
    rng = np.random.default_rng(C.SEED)
    return pd.DataFrame({"student": only, "fold": rng.integers(0, C.EXPECTED_N_FOLDS, len(only)),
                         "cal": rng.random(len(only)) < 0.2})


def roles_B(frame_a: pd.DataFrame, frame_b: pd.DataFrame, fold: int) -> dict:
    sa = frame_a["student_id"].astype(str).to_numpy()
    sb = frame_b["student_id"].astype(str).to_numpy()
    cj = np.load(CAL_JOBS / f"fold{fold}.npz")
    stud_fold = {}
    for f in range(C.EXPECTED_N_FOLDS):
        te = np.load(C.AMENDMENT_JOBS / f"M3_ngboost__{C.AMENDMENT_UNIT}__f{f}.npz")["test_idx"]
        stud_fold.update(dict.fromkeys(sa[te], f))
    cal_students = set(sa[cj["cal_idx"]])
    extra = zero_only_assignment(sb, set(sa))
    stud_fold.update(dict(zip(extra["student"], extra["fold"].astype(int))))
    cal_students |= set(extra.loc[extra["cal"], "student"])
    f_b = np.array([stud_fold[s] for s in sb])
    is_cal = np.isin(sb, list(cal_students))
    te = np.flatnonzero(f_b == fold)
    return {"test_idx": te, "cal_idx": np.flatnonzero((f_b != fold) & is_cal),
            "fit_idx": np.flatnonzero((f_b != fold) & ~is_cal), "n_zero_only_students": int(len(extra))}


def run(population: str, fold: int) -> dict:
    t0 = time.time()
    cfg = ResearchConfig()
    frame_a, frame_b, _ = load_frames(cfg)
    frame = frame_a if population == "A" else frame_b
    y = frame[TARGET].to_numpy(float)
    times, out = {}, {}
    if population == "A":
        cj = dict(np.load(CAL_JOBS / f"fold{fold}.npz"))
        idx = {k: cj[k] for k in ("test_idx", "cal_idx", "fit_idx")}
        for part in ("cal", "te"):
            out[f"lgb_{part}"] = cj[f"lgb_{part}"]
            out[f"q25_{part}"] = cj[f"q0.25_{part}"]
            out[f"ngb_loc_{part}"], out[f"ngb_scale_{part}"] = cj[f"ngb_loc_{part}"], cj[f"ngb_scale_{part}"]
        info = {"reused_from": str(CAL_JOBS / f"fold{fold}.npz")}
    else:
        idx = roles_B(frame_a, frame_b, fold)
        info = {"n_zero_only_students": idx.pop("n_zero_only_students")}
    fit, cal, te = idx["fit_idx"], idx["cal_idx"], idx["test_idx"]
    st = frame["student_id"].astype(str)
    shared = (set(st.iloc[fit]) & set(st.iloc[cal])) | (set(st.iloc[cal]) & set(st.iloc[te])) | (set(st.iloc[fit]) & set(st.iloc[te]))
    if shared:
        raise RuntimeError("fit / calibration / test are not student-disjoint")
    Xf, yf, Xc, Xt = frame.iloc[fit], y[fit], frame.iloc[cal], frame.iloc[te]
    out.update(idx)

    def timed(name, fn):
        s = time.time()
        fn()
        times[name] = round(time.time() - s, 1)

    def fit_gbm(key, params, target):
        g = GBM("lightgbm", params, cfg).fit(Xf, target)
        out[f"{key}_cal"], out[f"{key}_te"] = g.predict(Xc), g.predict(Xt)

    def fit_weighted():
        w, w_low = class_weights(yf)
        cols = list(dict.fromkeys(BASE_CAT + BASE_NUM + ["course_key"]))
        m = WeightedLightGBM(BASE_CAT, BASE_NUM, cfg.lightgbm_params, cfg.seed).fit(Xf[cols].reset_index(drop=True), yf, w)
        out["weighted_cal"] = m.predict(Xc[cols].reset_index(drop=True))
        out["weighted_te"] = m.predict(Xt[cols].reset_index(drop=True))
        info["weight_low_grade_rows"] = round(w_low, 4)

    def fit_ngb():
        m = NGBU("student", C.SEED, 400).fit(Xf, yf)
        out["ngb_loc_cal"], out["ngb_scale_cal"] = m._dist(Xc)
        out["ngb_loc_te"], out["ngb_scale_te"] = m._dist(Xt)
        info["ngboost_best_iteration"] = m.fit_info_.get("best_iteration")

    if population == "B":
        timed("lightgbm_l2", lambda: fit_gbm("lgb", None, yf))
        timed("lightgbm_q25", lambda: fit_gbm("q25", {"objective": "quantile", "alpha": 0.25}, yf))
        timed("ngboost", fit_ngb)
    timed("lightgbm_l1", lambda: fit_gbm("l1", {"objective": "l1"}, yf))
    timed("classifier", lambda: fit_gbm("clf", {"objective": "binary"}, (yf < C.RISK_THRESHOLD).astype(float)))
    timed("weighted", fit_weighted)

    JOBS.mkdir(parents=True, exist_ok=True)
    stem = JOBS / f"{population}_fold{fold}"
    np.savez_compressed(f"{stem}.npz", **out)
    info.update({"population": population, "fold": fold, "n_rows_frame": int(len(frame)), "fit_rows": int(len(fit)),
                 "cal_rows": int(len(cal)), "test_rows": int(len(te)), "zero_rows_test": int((y[te] == 0).sum()),
                 "zero_rows_cal": int((y[cal] == 0).sum()), "zero_rows_fit": int((y[fit] == 0).sum()),
                 "step_seconds": times, "seconds": round(time.time() - t0, 1)})
    Path(f"{stem}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", choices=["A", "B"], required=True)
    p.add_argument("--fold", type=int, required=True)
    a = p.parse_args(argv)
    print(json.dumps(run(a.population, a.fold), indent=2))


if __name__ == "__main__":
    main()
