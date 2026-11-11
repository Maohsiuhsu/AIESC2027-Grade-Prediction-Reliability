"""One fit in an isolated subprocess (Amendment 1). Killed by the parent at its timeout.

Usage (by run.py only): python worker.py <job.json>
Writes <out>.npz (test indices, predictions [, NGBoost loc/scale]) and <out>.json (info).
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd  # noqa: I001  (pandas/pyarrow before torch on this machine)
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.advanced.common import SPLIT_GROUP, get_folds  # noqa: E402
from research.config import TARGET, ResearchConfig  # noqa: E402


class NGBU:
    """NGBoost (the original study's NGB class) + inner-validation scale multiplier for calibration."""

    def __init__(self, split, seed, n_estimators):
        from research.exploratory.candidates import NGB
        self.m = NGB(split, seed, n_estimators)
        self.split, self.seed = split, seed

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, float)
        self.m.fit(X, y)
        # the same 10% grouped validation split NGB used for early stopping (training fold only)
        g = X[SPLIT_GROUP[self.split]].astype(str).to_numpy()
        ug = np.unique(g)
        va = np.isin(g, np.random.default_rng(self.seed).choice(ug, size=max(1, len(ug) // 10), replace=False))
        loc, scale = self._dist(X[va])
        self.scale_mult_ = float(np.sqrt(np.mean(((y[va] - loc) / scale) ** 2)))
        self.fit_info_ = {**self.m.fit_info_, "scale_multiplier": round(self.scale_mult_, 4), "val_rows": int(va.sum())}
        return self

    def _dist(self, X):
        X = X.reset_index(drop=True)
        d = self.m.model_.pred_dist(self.m._matrix(X, self.m.enc_.transform(X)))
        return np.asarray(d.loc, float), np.asarray(d.scale, float)

    def predict(self, X):
        return self._dist(X)[0]


def build(model: str, split: str, seed: int, quick: bool):
    from research.exploratory.nn import OfficialNN
    if model == "M1_tabm":
        return OfficialNN("tabm", SPLIT_GROUP[split], seed, 2 if quick else 30, time_cap_s=float("inf"))
    if model == "M2_ft_transformer":
        return OfficialNN("ft", SPLIT_GROUP[split], seed, 2 if quick else 30, time_cap_s=float("inf"))
    if model == "M3_ngboost":
        return NGBU(split, seed, 40 if quick else 400)
    if model == "sleep_test":          # used by the timeout test only
        class Sleep:
            def fit(self, X, y):
                time.sleep(3600)
        return Sleep()
    raise ValueError(model)


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    t0 = time.time()
    frame = pd.read_pickle(job["frame_path"])
    cfg = ResearchConfig(quick=job.get("quick", False))
    folds = get_folds(frame, job["split"], cfg)
    tr, te = folds[job["fold"]]
    y = frame[TARGET].to_numpy()
    m = build(job["model"], job["split"], job["seed"], job.get("quick", False))
    m.fit(frame.iloc[tr], y[tr])
    out = {"test_idx": te, "pred": np.clip(m.predict(frame.iloc[te]), 0, 100)}
    if job["model"] == "M3_ngboost":
        out["loc"], out["scale"] = m._dist(frame.iloc[te])
    np.savez_compressed(job["out"] + ".npz", **out)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    info = {"seconds": round(time.time() - t0, 1), "cpu_seconds": round(ru.ru_utime + ru.ru_stime, 1),
            "max_rss_mb": round(ru.ru_maxrss / 1024, 1), **(getattr(m, "fit_info_", {}) or {})}
    Path(job["out"] + ".json").write_text(json.dumps(info, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
