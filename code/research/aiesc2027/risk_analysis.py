"""Analysis of follow-up 2: risk-oriented decision rules and objectives (design: risk_methods_spec.json).

Every threshold and recalibration map for test fold k is computed from the calibration students of training
fold k only; test labels are used only for evaluation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import bootstrap as B
from . import config as C
from . import metrics as M

JOBS = C.OUT / "risk_methods" / "jobs"
METHODS = {"lgb": "LightGBM-L2 regression", "l1": "LightGBM-L1 regression",
           "classifier": "LightGBM classifier (grade < 60)", "quantile_q25": "LightGBM quantile regression (q = 0.25)",
           "weighted": "LightGBM-L2, class-weighted", "ngboost_prob": "NGBoost P(Y < 60)"}
OPS = {"native": "Native rule", "OP1": "OP1: flag budget = calibration prevalence",
       "OP2": "OP2: calibration recall >= 0.50"}
REGRESSIONS = {"lgb": "lgb", "l1": "l1", "weighted": "weighted", "quantile_q25": "q25"}
CLS = ["precision", "recall", "f1", "specificity", "balanced_accuracy", "flag_rate"]
PRIMARY_METRICS = ["pr_auc", "roc_auc", "f1_OP1", "precision_OP2"]
PRIMARY_REFERENCE = "lgb"
THRESHOLD_CONTRASTS = [("lgb", "OP1"), ("lgb", "OP2"), ("l1", "OP1"), ("l1", "OP2")]


def load_jobs(population: str) -> tuple[dict, list[dict]]:
    jobs, infos = {}, []
    for f in range(C.EXPECTED_N_FOLDS):
        p = JOBS / f"{population}_fold{f}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run risk_fit.py --population {population} --fold {f}")
        jobs[f] = dict(np.load(p))
        infos.append(json.loads((JOBS / f"{population}_fold{f}.json").read_text()))
    return jobs, infos


def check_jobs(jobs: dict, y: np.ndarray, students: np.ndarray, saved_fold: np.ndarray | None) -> pd.DataFrame:
    rows = []

    def add(name, ok, detail=""):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})

    seen = np.zeros(len(y), int)
    for f, j in jobs.items():
        te, cal, fit = j["test_idx"], j["cal_idx"], j["fit_idx"]
        if saved_fold is not None:
            add(f"fold {f}: test indices equal the saved partition", np.array_equal(np.sort(te), np.flatnonzero(saved_fold == f)))
        add(f"fold {f}: fit / calibration / test cover all rows once",
            np.array_equal(np.sort(np.concatenate([fit, cal, te])), np.arange(len(y))))
        sf, sc, st = (set(students[i]) for i in (fit, cal, te))
        add(f"fold {f}: students disjoint across fit / calibration / test", not (sf & sc or sc & st or sf & st))
        add(f"fold {f}: all predictions finite", all(np.isfinite(v).all() for k, v in j.items() if k.endswith(("_cal", "_te"))))
        add(f"fold {f}: classifier output within [0, 1]", bool((j["clf_te"] >= 0).all() and (j["clf_te"] <= 1).all()))
        seen[te] += 1
    add("test folds form a partition of all rows", bool((seen == 1).all()))
    return pd.DataFrame(rows)


# ---- scores, thresholds and flags --------------------------------------------------------------------------
def _fold_scores(j: dict, y_cal: np.ndarray) -> dict:
    """method -> (score_cal, score_te, native_flag_te, calibrated_probability_te or None); higher score = riskier."""
    from sklearn.isotonic import IsotonicRegression
    a_cal = (y_cal < C.RISK_THRESHOLD).astype(float)
    out = {}
    for m, key in REGRESSIONS.items():
        out[m] = (-j[f"{key}_cal"], -j[f"{key}_te"], j[f"{key}_te"] < C.RISK_THRESHOLD, None)
    p_ngb_cal = norm.cdf((C.RISK_THRESHOLD - j["ngb_loc_cal"]) / j["ngb_scale_cal"])
    p_ngb_te = norm.cdf((C.RISK_THRESHOLD - j["ngb_loc_te"]) / j["ngb_scale_te"])
    for m, pc, pt in (("classifier", j["clf_cal"], j["clf_te"]), ("ngboost_prob", p_ngb_cal, p_ngb_te)):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(pc, a_cal)
        prob = iso.predict(pt)
        out[m] = (pc, pt, prob >= 0.5, prob)
    return out


def threshold_op1(score_cal: np.ndarray, pos_cal: np.ndarray) -> float:
    k = max(1, int(round(pos_cal.mean() * len(score_cal))))
    return float(np.sort(score_cal)[::-1][k - 1])


def threshold_op2(score_cal: np.ndarray, pos_cal: np.ndarray, target: float = 0.5) -> float:
    s = np.sort(score_cal[pos_cal])[::-1]
    return float(s[int(np.ceil(target * len(s))) - 1])


def assemble(jobs: dict, y: np.ndarray) -> tuple[dict, dict, dict, pd.DataFrame]:
    """OOF risk scores, flags per (method, operating point), calibrated probabilities and per-fold thresholds."""
    n = len(y)
    scores = {m: np.full(n, np.nan) for m in METHODS}
    flags = {(m, op): np.zeros(n, bool) for m in METHODS for op in OPS}
    probs = {m: np.full(n, np.nan) for m in ("classifier", "ngboost_prob")}
    thr = []
    for f, j in jobs.items():
        te, cal = j["test_idx"], j["cal_idx"]
        pos_cal = y[cal] < C.RISK_THRESHOLD
        for m, (sc, st, native, prob) in _fold_scores(j, y[cal]).items():
            scores[m][te] = st
            flags[(m, "native")][te] = native
            t1, t2 = threshold_op1(sc, pos_cal), threshold_op2(sc, pos_cal)
            flags[(m, "OP1")][te] = st >= t1
            flags[(m, "OP2")][te] = st >= t2
            if prob is not None:
                probs[m][te] = prob
            reg = m in REGRESSIONS
            thr.append({"fold": f, "method": m, "threshold_OP1": t1, "threshold_OP2": t2,
                        "grade_cutoff_OP1": -t1 if reg else np.nan, "grade_cutoff_OP2": -t2 if reg else np.nan,
                        "cal_flag_rate_OP1": float((sc >= t1).mean()), "cal_recall_OP2": float((sc[pos_cal] >= t2).mean()),
                        "cal_prevalence": float(pos_cal.mean())})
    return scores, flags, probs, pd.DataFrame(thr)


def metrics_table(y, scores, flags, preds_reg: dict, rows_mask=None) -> pd.DataFrame:
    s = np.ones(len(y), bool) if rows_mask is None else rows_mask
    pos = y[s] < C.RISK_THRESHOLD
    out = []
    for m in METHODS:
        rk = M.ranking(pos, scores[m][s])
        for op in OPS:
            c = M.classification(pos, flags[(m, op)][s])
            row = {"method": m, "label": METHODS[m], "operating_point": op, "n": int(s.sum()),
                   "prevalence": float(pos.mean()), **rk, **{k: c[k] for k in ("TP", "FP", "FN", "TN", *CLS)}}
            if m in preds_reg:
                r = M.regression(y[s], preds_reg[m][s])
                row.update({"MAE": r["MAE"], "RMSE": r["RMSE"]})
            out.append(row)
    return pd.DataFrame(out)


def probability_table(y, probs, fold) -> pd.DataFrame:
    actual = y < C.RISK_THRESHOLD
    base = np.empty(len(y))
    for f in np.unique(fold):
        base[fold == f] = actual[fold != f].mean()
    rows = []
    for m, p in probs.items():
        _, ece = M.reliability(actual, p)
        bs, bs0 = M.brier(actual, p), M.brier(actual, base)
        rows.append({"method": m, "label": METHODS[m] + " (isotonic)", "mean_probability": float(p.mean()),
                     "prevalence": float(actual.mean()), "brier": bs, "brier_skill_score": 1 - bs / bs0, "ece_10_bins": ece})
    return pd.DataFrame(rows)


# ---- paired student-cluster bootstrap ------------------------------------------------------------------------
class RiskBootstrap:
    def __init__(self, students, pos, scores: dict, flags: dict):
        _, self.inv = np.unique(np.asarray(students).astype(str), return_inverse=True)
        self.G = int(self.inv.max()) + 1
        self.pos = np.asarray(pos, bool)
        self.blocks = {m: M.RankBlocks(s) for m, s in scores.items()}
        agg = lambda v: np.bincount(self.inv, weights=v.astype(float), minlength=self.G)  # noqa: E731
        self.S = {k: np.column_stack([agg(self.pos & f), agg(~self.pos & f), agg(self.pos & ~f), agg(~self.pos & ~f)])
                  for k, f in flags.items()}

    def _one(self, wc) -> dict:
        wr = wc[self.inv]
        res = {}
        for m, rb in self.blocks.items():
            roc, ap = M.auc_from_blocks(*rb.sums(self.pos, wr))
            res[m] = {"pr_auc": ap, "roc_auc": roc}
        for (m, op), S in self.S.items():
            c = M.from_confusion(*(wc @ S))
            res[m].update({f"{k}_{op}": c[k] for k in CLS})
        return res

    def point(self) -> dict:
        return self._one(np.ones(self.G))

    def run(self, n_boot: int, seed: int) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        rec = {m: [] for m in self.blocks}
        for _ in range(n_boot):
            r = self._one(np.bincount(rng.integers(0, self.G, self.G), minlength=self.G).astype(float))
            for m in rec:
                rec[m].append(r[m])
        return {m: pd.DataFrame(v) for m, v in rec.items()}


def per_method_ci(point: dict, reps: dict, alpha=C.ALPHA) -> pd.DataFrame:
    rows = []
    for m, df in reps.items():
        for k in df.columns:
            lo, hi = B.percentile_ci(df[k], alpha)
            rows.append({"method": m, "metric": k, "estimate": point[m][k], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def _contrast_rows(pairs, point, reps, family, alpha, practical):
    rows = []
    for (ma, ka), (mb, kb), name in pairs:
        d = (reps[ma][ka] - reps[mb][kb]).to_numpy()
        est = point[ma][ka] - point[mb][kb]
        lo, hi = B.percentile_ci(d, alpha)
        alo, ahi = B.percentile_ci(d, alpha / len(pairs))
        rows.append({"family": family, "contrast": name, "candidate": f"{ma}:{ka}", "reference": f"{mb}:{kb}",
                     "candidate_value": point[ma][ka], "reference_value": point[mb][kb], "difference": est,
                     "ci_low_95": lo, "ci_high_95": hi, "ci_low_bonf": alo, "ci_high_bonf": ahi, "p_boot": B.bootstrap_p(d),
                     "practically_important": abs(est) >= practical})
    df = pd.DataFrame(rows)
    df["p_holm"] = B.holm(df["p_boot"])
    df["supported_holm_0.05"] = df["p_holm"] < alpha
    df["bonf_ci_excludes_0"] = (df["ci_low_bonf"] > 0) | (df["ci_high_bonf"] < 0)
    df["family_size"] = len(pairs)
    df["improves"] = df["supported_holm_0.05"] & df["practically_important"] & (df["difference"] > 0)
    return df


def contrasts(point: dict, reps: dict, alpha=C.ALPHA, practical=0.05) -> pd.DataFrame:
    prim = [((m, k), (PRIMARY_REFERENCE, k), f"{m} vs {PRIMARY_REFERENCE} (tuned): {k}")
            for m in METHODS if m != PRIMARY_REFERENCE for k in PRIMARY_METRICS]
    thr = [((m, f"{k}_{op}"), (m, f"{k}_native"), f"{m}: {op} vs native rule: {k}")
           for m, op in THRESHOLD_CONTRASTS for k in ("recall", "f1")]
    return pd.concat([_contrast_rows(prim, point, reps, "primary", alpha, practical),
                      _contrast_rows(thr, point, reps, "threshold", alpha, practical)], ignore_index=True)


def recall_flag_curves(y, scores, max_points=200) -> pd.DataFrame:
    pos = y < C.RISK_THRESHOLD
    rows = []
    for m, s in scores.items():
        order = np.argsort(-s, kind="mergesort")
        tp = np.cumsum(pos[order])
        idx = np.unique(np.linspace(0, len(y) - 1, max_points).astype(int))
        for i in idx:
            rows.append({"method": m, "flag_rate": (i + 1) / len(y), "recall": tp[i] / pos.sum(),
                         "precision": tp[i] / (i + 1)})
    return pd.DataFrame(rows)
