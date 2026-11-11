"""Metrics. All functions accept optional non-negative row weights so the student-cluster bootstrap can
reuse them (a bootstrap replicate = integer weights of resampled students broadcast to their rows).

Risk convention: positive class = "actual grade below the threshold" (low grade / academic risk).
* Fixed-threshold rule: predicted grade < threshold  -> flagged.
* Ranking score for PR-AUC / ROC-AUC: the NEGATIVE predicted grade. This is a ranking, not a calibrated
  probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NA = float("nan")


def _w(n, w):
    return np.ones(n) if w is None else np.asarray(w, float)


# ---- regression -----------------------------------------------------------------------------------------
def regression(y, p, w=None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = _w(len(y), w)
    e = p - y
    sw = w.sum()
    return {"MAE": float(w @ np.abs(e) / sw), "RMSE": float(np.sqrt(w @ (e ** 2) / sw)),
            "mean_signed_error": float(w @ e / sw)}


def ae_quantile(y, p, q=0.9) -> float:
    return float(np.quantile(np.abs(np.asarray(p, float) - np.asarray(y, float)), q))


# ---- classification at a fixed grade threshold -----------------------------------------------------------
def confusion(actual_pos, flagged, w=None) -> dict:
    a, f = np.asarray(actual_pos, bool), np.asarray(flagged, bool)
    w = _w(len(a), w)
    return {"TP": float(w[a & f].sum()), "FP": float(w[~a & f].sum()), "FN": float(w[a & ~f].sum()),
            "TN": float(w[~a & ~f].sum())}


def _div(a, b):
    return a / b if b > 0 else NA


def from_confusion(tp, fp, fn, tn) -> dict:
    prec, rec, spec = _div(tp, tp + fp), _div(tp, tp + fn), _div(tn, tn + fp)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    bal = 0.5 * (rec + spec) if np.isfinite(rec) and np.isfinite(spec) else NA
    return {"precision": prec, "recall": rec, "f1": f1, "specificity": spec, "balanced_accuracy": bal,
            "flag_rate": _div(tp + fp, tp + fp + fn + tn), "prevalence": _div(tp + fn, tp + fp + fn + tn)}


def classification(actual_pos, flagged, w=None) -> dict:
    cm = confusion(actual_pos, flagged, w)
    return {**cm, **from_confusion(cm["TP"], cm["FP"], cm["FN"], cm["TN"])}


# ---- threshold-free ranking metrics (tie-aware, weighted) ------------------------------------------------
class RankBlocks:
    """Pre-sorted blocks of tied scores (descending), so weighted AUCs cost O(n) per replicate."""

    def __init__(self, score):
        s = np.asarray(score, float)
        order = np.argsort(-s, kind="mergesort")
        ss = s[order]
        new = np.r_[True, ss[1:] != ss[:-1]]
        self.block = np.empty(len(s), int)
        self.block[order] = np.cumsum(new) - 1
        self.n_blocks = int(new.sum())

    def sums(self, actual_pos, w):
        a = np.asarray(actual_pos, bool)
        pos = np.bincount(self.block, weights=w * a, minlength=self.n_blocks)
        neg = np.bincount(self.block, weights=w * ~a, minlength=self.n_blocks)
        return pos, neg


def auc_from_blocks(pos, neg) -> tuple[float, float]:
    """(ROC-AUC, average precision) from per-block weighted positives / negatives in descending order.
    ROC-AUC counts ties as 1/2 (Mann-Whitney); AP is the step-wise sum used by scikit-learn."""
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0:
        return NA, NA
    cum_pos_before = np.cumsum(pos) - pos          # positives scored strictly higher than this block
    roc = float((neg * (cum_pos_before + 0.5 * pos)).sum() / (P * N))
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    precision = tp / np.maximum(tp + fp, 1e-300)
    ap = float((pos / P * precision).sum())
    return roc, ap


def ranking(actual_pos, score, w=None) -> dict:
    a = np.asarray(actual_pos, bool)
    w = _w(len(a), w)
    rb = RankBlocks(score)
    roc, ap = auc_from_blocks(*rb.sums(a, w))
    return {"roc_auc": roc, "pr_auc": ap}


def recall_at_budget(actual_pos, score, k: int) -> float:
    """Share of positives among the k highest-risk rows divided by all positives (ties broken by order)."""
    a = np.asarray(actual_pos, bool)
    top = np.argsort(-np.asarray(score, float), kind="mergesort")[:k]
    return float(a[top].sum() / a.sum()) if a.sum() else NA


# ---- probabilistic risk (only for models with a predictive distribution) --------------------------------
def brier(actual_pos, prob) -> float:
    return float(np.mean((np.asarray(prob, float) - np.asarray(actual_pos, float)) ** 2))


def log_loss(actual_pos, prob, eps=1e-12) -> float:
    p = np.clip(np.asarray(prob, float), eps, 1 - eps)
    a = np.asarray(actual_pos, float)
    return float(-np.mean(a * np.log(p) + (1 - a) * np.log(1 - p)))


def reliability(actual_pos, prob, n_bins=10) -> tuple[pd.DataFrame, float]:
    """Equal-width reliability table and expected calibration error (ECE)."""
    p, a = np.asarray(prob, float), np.asarray(actual_pos, float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = b == i
        rows.append({"bin": i, "lower": edges[i], "upper": edges[i + 1], "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else NA,
                     "observed_rate": float(a[m].mean()) if m.any() else NA})
    t = pd.DataFrame(rows)
    ok = t["n"] > 0
    ece = float((t.loc[ok, "n"] * (t.loc[ok, "mean_predicted"] - t.loc[ok, "observed_rate"]).abs()).sum() / len(p))
    return t, ece


# ---- grade-range bias -----------------------------------------------------------------------------------
def band_of(y, bands) -> np.ndarray:
    y = np.asarray(y, float)
    out = np.full(len(y), None, dtype=object)
    for name, lo, hi in bands:
        out[(y >= lo) & (y < hi)] = name
    if any(v is None for v in out):
        raise ValueError("grade outside the defined ranges")
    return out


def band_bias(y, p, bands) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = band_of(y, bands)
    rows = []
    for name, _, _ in bands:
        m = b == name
        e = p[m] - y[m]
        rows.append({"grade_range": name, "n": int(m.sum()), "mean_actual": float(y[m].mean()),
                     "mean_predicted": float(p[m].mean()), "median_actual": float(np.median(y[m])),
                     "median_predicted": float(np.median(p[m])), "MAE": float(np.abs(e).mean()),
                     "RMSE": float(np.sqrt((e ** 2).mean())), "mean_signed_error": float(e.mean()),
                     "ae_p90": float(np.quantile(np.abs(e), 0.9))})
    return pd.DataFrame(rows)


def compression(y, p) -> dict:
    """How strongly predictions shrink toward the centre: OLS slope of prediction on actual grade and the
    ratio of standard deviations (1 = no compression)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    slope = float(np.cov(y, p, bias=True)[0, 1] / np.var(y))
    return {"slope_pred_on_actual": slope, "sd_ratio": float(np.std(p) / np.std(y)),
            "pred_sd": float(np.std(p)), "actual_sd": float(np.std(y)),
            "pred_range_p05_p95": float(np.quantile(p, 0.95) - np.quantile(p, 0.05))}
