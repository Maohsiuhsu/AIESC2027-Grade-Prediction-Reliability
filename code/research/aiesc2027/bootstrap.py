"""Paired student-cluster bootstrap for all models at once.

Each replicate resamples STUDENTS with replacement (seeded numpy Generator); every model is evaluated
on the same replicate, so differences between models are paired. Additive statistics (absolute and
squared errors, confusion counts) are pre-aggregated per student; ranking metrics use per-row weights
on pre-sorted tie blocks.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import metrics as M

log = logging.getLogger("aiesc2027")
CLS = ["precision", "recall", "f1", "specificity", "balanced_accuracy"]


class PairedClusterBootstrap:
    def __init__(self, groups, y, preds: dict[str, np.ndarray], threshold: float,
                 extra_scores: dict[str, np.ndarray] | None = None):
        _, self.inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        self.G = int(self.inv.max()) + 1
        self.y = np.asarray(y, float)
        self.pos = self.y < threshold
        self.models = list(preds)
        self.S, self.blocks = {}, {}
        for k, p in preds.items():
            e = p - self.y
            flag = p < threshold
            cols = [np.abs(e), e ** 2, np.ones_like(e), self.pos & flag, ~self.pos & flag, self.pos & ~flag,
                    ~self.pos & ~flag]
            self.S[k] = np.column_stack([np.bincount(self.inv, weights=c.astype(float), minlength=self.G) for c in cols])
            self.blocks[k] = M.RankBlocks(-p)
        self.extra = {k: M.RankBlocks(s) for k, s in (extra_scores or {}).items()}

    def _metrics(self, k, wc, wr) -> dict:
        s = wc @ self.S[k]
        out = {"MAE": s[0] / s[2], "RMSE": np.sqrt(s[1] / s[2])}
        out.update({m: v for m, v in M.from_confusion(*s[3:7]).items() if m in CLS})
        roc, ap = M.auc_from_blocks(*self.blocks[k].sums(self.pos, wr))
        out["roc_auc"], out["pr_auc"] = roc, ap
        return out

    def point(self) -> dict:
        wc, wr = np.ones(self.G), np.ones(len(self.y))
        res = {k: self._metrics(k, wc, wr) for k in self.models}
        for k, rb in self.extra.items():
            roc, ap = M.auc_from_blocks(*rb.sums(self.pos, wr))
            res[k] = {"roc_auc": roc, "pr_auc": ap}
        return res

    def run(self, n_boot: int, seed: int) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        rec = {k: [] for k in self.models + list(self.extra)}
        for b in range(n_boot):
            wc = np.bincount(rng.integers(0, self.G, self.G), minlength=self.G).astype(float)
            wr = wc[self.inv]
            for k in self.models:
                rec[k].append(self._metrics(k, wc, wr))
            for k, rb in self.extra.items():
                roc, ap = M.auc_from_blocks(*rb.sums(self.pos, wr))
                rec[k].append({"roc_auc": roc, "pr_auc": ap})
            if (b + 1) % 1000 == 0:
                log.info("bootstrap %d / %d", b + 1, n_boot)
        return {k: pd.DataFrame(v) for k, v in rec.items()}


def percentile_ci(x, alpha):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 100 * alpha / 2)), float(np.percentile(x, 100 * (1 - alpha / 2)))


def bootstrap_p(d) -> float:
    """Two-sided percentile-bootstrap p-value for H0: difference = 0."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    B = len(d)
    lo, hi = (np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def summarise(point: dict, reps: dict[str, pd.DataFrame], reference: str, metrics: list[str],
              lower_is_better: set[str], alpha: float, practical: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model CIs and paired differences vs `reference` with Holm and Bonferroni control over the
    whole family (every non-reference model x every metric)."""
    rows = []
    for k, df in reps.items():
        for m in df.columns:
            lo, hi = percentile_ci(df[m], alpha)
            rows.append({"model": k, "metric": m, "estimate": point[k][m], "ci_low": lo, "ci_high": hi})
    per_model = pd.DataFrame(rows)
    others = [k for k in reps if k != reference and set(metrics) <= set(reps[k].columns)]
    m_family = len(others) * len(metrics)
    a_adj = alpha / m_family
    rows = []
    for k in others:
        for m in metrics:
            d = (reps[k][m] - reps[reference][m]).to_numpy()
            est = point[k][m] - point[reference][m]
            lo, hi = percentile_ci(d, alpha)
            alo, ahi = percentile_ci(d, a_adj)
            better = -est if m in lower_is_better else est
            rows.append({"model": k, "reference": reference, "metric": m, "model_value": point[k][m],
                         "reference_value": point[reference][m], "difference": est,
                         "relative_difference": est / point[reference][m] if point[reference][m] else np.nan,
                         "ci_low_95": lo, "ci_high_95": hi, "ci_low_bonf": alo, "ci_high_bonf": ahi,
                         "p_boot": bootstrap_p(d),
                         "direction": "better" if better > 0 else ("worse" if better < 0 else "equal"),
                         "practical_threshold": practical[m], "practically_important": abs(est) >= practical[m]})
    diff = pd.DataFrame(rows)
    diff["p_holm"] = holm(diff["p_boot"])
    diff["supported_holm_0.05"] = diff["p_holm"] < alpha
    diff["bonf_ci_excludes_0"] = (diff["ci_low_bonf"] > 0) | (diff["ci_high_bonf"] < 0)
    diff["family_size"] = m_family
    return per_model, diff
