"""Decision-reliability update v1 (design: decision_reliability_spec.json): fixed review budgets and risk strata.

Only saved out-of-fold scores are re-analysed; nothing is refitted. The unit is the enrolment record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from . import bootstrap as B
from . import calibration_analysis as CA
from . import config as C
from . import io_data
from . import risk_analysis as R
from . import uncertainty as U

SPEC = json.loads((C.ROOT / "research" / "aiesc2027" / "decision_reliability_spec.json").read_text(encoding="utf-8"))
BUDGETS = SPEC["budget_analysis"]["budgets"]
CURVE_GRID = np.round(np.arange(0.005, 1.0001, 0.005), 3)
STRATA = [("top 5%", 0.0, 0.05), ("5-10%", 0.05, 0.10), ("10-20%", 0.10, 0.20), ("20-40%", 0.20, 0.40),
          ("40-100%", 0.40, 1.0)]
PRACTICAL = SPEC["budget_analysis"]["contrasts"]["practical_threshold"]
CONFIG_LABEL = {"original_oof": "Original OOF (full training folds)",
                "refit_80": "Refit on 80% of each training fold (follow-ups 1-2)"}


@dataclass
class Score:
    key: str
    label: str
    config: str
    score: np.ndarray                    # higher = riskier
    kind: str                            # "ranking" or "probability"
    predicted_grade: np.ndarray | None = None
    probabilities: dict | None = None    # version label -> probability (strata always use `score`)
    reference: bool = False


# ---- tie-aware top-share selection ---------------------------------------------------------------------------
class FoldBlocks:
    """Descending tie blocks of a score within each fold; capture at a share of each fold's (weighted) records."""

    def __init__(self, score, fold):
        s, f = np.asarray(score, float), np.asarray(fold)
        order = np.lexsort((-s, f))
        fs, ss = f[order], s[order]
        new = np.r_[True, (fs[1:] != fs[:-1]) | (ss[1:] != ss[:-1])]
        self.block = np.empty(len(s), int)
        self.block[order] = np.cumsum(new) - 1
        self.n_blocks = int(new.sum())
        bf = fs[new]
        self.slices = [(int(np.searchsorted(bf, k, "left")), int(np.searchsorted(bf, k, "right"))) for k in np.unique(bf)]

    def capture(self, pos, w, shares) -> tuple[np.ndarray, np.ndarray, float]:
        """(flagged, captured) pooled over folds for each share, and the weighted number of positives.
        Records tied at a fold's cut-off enter with the fraction that fills the budget exactly."""
        shares = np.asarray(shares, float)
        bw = np.bincount(self.block, weights=w, minlength=self.n_blocks)
        bp = np.bincount(self.block, weights=w * pos, minlength=self.n_blocks)
        flagged, captured = np.zeros(len(shares)), np.zeros(len(shares))
        for a, b in self.slices:
            cw, cp = np.cumsum(bw[a:b]), np.cumsum(bp[a:b])
            budget = shares * cw[-1]
            i = np.minimum(np.searchsorted(cw, budget - 1e-9, "left"), b - a - 1)
            prev_w = np.where(i > 0, cw[i - 1], 0.0)
            prev_p = np.where(i > 0, cp[i - 1], 0.0)
            frac = np.where(bw[a:b][i] > 0, (budget - prev_w) / bw[a:b][i], 0.0)
            flagged += budget
            captured += prev_p + np.clip(frac, 0, 1) * bp[a:b][i]
        return flagged, captured, float(bp.sum())


def strata_of(score, fold) -> np.ndarray:
    """Stratum index per record from the within-fold mid-rank share (ties stay in one stratum)."""
    s, f = np.asarray(score, float), np.asarray(fold)
    q = np.empty(len(s))
    for k in np.unique(f):
        m = f == k
        q[m] = (rankdata(-s[m], method="average") - 0.5) / m.sum()
    uppers = np.array([u for _, _, u in STRATA])
    return np.minimum(np.searchsorted(uppers, q, "right"), len(STRATA) - 1)


# ---- inputs --------------------------------------------------------------------------------------------------
def load_inputs():
    d = io_data.load_all()
    y, fold = d.y, d.fold
    students = d.frame["student_id"].astype(str).to_numpy()
    cat = d.frame["course_category"].astype(str).to_numpy()
    n = len(y)
    p_orig = U.prob_below(C.RISK_THRESHOLD, d.ngb_loc, d.ngb_scale)
    scores = [Score(m, C.MODELS[m]["label"], "original_oof", -d.preds[m], "ranking", d.preds[m],
                    reference=m == C.REFERENCE_MODEL) for m in C.MODEL_ORDER]
    scores.append(Score("ngboost_prob", "NGBoost P(Y<60)", "original_oof", p_orig, "probability", d.preds["ngboost"],
                        {"raw": p_orig}))

    jobs_r, _ = R.load_jobs("A")
    sc_r, _, probs_r, _ = R.assemble(jobs_r, y)
    jobs_c, infos_c = CA.load_jobs()
    rp = CA.recalibrated_probabilities(jobs_c, y, d.ngb_loc, d.ngb_scale)
    refit_grade = {m: CA.oof_vector(jobs_r, f"{k}_te", n) for m, k in R.REGRESSIONS.items()}
    ngb_mu = CA.oof_vector(jobs_c, "ngb_loc_te", n)
    platt_clf = np.full(n, np.nan)
    from sklearn.linear_model import LogisticRegression
    for j in jobs_r.values():
        te, cal = j["test_idx"], j["cal_idx"]
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(CA._logit(j["clf_cal"])[:, None], (y[cal] < C.RISK_THRESHOLD).astype(int))
        platt_clf[te] = lr.predict_proba(CA._logit(j["clf_te"])[:, None])[:, 1]
    lab = R.METHODS
    for m in ("lgb", "l1", "quantile_q25", "weighted"):
        scores.append(Score(f"refit_{m}", lab[m], "refit_80", sc_r[m], "ranking", np.clip(refit_grade[m], 0, 100),
                            reference=m == R.PRIMARY_REFERENCE))
    scores.append(Score("refit_classifier", lab["classifier"], "refit_80", sc_r["classifier"], "probability", None,
                        {"raw": sc_r["classifier"], "isotonic": probs_r["classifier"], "platt": platt_clf}))
    scores.append(Score("refit_ngboost_prob", lab["ngboost_prob"], "refit_80", sc_r["ngboost_prob"], "probability",
                        np.clip(ngb_mu, 0, 100), {"raw": rp["refit_raw"], "isotonic": rp["refit_isotonic"],
                                                  "platt": rp["refit_platt"]}))
    iv_orig = U.intervals(y, d.ngb_loc, d.ngb_scale, d.ngb_scale_mult, fold, cat, 0.9, C.MONDRIAN_MIN_CALIB)
    iv_refit = CA.assemble(jobs_c, infos_c, y, cat, 0.9)
    checks = _input_checks(y, fold, scores, sc_r, probs_r, rp, jobs_r, jobs_c)
    return dict(y=y, fold=fold, students=students, scores=scores, iv_orig=iv_orig, iv_refit=iv_refit, checks=checks)


def _input_checks(y, fold, scores, sc_r, probs_r, rp, jobs_r, jobs_c) -> pd.DataFrame:
    rows = []

    def add(name, ok, detail=""):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    add("population A: 83,494 records", len(y) == C.EXPECTED_N, len(y))
    add("five test folds partition the records", sorted(np.unique(fold).tolist()) == list(range(C.EXPECTED_N_FOLDS)))
    for s in scores:
        add(f"{s.config} / {s.key}: finite score for every record", bool(np.isfinite(s.score).all()))
        for v, p in (s.probabilities or {}).items():
            add(f"{s.config} / {s.key} / {v}: probabilities in [0, 1]", bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()))
    for f in jobs_r:
        same = all(np.array_equal(jobs_r[f][k], jobs_c[f][k]) for k in ("test_idx", "cal_idx", "fit_idx"))
        add(f"fold {f}: follow-up 2 partition = follow-up 1 partition", same)
        add(f"fold {f}: refitted LightGBM-L2 identical in both follow-ups", np.allclose(jobs_r[f]["lgb_te"], jobs_c[f]["lgb_te"]))
    diff = float(np.nanmax(np.abs(probs_r["ngboost_prob"] - rp["refit_isotonic"])))
    add("isotonic NGBoost probability identical in follow-ups 1 and 2", diff < 1e-12, f"max |diff| {diff:.1e}")
    return pd.DataFrame(rows)


# ---- budget analysis -----------------------------------------------------------------------------------------
def budget_point(y, fold, scores) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = (y < C.RISK_THRESHOLD).astype(float)
    w = np.ones(len(y))
    prev = pos.mean()
    rows, curve = [], []
    for s in scores:
        fb = FoldBlocks(s.score, fold)
        for shares, target in ((np.array(BUDGETS), rows), (CURVE_GRID, curve)):
            fl, cap, P = fb.capture(pos, w, shares)
            for x, a, c in zip(shares, fl, cap):
                target.append({"configuration": s.config, "configuration_label": CONFIG_LABEL[s.config], "score_key": s.key,
                               "score_label": s.label, "score_type": s.kind, "budget_share": float(x),
                               "records_flagged": a, "low_grade_captured": c, "low_grade_total": P,
                               "low_grade_missed": P - c, "recall": c / P, "precision": c / a if a else np.nan,
                               "prevalence": prev, "lift": (c / a) / prev if a else np.nan,
                               "records_per_low_grade_found": a / c if c else np.nan, "random_recall": float(x)})
    return pd.DataFrame(rows), pd.DataFrame(curve)


def budget_bootstrap(y, fold, students, scores, n_boot=C.N_BOOT, seed=C.SEED) -> dict[str, np.ndarray]:
    """Replicates of (recall, precision) per score and budget: array (n_boot, n_budgets, 2)."""
    pos = (y < C.RISK_THRESHOLD).astype(float)
    _, inv = np.unique(students, return_inverse=True)
    G = int(inv.max()) + 1
    blocks = {s.key: FoldBlocks(s.score, fold) for s in scores}
    out = {k: np.empty((n_boot, len(BUDGETS), 2)) for k in blocks}
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        wr = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)[inv]
        for k, fb in blocks.items():
            fl, cap, P = fb.capture(pos, wr, BUDGETS)
            out[k][b, :, 0], out[k][b, :, 1] = cap / P, cap / fl
    return out


def budget_tables(point: pd.DataFrame, reps: dict, scores) -> tuple[pd.DataFrame, pd.DataFrame]:
    pt = point.copy()
    for col, i in (("recall", 0), ("precision", 1)):
        lo, hi = [], []
        for _, r in pt.iterrows():
            x = reps[r["score_key"]][:, BUDGETS.index(r["budget_share"]), i]
            a, b_ = B.percentile_ci(x, C.ALPHA)
            lo.append(a)
            hi.append(b_)
        pt[f"{col}_ci_low"], pt[f"{col}_ci_high"] = lo, hi
    con = []
    for config in CONFIG_LABEL:
        group = [s for s in scores if s.config == config]
        if len(group) < 2:
            continue
        ref = next(s for s in group if s.reference)
        fam = []
        for s in group:
            if s is ref:
                continue
            for i, x in enumerate(BUDGETS):
                d_rec = reps[s.key][:, i, 0] - reps[ref.key][:, i, 0]
                d_pre = reps[s.key][:, i, 1] - reps[ref.key][:, i, 1]
                p_s = pt[(pt.score_key == s.key) & (pt.budget_share == x)].iloc[0]
                p_r = pt[(pt.score_key == ref.key) & (pt.budget_share == x)].iloc[0]
                lo, hi = B.percentile_ci(d_rec, C.ALPHA)
                plo, phi = B.percentile_ci(d_pre, C.ALPHA)
                fam.append({"configuration": config, "family": f"budget recall, {config}", "score_key": s.key,
                            "score_label": s.label, "reference_key": ref.key, "reference_label": ref.label,
                            "budget_share": x, "recall": p_s["recall"], "reference_recall": p_r["recall"],
                            "recall_difference": p_s["recall"] - p_r["recall"], "recall_ci_low_95": lo, "recall_ci_high_95": hi,
                            "precision_difference": p_s["precision"] - p_r["precision"], "precision_ci_low_95": plo,
                            "precision_ci_high_95": phi, "p_boot": B.bootstrap_p(d_rec)})
        fam = pd.DataFrame(fam)
        fam["p_holm"] = B.holm(fam["p_boot"].to_numpy())
        fam["family_size"] = len(fam)
        fam["supported_holm_0.05"] = fam["p_holm"] < C.ALPHA
        fam["practically_important"] = fam["recall_difference"].abs() >= PRACTICAL
        fam["direction"] = np.where(fam["recall_difference"] > 0, "higher recall than reference", "lower recall than reference")
        con.append(fam)
    return pt, pd.concat(con, ignore_index=True)


# ---- risk strata ---------------------------------------------------------------------------------------------
def _wmean(idx, w, v, k):
    num = np.bincount(idx, weights=w * v, minlength=k)
    den = np.bincount(idx, weights=w, minlength=k)
    return np.divide(num, den, out=np.full(k, np.nan), where=den > 0)


def strata_tables(y, fold, students, scores, n_boot=C.N_BOOT, seed=C.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (score, probability version, stratum): ranking reliability and probability calibration with 95% CIs.
    Second frame: paired calibrated-minus-raw differences of the calibration gap per stratum."""
    pos = (y < C.RISK_THRESHOLD).astype(float)
    prev = pos.mean()
    k = len(STRATA)
    _, inv = np.unique(students, return_inverse=True)
    G = int(inv.max()) + 1
    items = []                                   # (score, version, probability or None, stratum index)
    for s in scores:
        st = strata_of(s.score, fold)
        items.append((s, "ranking", None, st))
        for v, p in (s.probabilities or {}).items():
            items.append((s, v, p, st))

    def stats(w):
        res = []
        for s, v, p, st in items:
            r = {"rate": _wmean(st, w, pos, k), "n": np.bincount(st, weights=w, minlength=k)}
            if p is not None:
                r["mean_p"] = _wmean(st, w, p, k)
                r["gap"] = r["mean_p"] - r["rate"]
                r["brier"] = _wmean(st, w, (p - pos) ** 2, k)
            res.append(r)
        return res
    point = stats(np.ones(len(y)))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        reps.append(stats(np.bincount(rng.integers(0, G, G), minlength=G).astype(float)[inv]))
    rows, diffs = [], []
    for i, (s, v, p, st) in enumerate(items):
        for j, (name, lo_s, hi_s) in enumerate(STRATA):
            m = st == j
            r = {"configuration": s.config, "configuration_label": CONFIG_LABEL[s.config], "score_key": s.key,
                 "score_label": s.label, "score_type": s.kind, "probability_version": v if p is not None else "NA (ranking only)",
                 "stratum": name, "stratum_order": j + 1, "nominal_share": hi_s - lo_s, "n_records": int(m.sum()),
                 "realised_share": float(m.mean()), "n_low_grade": int(pos[m].sum()), "observed_low_rate": point[i]["rate"][j]}
            r["observed_low_rate_ci_low"], r["observed_low_rate_ci_high"] = B.percentile_ci([q[i]["rate"][j] for q in reps], C.ALPHA)
            r["lift"] = r["observed_low_rate"] / prev
            r["mean_actual_grade"] = float(y[m].mean())
            r["mean_predicted_grade"] = float(s.predicted_grade[m].mean()) if s.predicted_grade is not None else np.nan
            for col in ("mean_p", "gap", "brier"):
                name_ = {"mean_p": "mean_predicted_probability", "gap": "calibration_gap", "brier": "brier"}[col]
                if p is None:
                    r[name_] = r[f"{name_}_ci_low"] = r[f"{name_}_ci_high"] = np.nan
                else:
                    r[name_] = point[i][col][j]
                    r[f"{name_}_ci_low"], r[f"{name_}_ci_high"] = B.percentile_ci([q[i][col][j] for q in reps], C.ALPHA)
            rows.append(r)
            if p is not None and v != "raw":
                i_raw = next(t for t, it in enumerate(items) if it[0] is s and it[1] == "raw")
                d = np.array([q[i]["gap"][j] - q[i_raw]["gap"][j] for q in reps])
                dd = np.array([abs(q[i]["gap"][j]) - abs(q[i_raw]["gap"][j]) for q in reps])
                lo, hi = B.percentile_ci(d, C.ALPHA)
                alo, ahi = B.percentile_ci(dd, C.ALPHA)
                diffs.append({"configuration": s.config, "score_key": s.key, "score_label": s.label, "version": v,
                              "reference_version": "raw", "stratum": name, "stratum_order": j + 1,
                              "calibration_gap": point[i]["gap"][j], "raw_calibration_gap": point[i_raw]["gap"][j],
                              "gap_difference": point[i]["gap"][j] - point[i_raw]["gap"][j], "gap_difference_ci_low": lo,
                              "gap_difference_ci_high": hi,
                              "abs_gap_difference": abs(point[i]["gap"][j]) - abs(point[i_raw]["gap"][j]),
                              "abs_gap_difference_ci_low": alo, "abs_gap_difference_ci_high": ahi})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


INTERVAL_LABEL = {"normal_raw": "NGBoost Normal, raw", "normal_rescaled": "NGBoost Normal, rescaled",
                  "cf_absolute": "Cross-fold conformal, absolute", "cf_normalised": "Cross-fold conformal, sigma-normalised",
                  "cf_mondrian": "Cross-fold Mondrian (course category)", **CA.METHODS}
LIGHTGBM_BASED = {"cqr_lightgbm", "split_abs_lightgbm"}


def interval_strata(y, fold, students, scores, iv_orig, iv_refit, n_boot=C.N_BOOT, seed=C.SEED) -> pd.DataFrame:
    pos = (y < C.RISK_THRESHOLD).astype(float)
    sk = {s.key: s for s in scores}
    k = len(STRATA)
    _, inv = np.unique(students, return_inverse=True)
    G = int(inv.max()) + 1
    items = []
    for config, ivs in (("original_oof", iv_orig), ("refit_80", iv_refit)):
        for m, (lo, hi) in ivs.items():
            ref = ("refit_lgb" if m in LIGHTGBM_BASED else "refit_ngboost_prob") if config == "refit_80" else "ngboost_prob"
            st = strata_of(sk[ref].score, fold)
            cover = ((y >= lo) & (y <= hi)).astype(float)
            items.append(dict(config=config, method=m, ref=ref, st=st, cover=cover, width=np.maximum(hi - lo, 0),
                              below=(y < lo).astype(float)))

    def stats(w):
        res = []
        for it in items:
            st, c = it["st"], it["cover"]
            r = {"cov": _wmean(st, w, c, k), "cov_all": float((w * c).sum() / w.sum()),
                 "cov_low": _wmean(st, w * pos, c, k), "width": _wmean(st, w, it["width"], k),
                 "below": _wmean(st, w, it["below"], k)}
            r["diff"] = r["cov"] - r["cov_all"]
            res.append(r)
        return res
    point = stats(np.ones(len(y)))
    rng = np.random.default_rng(seed)
    reps = [stats(np.bincount(rng.integers(0, G, G), minlength=G).astype(float)[inv]) for _ in range(n_boot)]
    rows = []
    for i, it in enumerate(items):
        for j, (name, lo_s, hi_s) in enumerate(STRATA):
            m = it["st"] == j
            r = {"configuration": it["config"], "configuration_label": CONFIG_LABEL[it["config"]], "method": it["method"],
                 "method_label": INTERVAL_LABEL[it["method"]], "strata_defined_by": sk[it["ref"]].label + f" ({it['config']})",
                 "stratum": name, "stratum_order": j + 1, "n_records": int(m.sum()), "n_low_grade": int(pos[m].sum()),
                 "nominal_coverage": 0.9, "coverage": point[i]["cov"][j], "coverage_all_records": point[i]["cov_all"]}
            r["coverage_ci_low"], r["coverage_ci_high"] = B.percentile_ci([q[i]["cov"][j] for q in reps], C.ALPHA)
            r["coverage_minus_all"] = point[i]["diff"][j]
            r["coverage_minus_all_ci_low"], r["coverage_minus_all_ci_high"] = B.percentile_ci([q[i]["diff"][j] for q in reps], C.ALPHA)
            r["coverage_actual_low"] = point[i]["cov_low"][j]
            r["coverage_actual_low_ci_low"], r["coverage_actual_low_ci_high"] = B.percentile_ci(
                [q[i]["cov_low"][j] for q in reps], C.ALPHA)
            r["share_actual_below_interval"] = point[i]["below"][j]
            r["mean_width"] = point[i]["width"][j]
            rows.append(r)
    return pd.DataFrame(rows)
