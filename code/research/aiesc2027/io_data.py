"""Load and validate the analysis frame, the saved fold partition and the five models' OOF predictions.

Inputs  (read-only): outputs/data/cleaned_grades.csv, outputs/research/exploratory/oof.npz,
                     outputs/research/exploratory_amendment1/jobs/M{2,3}_*__T3__f*.npz (+ .json),
                     outputs/research/final/comparison_student.csv (registered MAE / RMSE)
Output: an `AlignedData` object plus a list of validation checks (each PASS / FAIL with details).

The analysis frame is rebuilt with the IR study's own `research.data.build_frame`, so row order is the
order in which every saved OOF array was written. The fold partition is NOT recomputed (GroupKFold orders
equal-sized students with numpy's unstable argsort, whose tie order depends on the CPU's SIMD sort kernel, so the
partition is not reproducible on this host even with the original package versions); it is read from the saved
test indices.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

sys.path.insert(0, str(C.ROOT))
from research.config import (COURSE_CATEGORICAL, COURSE_NUMERIC, STUDENT_CATEGORICAL,  # noqa: E402
                             TARGET, ResearchConfig)
from research.data import build_frame, load_cleaned  # noqa: E402

log = logging.getLogger("aiesc2027")
FEATURES = COURSE_CATEGORICAL + STUDENT_CATEGORICAL + COURSE_NUMERIC


class ValidationError(RuntimeError):
    pass


@dataclass
class AlignedData:
    frame: pd.DataFrame                 # row_id, student_id (pseudonym, never exported), fold, y, ...
    preds: dict[str, np.ndarray]        # model key -> OOF point prediction (clipped to [0, 100])
    ngb_loc: np.ndarray
    ngb_scale: np.ndarray
    ngb_scale_mult: np.ndarray          # in-fold validation scale multiplier, per row (by fold)
    checks: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        return self.frame["y"].to_numpy(float)

    @property
    def fold(self) -> np.ndarray:
        return self.frame["fold"].to_numpy(int)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, name: str, ok: bool, detail="", fatal: bool = True):
        self.rows.append({"check": name, "status": "PASS" if ok else ("FAIL" if fatal else "WARN"),
                          "detail": str(detail)})
        (log.info if ok else log.error)("check %-55s %s %s", name, "PASS" if ok else "FAIL", detail)
        if not ok and fatal:
            raise ValidationError(f"{name}: {detail}")


# ---- frame --------------------------------------------------------------------------------------------
def load_frame(chk: Checks) -> tuple[pd.DataFrame, dict]:
    for name, path in (("IR.csv", C.RAW_CSV), ("cleaned_grades.csv", C.CLEANED_CSV)):
        chk.add(f"input exists: {name}", path.exists(), path)
        h = sha256(path)
        chk.add(f"data version (SHA-256 matches IR manifest): {name}", h == C.EXPECTED_SHA256[name], h[:16])
    raw_rows = sum(1 for _ in open(C.RAW_CSV, encoding="utf-8-sig")) - 1      # header excluded; no fields kept
    cleaned_all = pd.read_csv(C.CLEANED_CSV, usecols=["row_id", "term_label"], dtype={"term_label": "string"},
                              encoding="utf-8-sig")
    cfg = ResearchConfig()
    term = load_cleaned(cfg)
    frame, meta = build_frame(term, cfg)
    counts = {"raw_rows": raw_rows, "cleaned_rows": int(len(cleaned_all)), "term_rows": int(len(term)),
              "zero_rows": int(meta.get("rows_zero_grade_excluded", -1)), "analysis_rows": int(len(frame)),
              "students": int(frame["student_id"].nunique()), "sections": int(frame["section_key"].nunique()),
              "courses": int(frame["course_key"].nunique())}
    chk.add("raw rows = 86,014", raw_rows == C.EXPECTED_RAW_ROWS, raw_rows)
    chk.add("cleaned rows = 86,014 (no row dropped in cleaning)", counts["cleaned_rows"] == C.EXPECTED_RAW_ROWS,
            counts["cleaned_rows"])
    chk.add("study term 113-2 rows = 85,567", counts["term_rows"] == C.EXPECTED_TERM_ROWS, counts["term_rows"])
    chk.add("unexplained zero grades excluded = 2,073", counts["zero_rows"] == C.EXPECTED_ZERO_ROWS, counts["zero_rows"])
    chk.add("analysis rows = 83,494", counts["analysis_rows"] == C.EXPECTED_N, counts["analysis_rows"])
    chk.add("rows with invalid target or pass/band mismatch = 0",
            meta["rows_invalid_target"] == 0 and len(term) - counts["zero_rows"] == len(frame), meta["rows_invalid_target"])

    # schema / dtypes / missingness of the columns this analysis uses
    need = {"row_id": "integer", TARGET: "numeric", "student_id": "string", "course_category": "string",
            "program": "string"}
    for col, kind in need.items():
        ok = col in frame.columns and (
            pd.api.types.is_integer_dtype(frame[col]) if kind == "integer" else
            pd.api.types.is_numeric_dtype(frame[col]) if kind == "numeric" else
            pd.api.types.is_string_dtype(frame[col]))
        chk.add(f"column {col} present with {kind} dtype", ok, frame[col].dtype if col in frame else "missing")
        chk.add(f"column {col} has no missing values", int(frame[col].isna().sum()) == 0, int(frame[col].isna().sum()))
    y = frame[TARGET]
    chk.add("target within (0, 100] (zeros excluded)", bool(y.between(1e-9, 100).all()), f"min {y.min()}, max {y.max()}")
    chk.add("row_id unique", frame["row_id"].is_unique, int(frame["row_id"].duplicated().sum()))
    dup = int(frame.duplicated(["student_id", "section_key"]).sum())
    chk.add("no duplicate (student, section) enrolment", dup == 0, dup)
    chk.add("no fully duplicated record", int(frame.duplicated().sum()) == 0, int(frame.duplicated().sum()))
    leaked = sorted(set(FEATURES) & C.FORBIDDEN_PREDICTORS)
    chk.add("model features contain no target-derived / in-term assessment column", not leaked, leaked or "none")

    # official, programme-specific pass threshold: evaluation label only (merged by row_id)
    thr = pd.read_csv(C.CLEANED_CSV, usecols=["row_id", C.OFFICIAL_THRESHOLD_COL], encoding="utf-8-sig")
    frame = frame.merge(thr, on="row_id", how="left", validate="one_to_one")
    chk.add("official pass threshold available for every row", int(frame[C.OFFICIAL_THRESHOLD_COL].isna().sum()) == 0,
            frame[C.OFFICIAL_THRESHOLD_COL].value_counts().to_dict())
    out = frame[["row_id", "student_id", "course_category", "program", C.OFFICIAL_THRESHOLD_COL] + FEATURES].copy()
    out = out.loc[:, ~out.columns.duplicated()]
    out["y"] = y.to_numpy(float)
    return out, counts


# ---- folds ---------------------------------------------------------------------------------------------
def load_folds(frame: pd.DataFrame, chk: Checks) -> np.ndarray:
    n = len(frame)
    fold = np.full(n, -1)
    ref = {}
    for f in range(C.EXPECTED_N_FOLDS):
        idx = {}
        for m in ("M2_ft_transformer", "M3_ngboost"):
            p = C.AMENDMENT_JOBS / f"{m}__{C.AMENDMENT_UNIT}__f{f}.npz"
            chk.add(f"fold file exists: {p.name}", p.exists(), p)
            idx[m] = np.load(p)["test_idx"]
        chk.add(f"fold {f}: FT-Transformer and NGBoost test indices identical",
                np.array_equal(idx["M2_ft_transformer"], idx["M3_ngboost"]), len(idx["M3_ngboost"]))
        te = idx["M3_ngboost"]
        chk.add(f"fold {f}: indices in range and not yet assigned", bool(te.min() >= 0 and te.max() < n and (fold[te] == -1).all()),
                f"{len(te)} rows")
        fold[te] = f
        ref[f] = te
    chk.add("folds form a partition of all 83,494 rows", bool((fold >= 0).all()), np.bincount(fold).tolist())
    per_student = pd.Series(fold).groupby(frame["student_id"].to_numpy()).nunique()
    chk.add("student-grouped: every student in exactly one test fold", int(per_student.max()) == 1, int(per_student.max()))
    return fold


def fold_consistency(frame: pd.DataFrame, fold: np.ndarray, pred: np.ndarray) -> dict:
    """Rows with identical features get identical predictions from one fold model and (almost surely)
    different predictions from different fold models. Agreement with the saved partition therefore
    verifies that a prediction array was produced on these folds and in this row order."""
    key = frame[FEATURES].astype(str).agg("|".join, axis=1).to_numpy()
    d = pd.DataFrame({"k": key, "f": fold, "p": np.round(pred, 8)})
    within = float((d.groupby(["k", "f"])["p"].nunique() == 1).mean())
    g = d.groupby("k").agg(nf=("f", "nunique"), npred=("p", "nunique"))
    g = g[g["nf"] > 1]
    across = float((g["nf"] == g["npred"]).mean()) if len(g) else float("nan")
    return {"within_fold_identical": within, "across_fold_distinct": across, "multi_fold_feature_keys": int(len(g))}


# ---- predictions ---------------------------------------------------------------------------------------
def load_predictions(frame: pd.DataFrame, fold: np.ndarray, chk: Checks):
    n = len(frame)
    y = frame["y"].to_numpy(float)
    z = np.load(C.EXPLORATORY_OOF)
    fs = pd.read_csv(C.AMENDMENT_FIT_STATUS)
    preds, loc, scale, mult = {}, np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for key, spec in C.MODELS.items():
        if spec["kind"] == "npz":
            chk.add(f"{spec['label']}: OOF key present", spec["key"] in z.files, spec["key"])
            p = z[spec["key"]].astype(float)
            chk.add(f"{spec['label']}: OOF length = analysis rows", len(p) == n, len(p))
        else:
            done = fs[(fs["model"] == spec["key"]) & (fs["unit"] == C.AMENDMENT_UNIT) & (fs["status"] == "complete")]
            chk.add(f"{spec['label']}: all 5 student-split folds complete", done["fold"].nunique() == 5,
                    sorted(done["fold"].tolist()))
            p = np.full(n, np.nan)
            for f in range(C.EXPECTED_N_FOLDS):
                zz = np.load(C.AMENDMENT_JOBS / f"{spec['key']}__{C.AMENDMENT_UNIT}__f{f}.npz")
                p[zz["test_idx"]] = zz["pred"]
                if key == "ngboost":
                    loc[zz["test_idx"]], scale[zz["test_idx"]] = zz["loc"], zz["scale"]
                    info = json.loads((C.AMENDMENT_JOBS / f"{spec['key']}__{C.AMENDMENT_UNIT}__f{f}.json").read_text())
                    mult[zz["test_idx"]] = info["scale_multiplier"]
        chk.add(f"{spec['label']}: no missing / non-finite prediction", bool(np.isfinite(p).all()), int((~np.isfinite(p)).sum()))
        chk.add(f"{spec['label']}: predictions within [0, 100]", bool(((p >= 0) & (p <= 100)).all()),
                f"[{np.nanmin(p):.3f}, {np.nanmax(p):.3f}]")
        preds[key] = p
    chk.add("NGBoost: distribution parameters present for every row", bool(np.isfinite(loc).all() and np.isfinite(scale).all()),
            int((~np.isfinite(loc)).sum()))
    chk.add("NGBoost: scale > 0", bool((scale > 0).all()), f"min {np.nanmin(scale):.4f}")
    chk.add("NGBoost: point prediction = clip(loc, 0, 100)", bool(np.allclose(preds["ngboost"], np.clip(loc, 0, 100))), "")
    chk.add("NGBoost: in-fold scale multiplier available per fold", bool(np.isfinite(mult).all()),
            sorted(set(np.round(mult, 4).tolist())))

    # registered MAE / RMSE of the IR study (comparison_student.csv) must be reproduced
    reg = pd.read_csv(C.FINAL_COMPARISON_STUDENT)
    for key, spec in C.MODELS.items():
        r = reg[reg["system"] == spec["registry_name"]]
        mae = float(np.mean(np.abs(y - preds[key])))
        rmse = float(np.sqrt(np.mean((y - preds[key]) ** 2)))
        ok = len(r) == 1 and abs(mae - r["MAE"].iat[0]) < 6e-5 and abs(rmse - r["RMSE"].iat[0]) < 6e-5
        chk.add(f"{spec['label']}: MAE/RMSE reproduce the IR registry", ok,
                f"MAE {mae:.4f} vs {r['MAE'].iat[0] if len(r) else 'NA'}; RMSE {rmse:.4f} vs {r['RMSE'].iat[0] if len(r) else 'NA'}")
    # fold / row-order consistency of the arrays that carry no fold information themselves
    cons = {}
    for key, spec in C.MODELS.items():
        if spec["kind"] != "npz":
            continue
        c = fold_consistency(frame, fold, preds[key])
        cons[key] = c
        chk.add(f"{spec['label']}: predictions consistent with saved folds", c["within_fold_identical"] == 1.0
                and c["across_fold_distinct"] >= 0.999, c)
    return preds, loc, scale, mult, cons


def load_all() -> AlignedData:
    chk = Checks()
    frame, counts = load_frame(chk)
    fold = load_folds(frame, chk)
    frame["fold"] = fold
    preds, loc, scale, mult, cons = load_predictions(frame, fold, chk)
    counts["fold_consistency"] = cons
    return AlignedData(frame=frame, preds=preds, ngb_loc=loc, ngb_scale=scale, ngb_scale_mult=mult,
                       checks=chk.rows, counts=counts)
