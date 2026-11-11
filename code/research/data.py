"""Load the pseudonymised cleaned table, audit it, and build the study frame.

Nothing is imputed here. Rows are only ever excluded (and counted), never altered or invented.
"""
from __future__ import annotations

import pandas as pd

from .config import (ASSESSMENT_NUMERIC, EXCLUDED_COLUMNS, FEATURE_SETS, STUDY_TERM, TARGET,
                     ResearchConfig)

STR_COLS = ["student_id", "instructor_id", "course_key", "section_key", "term_label", "student_college",
            "student_dept", "division", "program", "required_elective", "course_category", "source_file"]


def load_cleaned(cfg: ResearchConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.cleaned_csv, dtype={c: "string" for c in STR_COLS}, encoding="utf-8-sig",
                     low_memory=False)
    return df[df["term_label"] == STUDY_TERM].reset_index(drop=True)


def audit(df: pd.DataFrame) -> dict:
    """Trustworthiness of the target, the ambiguous zeros, and assessment feature availability."""
    g = df[TARGET]
    zero = df[g == 0]
    both = df["final_score"].notna() & g.notna()
    out = {
        "rows": int(len(df)), "students": int(df["student_id"].nunique()),
        "sections": int(df["section_key"].nunique()), "courses": int(df["course_key"].nunique()),
        "single_section_courses": int((df.groupby("course_key")["section_key"].nunique() == 1).sum()),
        "target": {
            "missing": int(g.isna().sum()),
            "out_of_range": int((~g.between(0, 100) & g.notna()).sum()),
            "pass_flag_mismatch": int(df["flag_pass_mismatch"].sum()),
            "grade_band_mismatch": int(df["flag_band_mismatch"].sum()),
            "final_score_equals_target_share": round(float((df.loc[both, "final_score"] == g[both]).mean()), 4),
            "mean": round(float(g.mean()), 3), "std": round(float(g.std()), 3),
        },
        "zero_grades": {
            "rows": int(len(zero)),
            "all_components_zero": int(((zero["daily_score"] == 0) & (zero["midterm_score"] == 0)).sum()),
            "components_missing": int(zero["daily_score"].isna().sum()),
            "share_of_rows": round(float(len(zero) / len(df)), 4),
            "verdict": "ambiguous: no status code separates real 0 from 扣考/停修/缺考; excluded from the "
                       "primary target, included in a sensitivity run",
        },
        "assessment_features": {},
        "excluded_columns": EXCLUDED_COLUMNS,
    }
    for c in ASSESSMENT_NUMERIC:
        present = df[c].notna()
        out["assessment_features"][c] = {
            "missing_share": round(float(1 - present.mean()), 4),
            "equals_target_share": round(float((df.loc[present, c] == g[present]).mean()), 4),
            "corr_with_target": round(float(df.loc[present, c].corr(g[present])), 3),
            "missing_by_college": df.groupby("student_college")[c].apply(lambda s: round(float(s.isna().mean()), 3)).to_dict(),
            "timing_verified": False,
            "note": "no assessment dates in the data; 日常考察 may accumulate all term. Models using these "
                    "are reported as 'with in-term assessment scores', not as early prediction.",
        }
    return out


def build_frame(df: pd.DataFrame, cfg: ResearchConfig) -> tuple[pd.DataFrame, dict]:
    """Rows with a trustworthy target; all feature-set columns; group ids for the splits."""
    meta = {"rows_in": int(len(df))}
    d = df[df[TARGET].notna() & df[TARGET].between(0, 100)]
    meta["rows_invalid_target"] = int(len(df) - len(d))
    d = d[(d["flag_pass_mismatch"] == False) & (d["flag_band_mismatch"] == False)]  # noqa: E712
    if cfg.exclude_zero_grades:
        n0 = int((d[TARGET] == 0).sum())
        d = d[d[TARGET] != 0]
        meta["rows_zero_grade_excluded"] = n0
    cols = sorted({c for fs in FEATURE_SETS.values() for c in fs["categorical"] + fs["numeric"]})
    bad = [c for c in cols if c in EXCLUDED_COLUMNS]
    if bad:
        raise ValueError(f"Excluded columns used as features: {bad}")
    frame = d[["row_id", "student_id", "section_key", "student_college", "student_dept", "program",
               "course_category", TARGET] + [c for c in cols if c not in ("student_college", "student_dept",
                                                                             "program", "course_category")]].copy()
    if cfg.quick:
        frame = frame.sample(n=min(len(frame), 4000), random_state=cfg.seed).reset_index(drop=True)
    meta["rows_used"] = int(len(frame))
    meta["missingness"] = {c: round(float(frame[c].isna().mean()), 4) for c in cols}
    return frame.reset_index(drop=True), meta
