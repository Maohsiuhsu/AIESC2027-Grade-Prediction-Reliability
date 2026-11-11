"""Configuration of the single-dataset research pipeline (separate from the production pipeline).

Task: predict the term grade of an enrolment in ONE semester (113-2) from information that is
available for that enrolment, evaluated with grouped cross-validation. This is NOT next-year,
NOT cross-college and NOT instructor-specific prediction; see README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET = "term_grade"
STUDY_TERM = "113-2"            # the only complete semester in IR.csv

# ---- feature sets --------------------------------------------------------------------------------
COURSE_CATEGORICAL = ["course_key", "required_elective", "course_category"]
COURSE_NUMERIC = ["credits", "section_size", "section_n_student_colleges"]
STUDENT_CATEGORICAL = ["student_college", "student_dept", "division", "program"]
INSTRUCTOR_CATEGORICAL = ["instructor_id"]          # recorded for ~21% of rows (one college only)
ASSESSMENT_NUMERIC = ["midterm_score", "daily_score"]  # timing NOT verified: not "early prediction"

FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "course": {"categorical": COURSE_CATEGORICAL, "numeric": COURSE_NUMERIC},
    "course+student": {"categorical": COURSE_CATEGORICAL + STUDENT_CATEGORICAL, "numeric": COURSE_NUMERIC},
    "course+student+instructor": {"categorical": COURSE_CATEGORICAL + STUDENT_CATEGORICAL + INSTRUCTOR_CATEGORICAL,
                                  "numeric": COURSE_NUMERIC},
    "course+student+assessment": {"categorical": COURSE_CATEGORICAL + STUDENT_CATEGORICAL,
                                  "numeric": COURSE_NUMERIC + ASSESSMENT_NUMERIC},
}
MAIN_FEATURE_SET = "course+student"

# Columns that must never be features, with the reason (checked by a test and at run time).
EXCLUDED_COLUMNS: dict[str, str] = {
    "final_score": "duplicate of the target (equal in 100% of rows)",
    "passed": "derived from the target", "grade_band": "derived from the target",
    "components_missing_flag": "known only at term end; derived from assessment completeness",
    "is_zero_grade": "derived from the target",
    "flag_pass_mismatch": "derived from the target", "flag_band_mismatch": "derived from the target",
    "course_college": "invalid label: follows the student's class, not the course",
    "course_dept": "invalid label: follows the student's class, not the course",
    "enrolled_class_college": "invalid label (copy of course_college)",
    "enrolled_class_dept": "invalid label (copy of course_dept)",
    "is_cross_college_strict": "built from the invalid course college",
    "is_multi_college_section": "built from the invalid course college",
    "is_cross_college_course": "built from the invalid course college",
    "is_target_population": "built from the invalid course college",
    "is_actual_cross_college": "built from the invalid course college",
    "cross_college_status": "built from the invalid course college",
    "course_code": "unavailable for 79% of rows (exported for one college only)",
    "offering_class": "unavailable for 79% of rows (exported for one college only)",
    "hours": "unavailable for 79% of rows (exported for one college only)",
    "student_class": "unavailable for 79% of rows (exported for one college only)",
    "grade_level": "unavailable for 81% of rows (exported for one college only)",
    "student_id": "identifier", "row_id": "identifier", "section_key": "identifier / split group",
    "section_no": "identifier", "source_file": "bookkeeping", "term_label": "constant in the study term",
    "term_order": "constant", "academic_year": "constant", "semester": "constant",
    "course_college_source": "bookkeeping", "instructor_source": "bookkeeping",
    "mismatch_review_status": "bookkeeping", "flag_same_course_multi_section": "bookkeeping",
    "pass_threshold": "determines pass/fail semantics of the target; kept out to avoid target coupling",
}

MODELS = ["mean_global", "mean_course", "ridge", "catboost", "lightgbm", "mlp"]
FAST_MODELS = ["mean_global", "mean_course", "ridge", "catboost", "lightgbm"]   # used for ablation
SPLITS = ["student", "section"]

# Report per-group metrics only for groups with at least this many rows.
MIN_GROUP_ROWS = 200


@dataclass
class ResearchConfig:
    cleaned_csv: Path = PROJECT_ROOT / "outputs" / "data" / "cleaned_grades.csv"
    raw_csv: Path = PROJECT_ROOT / "IR.csv"           # only for the privacy scan (identifiers to look for)
    private_dir: Path = PROJECT_ROOT / "private"
    output_dir: Path = PROJECT_ROOT / "outputs" / "research"
    n_folds: int = 5
    seed: int = 42
    exclude_zero_grades: bool = True    # primary target policy; zeros are also run as a sensitivity
    quick: bool = False                 # tests: fewer iterations / folds / rows
    catboost_params: dict = field(default_factory=lambda: {
        "iterations": 600, "learning_rate": 0.08, "depth": 6, "l2_leaf_reg": 5.0,
        "loss_function": "RMSE", "nan_mode": "Min", "allow_writing_files": False,
        "thread_count": -1, "verbose": 0})
    lightgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
        "subsample": 0.9, "subsample_freq": 1, "colsample_bytree": 0.9, "verbose": -1, "n_jobs": 8})
    ridge_alpha: float = 1.0
    mlp_params: dict = field(default_factory=lambda: {
        "hidden_layer_sizes": (128, 64), "early_stopping": True, "max_iter": 60,
        "batch_size": 512, "learning_rate_init": 1e-3, "alpha": 1e-4})

    def __post_init__(self):
        if self.quick:
            self.n_folds = 2
            self.catboost_params = {**self.catboost_params, "iterations": 60}
            self.lightgbm_params = {**self.lightgbm_params, "n_estimators": 60}
            self.mlp_params = {**self.mlp_params, "max_iter": 20, "hidden_layer_sizes": (32,)}
