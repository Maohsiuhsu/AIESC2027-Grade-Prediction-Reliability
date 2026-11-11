"""Fixed definitions of the AIESC 2027 analysis (set before the risk metrics were computed).

Inputs are existing artefacts of the IR study; outputs go to outputs/aiesc2027/ only.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "aiesc2027"

# ---- inputs (read-only) -------------------------------------------------------------------------------
RAW_CSV = ROOT / "IR.csv"
CLEANED_CSV = ROOT / "outputs" / "data" / "cleaned_grades.csv"
EXPLORATORY_OOF = ROOT / "outputs" / "research" / "exploratory" / "oof.npz"
AMENDMENT_DIR = ROOT / "outputs" / "research" / "exploratory_amendment1"
AMENDMENT_JOBS = AMENDMENT_DIR / "jobs"
AMENDMENT_FIT_STATUS = AMENDMENT_DIR / "fit_status.csv"
FINAL_DIR = ROOT / "outputs" / "research" / "final"
FINAL_COMPARISON_STUDENT = FINAL_DIR / "comparison_student.csv"
FINAL_NGBOOST_UNCERTAINTY = FINAL_DIR / "ngboost_uncertainty.csv"
FINAL_MANIFEST = FINAL_DIR / "reproducibility_manifest.json"
FAMILY_INTERVALS = ROOT / "outputs" / "research" / "family" / "interval_coverage.csv"

# SHA-256 recorded in outputs/research/final/reproducibility_manifest.json (data version check)
EXPECTED_SHA256 = {
    "IR.csv": "e46bce19a2637365bf128f4fdd3ec6af4bd5328b94a24c5ff216ef913e36d321",
    "cleaned_grades.csv": "7e007475da01fc22c618983f01b38ef507dfac23c9d571552268f71ab031cf16",
}

# ---- study population (from the IR study; verified at run time) ---------------------------------------
EXPECTED_RAW_ROWS = 86014
EXPECTED_TERM_ROWS = 85567
EXPECTED_ZERO_ROWS = 2073
EXPECTED_N = 83494
EXPECTED_N_FOLDS = 5
SPLIT = "student"            # unseen-student 5-fold grouped CV
AMENDMENT_UNIT = "T3"        # Amendment 1 unit that corresponds to the student split

# ---- models (all five share rows, folds and features; see inventory) ----------------------------------
# kind "npz": key in exploratory/oof.npz; kind "jobs": Amendment-1 per-fold files <prefix>__T3__f<k>.npz
MODELS: dict[str, dict] = {
    "lightgbm": {"label": "LightGBM", "kind": "npz", "key": "student|frozen:lightgbm:cat",
                 "registry_name": "frozen:lightgbm:cat", "loss": "L2 (squared error)"},
    "catboost": {"label": "CatBoost", "kind": "npz", "key": "student|frozen:catboost:cat",
                 "registry_name": "frozen:catboost:cat", "loss": "L2 (RMSE)"},
    "ft_transformer": {"label": "FT-Transformer", "kind": "jobs", "key": "M2_ft_transformer",
                       "registry_name": "M2_ft_transformer", "loss": "L2 (MSE), grouped early stopping"},
    "ngboost": {"label": "NGBoost", "kind": "jobs", "key": "M3_ngboost",
                "registry_name": "M3_ngboost", "loss": "Normal NLL"},
    "lightgbm_l1": {"label": "LightGBM-L1", "kind": "npz", "key": "student|C5_lightgbm_l1",
                    "registry_name": "C5_lightgbm_l1", "loss": "L1 (absolute error)"},
}
MODEL_ORDER = list(MODELS)
REFERENCE_MODEL = "lightgbm"     # best-MAE frozen baseline of the student split in the IR study

# ---- risk definition ------------------------------------------------------------------------------------
RISK_THRESHOLD = 60.0            # fixed research threshold: actual term grade < 60
# Sensitivity: the official, programme-specific pass threshold column (60 or 70). Evaluation label only.
OFFICIAL_THRESHOLD_COL = "pass_threshold"

# ---- grade ranges (post-hoc evaluation strata; never model inputs) -------------------------------------
GRADE_BANDS = [("0-59", 0.0, 60.0), ("60-69", 60.0, 70.0), ("70-79", 70.0, 80.0),
               ("80-89", 80.0, 90.0), ("90-100", 90.0, 100.0001)]

# ---- uncertainty ----------------------------------------------------------------------------------------
INTERVAL_LEVELS = (0.5, 0.8, 0.9, 0.95)
MONDRIAN_MIN_CALIB = 100         # same rule as research/family/intervals.py
COURSE_CATEGORY_EN = {"系專業必修": "Departmental required", "系專業選修": "Departmental elective",
                      "校共同必修": "University common required", "通識教育": "General education",
                      "院必修": "College required"}

# ---- statistics -----------------------------------------------------------------------------------------
N_BOOT = 5000
SEED = 42
ALPHA = 0.05
# Every metric reported in Tables 2-3 is in the multiplicity family (4 contrasts vs the reference x 9).
FAMILY_METRICS = ["MAE", "RMSE", "precision", "recall", "f1", "specificity", "balanced_accuracy",
                  "pr_auc", "roc_auc"]
LOWER_IS_BETTER = {"MAE", "RMSE"}
# Heuristic practical-importance thresholds, fixed before the risk metrics were computed.
PRACTICAL_THRESHOLDS = {"MAE": 0.5, "RMSE": 0.5, "precision": 0.05, "recall": 0.05, "f1": 0.05,
                        "specificity": 0.05, "balanced_accuracy": 0.05, "pr_auc": 0.05, "roc_auc": 0.05}

# Columns that must never be used as predictors (target or target-derived); checked at run time.
FORBIDDEN_PREDICTORS = {"term_grade", "final_score", "passed", "grade_band", "is_zero_grade",
                        "flag_pass_mismatch", "flag_band_mismatch", "midterm_score", "daily_score",
                        "components_missing_flag", "pass_threshold"}
