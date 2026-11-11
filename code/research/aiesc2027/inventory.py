"""File inventory of the artefacts this analysis depends on (path, purpose, fields, producing code, status)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .io_data import sha256

R = "outputs/research"
ITEMS = [
    ("IR.csv", "Raw registrar export (86,014 enrolments; identifiers pseudonymised downstream)", "src/ir_pipeline (ingest)", "raw data"),
    ("outputs/data/cleaned_grades.csv", "Cleaned, pseudonymised table; source of the analysis frame", "scripts/preprocess.py", "data"),
    ("outputs/data/split_assignment.csv", "Chronological split of the production pipeline (not used here)", "scripts/preprocess.py", "data"),
    ("research/data.py", "build_frame: 113-2 rows, zero-grade exclusion, feature columns (reused)", "-", "code"),
    ("research/validation.py", "folds3: student-grouped GroupKFold, seed 42 (definition of the folds)", "-", "code"),
    ("research/exploratory/candidates.py", "NGB (NGBoost Normal) and GBM (LightGBM L1) definitions", "-", "code"),
    ("research/exploratory/nn.py", "FT-Transformer (rtdl_revisiting_models) definition", "-", "code"),
    ("research/models.py", "Frozen CatBoost / LightGBM learners and hyper-parameters", "-", "code"),
    ("research/config.py", "Feature sets, excluded columns, CatBoost / LightGBM parameters", "-", "code"),
    (f"{R}/exploratory/oof.npz", "OOF predictions (frozen LightGBM / CatBoost, LightGBM L1, ...) per split", "research/exploratory/run.py", "OOF"),
    (f"{R}/advanced/phase1_oof.npz", "Original frozen-baseline OOFs (reused by the exploratory study)", "research/advanced/phase1.py", "OOF"),
    (f"{R}/exploratory_amendment1/jobs/", "Per-fold test_idx, predictions; NGBoost loc / scale (T3 = student split)", "research/exploratory/amendment1/worker.py", "OOF"),
    (f"{R}/exploratory_amendment1/fit_status.csv", "Fit status, seconds, NGBoost scale multipliers per fold", "research/exploratory/amendment1/run.py", "log"),
    (f"{R}/exploratory/registry.jsonl", "Experiment registry with library versions", "research/exploratory/run.py", "log"),
    (f"{R}/exploratory_amendment1/registry.jsonl", "Experiment registry with library versions", "research/exploratory/amendment1/run.py", "log"),
    (f"{R}/final/comparison_student.csv", "Registered student-split MAE / RMSE / bootstrap CIs", "research/final/consolidate.py", "result"),
    (f"{R}/final/ngboost_uncertainty.csv", "Registered NGBoost interval coverage / width (T1-T5)", "research/final/consolidate.py", "result"),
    (f"{R}/final/reproducibility_manifest.json", "SHA-256 of data / code / outputs, seeds, versions", "research/final/consolidate.py", "manifest"),
    (f"{R}/final/最終研究報告.md", "Final IR report (representative numbers)", "research/final/report_zh.py", "report"),
    (f"{R}/family/interval_coverage.csv", "Split / Mondrian / CQR intervals on course and family splits (CatBoost+text)", "research/family/run_family.py", "result"),
    (f"{R}/oof_predictions.csv", "Earlier single-semester OOF (other feature sets; not used)", "research/run_research.py", "OOF"),
    (f"{R}/chatgpt_chart_data/figure2_six_prediction_sets.csv", "Ablation OOF export (student split; not used)", "research/export_figure2_data.py", "OOF"),
]


def describe(path):
    p = C.ROOT / path
    if not p.exists():
        return {"exists": False}
    if p.is_dir():
        files = sorted(p.glob("*.npz"))
        t3 = [f.name for f in files if "__T3__" in f.name]
        return {"exists": True, "detail": f"{len(files)} npz files; student split (T3): {len(t3)}"}
    info = {"exists": True, "size_bytes": p.stat().st_size, "sha256_16": sha256(p)[:16]}
    if p.suffix == ".csv":
        cols = pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns.tolist()
        info["detail"] = f"{len(cols)} columns: " + ", ".join(cols[:14]) + (" ..." if len(cols) > 14 else "")
    elif p.suffix == ".npz":
        z = np.load(p)
        keys = [k for k in z.files if k.startswith("student")] or z.files
        info["detail"] = f"{len(z.files)} arrays; e.g. " + ", ".join(keys[:12])
    elif p.suffix == ".json":
        info["detail"] = "keys: " + ", ".join(list(json.loads(p.read_text(encoding="utf-8")))[:10])
    return info


def build() -> pd.DataFrame:
    rows = []
    for path, purpose, producer, kind in ITEMS:
        rows.append({"path": path, "kind": kind, "purpose": purpose, "produced_by": producer, **describe(path)})
    return pd.DataFrame(rows)
