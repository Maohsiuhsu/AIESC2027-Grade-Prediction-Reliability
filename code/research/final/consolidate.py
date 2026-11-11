"""Final consolidation of the exploratory model comparison. READ-ONLY with respect to every earlier
output: no model is trained, no existing file is changed. New artefacts go to outputs/research/final/.

Inputs
  outputs/research/exploratory/            original pre-registered study (frozen baselines + 13 candidates)
  outputs/research/exploratory_amendment1/ Amendment 1 (TabM, FT-Transformer, NGBoost)
  outputs/research/advanced/registry.jsonl frozen-baseline costs
Outputs
  comparison_<split>.csv, comparison_single_course_families.csv, zero_grade_sensitivity.csv,
  model_inventory.csv, reproducibility_manifest.json, 最終研究報告.md

The only new statistic is a program-wide Bonferroni-adjusted CI across ALL exploratory comparisons of
both registrations, computed from the saved out-of-fold predictions (same bootstrap as the originals).
The originally registered adjusted CIs are kept unchanged alongside it.

Usage: .venv/bin/python research/final/consolidate.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd  # noqa: I001
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.advanced.common import SPLIT_GROUP, load_frames  # noqa: E402
from research.config import TARGET, ResearchConfig  # noqa: E402
from research.exploratory.run import paired_bootstrap  # noqa: E402
from research.family.groups import size_bins  # noqa: E402

R = ROOT / "outputs" / "research"
EXP, AMD, ADV = R / "exploratory", R / "exploratory_amendment1", R / "advanced"
OUT = R / "final"
SPLITS = ["student", "section", "course", "family"]
AMD_UNIT = {"student": "T3", "section": "T4", "course": "T5", "family": "T1"}
AMD_MODELS = ["M1_tabm", "M2_ft_transformer", "M3_ngboost"]
NAME = {
    "frozen:catboost:cat": "凍結 CatBoost（類別）", "frozen:lightgbm:cat": "凍結 LightGBM（類別）",
    "M1_tabm": "TabM（官方 tabm）", "M2_ft_transformer": "FT-Transformer（rtdl_revisiting_models）", "M3_ngboost": "NGBoost",
    "A1_tabm": "TabM", "A2_ft_transformer": "FT-Transformer", "C2_ngboost": "NGBoost",
    "B1_hte_lightgbm": "階層目標編碼＋LightGBM", "B2_hte_catboost": "階層目標編碼＋CatBoost",
    "B3_hte_plus_cat_lightgbm": "類別＋階層目標編碼（LightGBM）", "B4_missing_lightgbm": "缺漏／新穎指標（LightGBM）",
    "B5_missing_catboost": "缺漏／新穎指標（CatBoost）", "C1_ebm": "EBM", "C3_catboost_mae": "CatBoost MAE 損失",
    "C4_catboost_huber": "CatBoost Huber", "C5_lightgbm_l1": "LightGBM L1", "C6_lightgbm_huber": "LightGBM Huber",
}
SOURCE = {**{k: "原始登錄" for k in NAME if k[:2] in ("B1", "B2", "B3", "B4", "B5", "C1", "C3", "C4", "C5", "C6")},
          **{k: "修正案一" for k in AMD_MODELS}, "frozen:catboost:cat": "凍結基準", "frozen:lightgbm:cat": "凍結基準"}


def sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _reg(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


# ---- gather out-of-fold predictions (read-only) -------------------------------------------------------
def gather(frame: pd.DataFrame) -> tuple[dict, dict]:
    oofs, status = {}, {}
    z = np.load(EXP / "oof.npz")
    for k in z.files:
        split, name = k.split("|")
        oofs[(split, name)] = z[k]
    st = json.loads((EXP / "run_status.json").read_text())
    for k, v in st.items():
        name, split = k.split("|")
        status[(split, name)] = v
    fs = pd.read_csv(AMD / "fit_status.csv")
    n = len(frame)
    for m in AMD_MODELS:
        for split, unit in AMD_UNIT.items():
            g = fs[(fs["model"] == m) & (fs["unit"] == unit)]
            done = g[g["status"] == "complete"]
            if len(done) == 5:
                pred = np.full(n, np.nan)
                for f in range(5):
                    zz = np.load(AMD / "jobs" / f"{m}__{unit}__f{f}.npz")
                    pred[zz["test_idx"]] = zz["pred"]
                oofs[(split, m)] = pred
                status[(split, m)] = "complete"
            else:
                status[(split, m)] = "incomplete_budget" if len(g) else "not_run"
    return oofs, status


def costs() -> pd.DataFrame:
    rows = []
    for r in _reg(ADV / "registry.jsonl"):
        if r.get("phase") == "phase1" and r.get("status") == "success" and r.get("feature_set") == "cat" and r["split"] in SPLITS:
            rows.append({"system": f"frozen:{r['learner']}:cat", "split": r["split"], "wall_seconds": r["seconds"],
                         "cpu_seconds": None, "source": "advanced/registry.jsonl"})
    for r in _reg(EXP / "registry.jsonl"):
        if r.get("phase") == "candidates" and r.get("status") == "success":
            rows.append({"system": r["experiment"].replace("[retry]", ""), "split": r["split"], "wall_seconds": r["seconds"],
                         "cpu_seconds": None, "source": "exploratory/registry.jsonl"})
    fs = pd.read_csv(AMD / "fit_status.csv")
    inv = {v: k for k, v in AMD_UNIT.items()}
    for (m, unit), g in fs.groupby(["model", "unit"]):
        if unit in inv:
            wall = g["wall_seconds"].fillna(g["seconds"]).fillna(0)       # a killed fit has only `seconds`
            rows.append({"system": m, "split": inv[unit], "wall_seconds": round(float(wall.sum()), 1),
                         "cpu_seconds": round(float(g["cpu_seconds"].fillna(0).sum()), 1), "source": "exploratory_amendment1/fit_status.csv"})
    return pd.DataFrame(rows).groupby(["system", "split"], as_index=False).agg(
        wall_seconds=("wall_seconds", "sum"), cpu_seconds=("cpu_seconds", "sum"), cost_source=("source", "first"))


def comparisons(frame, oofs, status, n_boot=5000, seed=42):
    y = frame[TARGET].to_numpy()
    single = (size_bins(frame)["family_courses_bin"] == "1").to_numpy()
    exp_c = pd.read_csv(EXP / "contrasts.csv")
    amd_c = pd.read_csv(AMD / "contrasts.csv")
    cands = sorted({n for (_, n) in oofs if not n.startswith("frozen:")})
    # program-wide multiplicity: every (candidate, comparison) with complete predictions
    pops = [(s, "all") for s in SPLITS] + [("family", "single-course families")]
    m_total = sum(1 for c in cands for (s, p) in pops if (s, c) in oofs)
    alpha_prog = 0.05 / m_total
    rows = []
    for split, pop in pops:
        mask = single if pop != "all" else np.ones(len(y), bool)
        base = {b: oofs[(split, b)] for b in ("frozen:catboost:cat", "frozen:lightgbm:cat")}
        comp = min(base, key=lambda b: np.mean(np.abs(y - base[b])))   # the registered comparator rule
        for name in list(base) + cands:
            st = status.get((split, name), "complete") if not name.startswith("frozen") else "complete"
            rec = {"split": split, "population": pop, "system": name, "source": SOURCE.get(name, ""), "status": st,
                   "comparator": comp}
            if (split, name) not in oofs:
                rows.append(rec)
                continue
            o = oofs[(split, name)]
            rec.update({"n": int(mask.sum()), "MAE": round(float(np.mean(np.abs(y[mask] - o[mask]))), 4),
                        "RMSE": round(float(np.sqrt(np.mean((y[mask] - o[mask]) ** 2))), 4)})
            if not name.startswith("frozen"):
                sub = frame[mask].reset_index(drop=True)
                c = paired_bootstrap(sub, base[comp][mask], o[mask], SPLIT_GROUP[split], n_boot, (0.05, alpha_prog), seed)
                rec.update({"mae_diff": c["mae_diff"], "ci_low_95": c["ci_low_95"], "ci_high_95": c["ci_high_95"],
                            "ci_low_program_adj": c["ci_low_adj"], "ci_high_program_adj": c["ci_high_adj"]})
                # the originally registered adjusted CI (unchanged)
                src = amd_c if name in AMD_MODELS else exp_c
                key = (src["candidate"] == name) & (src["split"] == split) & (src["population"] == pop)
                if name in AMD_MODELS:
                    key &= src["unit"] == AMD_UNIT[split]
                if key.any():
                    r0 = src[key].iloc[0]
                    rec.update({"ci_low_registered_adj": r0["ci_low_adj"], "ci_high_registered_adj": r0["ci_high_adj"],
                                "registered_adjustment": "0.05/15 (修正案一)" if name in AMD_MODELS else "0.05/65 (原始登錄)"})
            rows.append(rec)
    return pd.DataFrame(rows), alpha_prog, m_total


def zero_table() -> pd.DataFrame:
    z1 = pd.read_csv(EXP / "zero_sensitivity.csv").assign(source="原始登錄")
    m = pd.read_csv(AMD / "metrics.csv")
    m = m[m["unit"] == "T2"]
    rows = []
    for sysname, g in m.groupby("system"):
        d = g.set_index("population")
        rows.append({"split": "family", "system": sysname, "n": int(d.loc["all", "n"]), "zeros": int(d.loc["zero-grade rows", "n"]),
                     "MAE_all": d.loc["all", "MAE"], "MAE_nonzero": d.loc["non-zero rows", "MAE"],
                     "MAE_zero": d.loc["zero-grade rows", "MAE"], "RMSE_all": d.loc["all", "RMSE"],
                     "source": "修正案一" if sysname in AMD_MODELS else "修正案一（凍結基準重算）"})
    return pd.concat([z1, pd.DataFrame(rows)], ignore_index=True)


def inventory() -> pd.DataFrame:
    X = lambda p: "✔" if (ROOT / p).exists() else "✘"  # noqa: E731
    rows = [
        ("正式流程 CatBoost（歷史特徵）", "src/ir_pipeline, scripts/train.py", "outputs/models/grade_pipeline.joblib",
         "次學年預測流程", "UNSUPPORTED；指標為探索性", "outputs/reports/model_report.md"),
        ("凍結 CatBoost／LightGBM（類別特徵）", "research/advanced/course_repr.py (RepModel, keep_id)", "—（OOF 保存）",
         "所有探索性比較之基準", "探索性基準", "outputs/research/advanced/phase1_oof.npz"),
        ("課程名稱表示 CatBoost（REF）", "research/advanced/course_repr.py", "outputs/research/advanced/models/*.joblib",
         "未見課程／家族", "探索性；對單一課程家族無可靠優勢", "outputs/research/family/未見課程泛化研究報告.md"),
        ("專家模型、FM、OOF 整合、Optuna", "research/advanced/", "—", "進階研究", "探索性；未超越最佳既有基準", "outputs/research/advanced/進階研究報告.md"),
        ("後設資料＋備援模型", "research/metadata/fallback.py", "—", "教學大綱擴充", "NO-GO", "outputs/research/metadata_fallback/最終報告_後設資料決策.md"),
        ("階層目標編碼／缺漏指標（B1–B5）", "research/exploratory/encoders.py, candidates.py", "—", "原始登錄 B", "探索性；無訊號", "outputs/research/exploratory/探索性研究報告.md"),
        ("穩健損失（C3–C6）", "research/exploratory/candidates.py", "—", "原始登錄 C", "C3 有探索性訊號（損失／指標對齊）", "outputs/research/exploratory/探索性研究報告.md"),
        ("EBM（C1）", "research/exploratory/candidates.py", "—", "原始登錄 C", "未完成（超出單次上限）", "outputs/research/exploratory/探索性研究報告.md"),
        ("TabM", "research/exploratory/nn.py (官方 tabm)", "—", "修正案一", "未完成：僅 T1、T2 完成（預算用盡，非模型失敗）", "outputs/research/exploratory_amendment1/修正案一報告.md"),
        ("FT-Transformer", "research/exploratory/nn.py (rtdl_revisiting_models)", "—", "修正案一", "完成；探索性", "outputs/research/exploratory_amendment1/修正案一報告.md"),
        ("NGBoost", "research/exploratory/candidates.py, amendment1/worker.py", "—", "修正案一", "完成；探索性（含不確定性）", "outputs/research/exploratory_amendment1/修正案一報告.md"),
    ]
    df = pd.DataFrame(rows, columns=["模型", "程式", "已存模型", "用途", "狀態", "報告"])
    df["報告存在"] = df["報告"].map(X)
    return df


def manifest(alpha_prog, m_total) -> dict:
    code = sorted([p for d in ("src", "scripts", "research", "tests") for p in (ROOT / d).rglob("*.py")]) + \
        [ROOT / "research" / "exploratory" / "prereg.json", ROOT / "research" / "exploratory" / "amendment1" / "amendment.json",
         ROOT / "requirements.txt", ROOT / "run_pipeline.sh"]
    outputs = sorted(p for d in ("exploratory", "exploratory_amendment1") for p in (R / d).glob("*")
                     if p.is_file() and p.suffix in (".csv", ".json", ".npz", ".md", ".jsonl"))
    vers = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "catboost", "lightgbm", "optuna", "ngboost", "interpret", "torch",
                "tabm", "rtdl_revisiting_models", "sentence_transformers"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "installed")
        except Exception:  # noqa: BLE001
            vers[mod] = "not importable in this process"
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "EXPLORATORY; not independently validated",
        "data": {"IR.csv_sha256": sha(ROOT / "IR.csv"),
                 "cleaned_grades_sha256": sha(ROOT / "outputs" / "data" / "cleaned_grades.csv"),
                 "term": "113-2", "primary_rows": 83494, "zero_inclusive_rows": 85567},
        "registrations": {
            "original_prereg_sha256": sha(ROOT / "research" / "exploratory" / "prereg.json"),
            "original_prereg_lock": json.loads((EXP / "prereg.lock.json").read_text()),
            "amendment1_sha256": sha(ROOT / "research" / "exploratory" / "amendment1" / "amendment.json"),
            "amendment1_lock": json.loads((AMD / "amendment.lock.json").read_text()),
        },
        "seeds": {"folds": 42, "models": 42, "bootstrap": 42, "n_boot": 5000},
        "program_wide_adjustment": {"comparisons": m_total, "alpha": alpha_prog},
        "hardware": {"cpu": cpu, "threads": 32, "gpu_used": False, "note": "GPUs were occupied by other users' jobs"},
        "environment": {"venv": ".venv (system site-packages)", "torch_note": "LD_LIBRARY_PATH=$HOME/anaconda3/lib when torch is imported before pandas", "versions": vers},
        "commands": [
            ".venv/bin/python scripts/preprocess.py",
            ".venv/bin/python research/advanced/run_advanced.py --phase 1",
            ".venv/bin/python research/exploratory/run.py",
            ".venv/bin/python research/exploratory/amendment1/run.py",
            ".venv/bin/python research/final/consolidate.py",
            ".venv/bin/python -m pytest -q tests",
        ],
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code},
        "exploratory_outputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in outputs},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = {str(p): sha(p) for p in sorted(list(EXP.glob("*")) + list(AMD.glob("*"))) if p.is_file()}
    frame, _, _ = load_frames(ResearchConfig())
    oofs, status = gather(frame)
    comp, alpha_prog, m_total = comparisons(frame, oofs, status)
    cost = costs()
    comp = comp.merge(cost, on=["system", "split"], how="left")
    for split in SPLITS:
        comp[(comp["split"] == split) & (comp["population"] == "all")].to_csv(OUT / f"comparison_{split}.csv", index=False, encoding="utf-8-sig")
    comp[comp["population"] == "single-course families"].to_csv(OUT / "comparison_single_course_families.csv", index=False, encoding="utf-8-sig")
    zero_table().to_csv(OUT / "zero_grade_sensitivity.csv", index=False, encoding="utf-8-sig")
    inventory().to_csv(OUT / "model_inventory.csv", index=False, encoding="utf-8-sig")
    unc = pd.read_csv(AMD / "ngboost_uncertainty.csv")
    unc.to_csv(OUT / "ngboost_uncertainty.csv", index=False, encoding="utf-8-sig")
    n_cv = sum(1 for d in ("advanced", "family", "metadata", "metadata_fallback", "exploratory", "exploratory_amendment1")
               for r in _reg(R / d / "registry.jsonl") if r.get("status") in ("success", "complete"))
    man = manifest(alpha_prog, m_total)
    man["program_scale"] = {"logged_cross_validation_runs_or_fits": n_cv,
                            "note": "count of successful CV runs / fits recorded in the research registries (lower bound; earlier "
                                    "single-semester studies had no registry)"}
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    from research.final.report_zh import write_report
    write_report(OUT, alpha_prog, m_total, n_cv)
    after = {str(p): sha(p) for p in sorted(list(EXP.glob("*")) + list(AMD.glob("*"))) if p.is_file()}
    assert before == after, "an existing output changed during consolidation"
    print(f"consolidated: {m_total} comparisons, program-wide alpha {alpha_prog:.5f}; existing outputs unchanged")


if __name__ == "__main__":
    main()
