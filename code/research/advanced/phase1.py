"""Phase 1: course representations.

Feature sets (all registration-time; student attributes = college, department, division, programme):
  cat              existing categorical course + student features (baseline; includes course_key)
  cat+tfidf        + character n-gram TF-IDF (SVD)
  cat+embed        + frozen multilingual embedding (PCA)
  cat+tfidf+embed
  tfidf / embed / tfidf+embed   course_key REMOVED: the representation replaces the course id
Learners: CatBoost, LightGBM. Splits: student, section, course (identical to earlier experiments)
and, as a stricter unseen-course test, course-family (英文(二) and 英文(四) held out together).
"""
from __future__ import annotations

import pandas as pd

from research.config import ResearchConfig

from .common import PRIMARY_SPLITS, Registry, contrast, cross_validate
from .course_repr import RepModel

FEATURE_SETS = {
    "cat": ((), True), "cat+tfidf": (("tfidf",), True), "cat+embed": (("embed",), True),
    "cat+tfidf+embed": (("tfidf", "embed"), True),
    "tfidf": (("tfidf",), False), "embed": (("embed",), False), "tfidf+embed": (("tfidf", "embed"), False),
}
LEARNERS = ["catboost", "lightgbm"]
SPLITS = PRIMARY_SPLITS + ["family"]


def run(frame: pd.DataFrame, cfg: ResearchConfig, reg: Registry, log=print,
        splits=SPLITS, feature_sets=FEATURE_SETS, learners=LEARNERS) -> dict:
    rows, cis, oofs = [], [], {}
    for split in splits:
        for learner in learners:
            for fs, (text, keep_id) in feature_sets.items():
                def make(_tr, text=text, keep_id=keep_id, learner=learner):
                    return RepModel(learner, text, keep_id, cfg=cfg)
                r = cross_validate(frame, split, make, cfg, reg, "phase1", f"{learner}:{fs}",
                                   extra={"learner": learner, "feature_set": fs, "text": list(text), "keep_id": keep_id})
                rows.append({"split": split, "learner": learner, "feature_set": fs, "status": r["status"],
                             "MAE": round(r.get("MAE", float("nan")), 4), "RMSE": round(r.get("RMSE", float("nan")), 4),
                             "seconds": r.get("seconds")})
                log(f"  P1 {split:8s} {learner:9s} {fs:16s} {r['status']:7s} MAE {r.get('MAE', float('nan')):.3f}")
                if r["status"] == "success":
                    oofs[(split, learner, fs)] = r["oof"]
        for learner in learners:
            base = oofs.get((split, learner, "cat"))
            if base is None:
                continue
            for fs in feature_sets:
                if fs == "cat" or (split, learner, fs) not in oofs:
                    continue
                c = contrast(frame, split, base, oofs[(split, learner, fs)], cfg.seed)
                cis.append({"split": split, "learner": learner, "baseline": "cat", "candidate": fs, **c})
    res = {"results": pd.DataFrame(rows), "contrasts": pd.DataFrame(cis), "oofs": oofs}
    res["gate"] = gate(res["contrasts"])
    return res


def gate(cis: pd.DataFrame) -> dict:
    """Pass when some representation measurably improves on the categorical baseline (same learner)
    on at least one PRIMARY split. The unseen-course answer is reported separately."""
    prim = cis[cis["split"].isin(PRIMARY_SPLITS)] if len(cis) else cis
    winners = prim[prim["improves"]] if len(prim) else prim
    best = winners.sort_values("mae_diff", ascending=False).head(1)
    unseen = cis[cis["split"].isin(["course", "family"])] if len(cis) else cis
    return {"passed": bool(len(winners)),
            "winners": winners[["split", "learner", "candidate", "mae_diff", "ci_low", "ci_high"]].to_dict("records"),
            "best": best.to_dict("records")[0] if len(best) else None,
            "unseen_course_improvements": unseen[unseen["improves"]][["split", "learner", "candidate", "mae_diff",
                                                                      "ci_low", "ci_high"]].to_dict("records")
            if len(unseen) else []}
