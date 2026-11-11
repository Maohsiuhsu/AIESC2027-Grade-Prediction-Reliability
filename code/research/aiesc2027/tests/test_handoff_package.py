"""Unit tests of the ChatGPT hand-off helpers (formatting, templates, MASTER rows); no real data needed."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.aiesc2027 import handoff_package as P  # noqa: E402
from research.aiesc2027 import handoff_values as H  # noqa: E402


def _v(value, fmt):
    return H.Value("k", value, fmt, "src.csv", "all rows", "col", "test").text()


def test_value_formats():
    assert _v(83494, "int") == "83,494"
    assert _v(0.12345, "f3") == "0.123"
    assert _v(-0.0001, "f3") == "0.000"
    assert _v(0.0234, "s3") == "+0.023"
    assert _v(-0.0004, "s3") == "0.000"
    assert _v(0.0791, "pct1") == "7.9%"
    assert _v(0.0004, "p") == "< 0.001"
    assert _v(0.0139, "p") == "0.014"
    with pytest.raises(ValueError):
        _v(1.0, "unknown")


def test_registry_rejects_duplicate_keys():
    r = H.Registry()
    r.add("a", 1.0, "f3", "x.csv", "all rows", "c", "d")
    with pytest.raises(KeyError):
        r.add("a", 2.0, "f3", "x.csv", "all rows", "c", "d")


def test_fill_requires_every_placeholder():
    text, used = P.fill("MAE {{a}} vs {{b}}; again {{a}}", {"a": "9.385", "b": "9.031", "c": "unused"})
    assert text == "MAE 9.385 vs 9.031; again 9.385"
    assert used == {"a", "b"}
    with pytest.raises(KeyError):
        P.fill("{{missing}}", {"a": "1"})


def test_master_row_uses_na_not_zero():
    r = P.row(experiment_id="E1", mae=9.38512, recall=0.0, pr_auc=np.nan)
    assert set(r) == set(P.MASTER_COLUMNS)
    assert r["MAE"] == "9.385"
    assert r["Recall"] == "0.000"
    assert r["PR-AUC"] == "NA"
    assert r["RMSE"] == "NA" and r["Confidence Interval"] == "NA"
    with pytest.raises(KeyError):
        P.row(not_a_column=1)


def test_master_has_required_columns():
    required = ["Experiment ID", "Experiment Type", "Model", "Training Configuration", "Evaluation Split", "Dataset Version",
                "Sample Size", "MAE", "RMSE", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "Calibration Metric",
                "Prediction Interval Coverage", "Prediction Interval Width", "Baseline Model", "Difference",
                "Confidence Interval", "Statistical Status", "Completion Status", "Source File"]
    assert not set(required) - set(P.MASTER_COLUMNS)


def test_signed_formatting():
    assert P.s3(0.241) == "+0.241" and P.s3(-0.0001) == "0.000"
    assert P.sci(-0.389, -0.319) == "[-0.389, -0.319]"


def test_templates_only_use_word_placeholders():
    for tpl in sorted(P.TEMPLATES.glob("*.md")):
        text = tpl.read_text(encoding="utf-8")
        assert text.count("{{") == text.count("}}"), tpl.name
        assert not re.findall(r"\{\{[^}]*\W[^}]*\}\}", text), tpl.name
