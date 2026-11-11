"""Assemble the AIESC 2027 Results chapter from the three generated drafts.

Reads only files under outputs/aiesc2027/ and writes only to outputs/aiesc2027/paper/. Tables and figures are
renumbered once, by order of first mention (main) and by section (supplementary); every reference in the text,
captions and notes is rewritten through the same map, and unmapped references abort the run.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "aiesc2027"
PAPER = OUT / "paper"

DRAFT_MAIN = OUT / "results" / "results_en.md"
DRAFT_RISK = OUT / "risk_methods" / "results" / "results_en_4_2_1_4_2_2.md"
DRAFT_CAL = OUT / "calibration_split" / "results" / "results_en_4_4_1.md"

T, CT, RT = OUT / "tables", OUT / "calibration_split" / "tables", OUT / "risk_methods" / "tables"
F, CF, RF = OUT / "figures", OUT / "calibration_split" / "figures", OUT / "risk_methods" / "figures"

# (kind, new, old, placement, section, source stem, new short name)
ITEMS = [
    ("Table", "1", "1", "main", "Methods", T / "table1_dataset", "dataset"),
    ("Table", "2", "2", "main", "4.1", T / "table2_overall", "overall_performance"),
    ("Table", "3", "3", "main", "4.2", T / "table3_risk", "risk_identification"),
    ("Table", "4", "8", "main", "4.2.1", RT / "table8_risk_methods", "risk_methods"),
    ("Table", "5", "9", "main", "4.2.2", RT / "table9_zero_grade_sensitivity", "zero_grade_sensitivity"),
    ("Table", "6", "4", "main", "4.3", T / "table4_bias", "grade_range_bias"),
    ("Table", "7", "5", "main", "4.4", T / "table5_uncertainty", "ngboost_uncertainty"),
    ("Table", "8", "6", "main", "4.4.1", CT / "table6_calibration_split_intervals", "calibration_split_intervals"),
    ("Table", "9", "7", "main", "4.4.1", CT / "table7_probability_recalibration", "probability_recalibration"),
    ("Table", "S1", "1b", "supplementary", "Methods", T / "table1b_folds", "folds"),
    ("Table", "S2", "S1", "supplementary", "4.1-4.2", T / "table_s1_paired_comparisons", "paired_comparisons"),
    ("Table", "S3", "S7a", "supplementary", "4.2.1", RT / "table_s7a_risk_contrasts", "risk_contrasts"),
    ("Table", "S4", "S7b", "supplementary", "4.2.1", RT / "table_s7b_thresholds", "risk_thresholds"),
    ("Table", "S5", "S7c", "supplementary", "4.2.1", RT / "table_s7c_regression_accuracy", "regression_variant_accuracy"),
    ("Table", "S6", "S7d", "supplementary", "4.2.1", RT / "table_s7d_probabilities", "risk_probability_calibration"),
    ("Table", "S7", "S6a", "supplementary", "4.4.1", CT / "table_s_calibration_split_all_levels", "intervals_all_levels"),
    ("Table", "S8", "S6e", "supplementary", "4.4.1", CT / "table_s_calibration_split_by_actual_range",
     "intervals_by_actual_range"),
    ("Table", "S9", "S6f", "supplementary", "4.4.1", CT / "table_s_calibration_split_by_predicted_range",
     "intervals_by_predicted_range"),
    ("Table", "S10", "S6g", "supplementary", "4.4.1", CT / "table_s_calibration_split_by_category", "intervals_by_category"),
    ("Table", "S11", "S6b", "supplementary", "4.4.1", CT / "table_s_point_accuracy", "refit_point_accuracy"),
    ("Table", "S12", "S6c", "supplementary", "4.4.1", CT / "table_s_ngboost_scores", "refit_ngboost_scores"),
    ("Table", "S13", "S6d", "supplementary", "4.4.1", CT / "table_s_reproduction_gate", "reproduction_gate"),
    ("Figure", "1", "1", "main", "Methods", F / "figure1_workflow", "workflow"),
    ("Figure", "2", "2", "main", "4.1", F / "figure2_mae_rmse", "mae_rmse"),
    ("Figure", "3", "3", "main", "4.2", F / "figure3_risk_identification", "risk_identification"),
    ("Figure", "4", "7", "main", "4.2.1-4.2.2", RF / "figure7_risk_methods", "risk_methods"),
    ("Figure", "5", "4", "main", "4.3", F / "figure4_grade_range_bias", "grade_range_bias"),
    ("Figure", "6", "5", "main", "4.4", F / "figure5_ngboost_calibration", "ngboost_calibration"),
    ("Figure", "7", "6", "main", "4.4.1", CF / "figure6_calibration_split", "calibration_split"),
    ("Figure", "S1", "S1", "supplementary", "4.4", F / "figureS1_ngboost_pit", "ngboost_pit"),
    ("Figure", "S2", "S2", "supplementary", "4.4", F / "figureS2_coverage_by_course_category", "coverage_by_category"),
]

FIGURE_CAPTIONS = {
    "1": "Study population and analysis workflow. Left branch: re-verified out-of-fold predictions of the IR study "
         "(Sections 4.1-4.4). Right branch: pre-specified follow-ups with models refitted on 80% of the students of each "
         "training fold, the remaining 20% serving as calibration set (Sections 4.2.1, 4.2.2 and 4.4.1).",
    "2": "Overall prediction error of the five models for unseen students (n = 83,494): (a) MAE and (b) RMSE. Whiskers: "
         "95% student-cluster bootstrap confidence intervals; y-axes truncated.",
    "3": "Identification of enrolments with an actual grade below 60: (a) precision, recall and F1 of the fixed rule "
         "\"predicted grade < 60\"; (b) precision-recall curves using the negative predicted grade as ranking score, with "
         "the NGBoost probability P(Y < 60) shown dashed and the prevalence as reference line.",
    "4": "Risk-oriented decision rules and training objectives (refitted models): (a) recall against the share of "
         "enrolments flagged; (b) recall under the native rule and the two operating points chosen on the calibration "
         "students (n = 83,494); (c) share of zero-grade enrolments flagged at operating point OP1 when the 2,073 "
         "zero grades are included (n = 85,567).",
    "5": "Prediction bias across actual grade ranges: (a) mean signed error (predicted minus actual) and (b) mean "
         "predicted grade against mean actual grade per range. Ranges are post-hoc strata of the actual grade, not model "
         "inputs; dashed line at 60.",
    "6": "Predictive uncertainty of NGBoost (original out-of-fold predictions): (a) empirical minus nominal coverage and "
         "(b) mean interval width across nominal levels; (c) coverage of nominal 90% intervals by actual grade range; "
         "(d) reliability of the predicted probability P(Y < 60). CF denotes post-hoc cross-fold conformal intervals "
         "without a separate calibration set.",
    "7": "Intervals and probabilities with an in-fold calibration set (refitted models): (a) coverage of nominal 90% "
         "intervals by actual grade range; (b) mean width against coverage overall (filled) and for actual grades 0-59 "
         "(open); (c) reliability of P(Y < 60) before and after recalibration.",
    "S1": "Probability integral transform of the NGBoost predictive distributions (original out-of-fold predictions).",
    "S2": "Coverage of nominal 90% NGBoost intervals by course category (original out-of-fold predictions).",
}

PREFACE = """\
> **Scope of the evidence.** Sections 4.1-4.4 re-analyse the existing out-of-fold predictions of the IR study after
> an alignment audit; no model was retrained for them. Sections 4.2.1, 4.2.2 and 4.4.1 report pre-specified
> follow-ups in which models were refitted on 80% of the students of each saved training fold and the remaining 20%
> served as calibration set; their design files were fixed before fitting, and each follow-up has its own
> multiplicity family. The same semester of data was used repeatedly for model selection in the IR study, so all
> results are exploratory and none is an independent or cross-semester validation. Numbers are rounded to three
> decimals."""

REF = re.compile(r"\b(Table|Figure)(s?)\s+(S?\d+)([a-z]?)(?:(\s*[-–]\s*|\s+and\s+)(S?\d+)([a-z]?))?")
LABEL = re.compile(r"\\label\{tab:[^}]*\}")


def build_map():
    m = {}
    for kind, new, old, *_ in ITEMS:
        key = (kind, old)
        if key in m:
            raise ValueError(f"duplicate old label {key}")
        m[key] = new
    return m


MAP = build_map()


def new_label(kind: str, num: str, letter: str) -> str:
    if (kind, num + letter) in MAP:
        return MAP[(kind, num + letter)]
    if (kind, num) in MAP and not MAP[(kind, num)].startswith("S") and not num.startswith("S"):
        return MAP[(kind, num)] + letter
    raise KeyError(f"unmapped reference: {kind} {num}{letter}")


def relabel(text: str) -> str:
    def sub(mt: re.Match) -> str:
        kind, plural, n1, l1, sep, n2, l2 = mt.groups()
        out = f"{kind}{plural} {new_label(kind, n1, l1)}"
        if sep is not None:
            out += sep.replace("-", "–") + new_label(kind, n2, l2)
        return out
    return REF.sub(sub, text)


def strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("<!--")).strip() + "\n"


def split_main(text: str):
    lines = text.splitlines()
    body_start = next(i for i, line in enumerate(lines) if line.startswith("## 4.1"))
    intro = [line for line in lines[1:body_start] if not line.startswith(">")]
    sections, current = {}, None
    for line in lines[body_start:]:
        if line.startswith("## "):
            current = line.split()[1]
            sections[current] = []
        sections[current].append(line)
    return "\n".join(intro).strip(), {k: "\n".join(v).strip() for k, v in sections.items()}


def assemble_text() -> str:
    intro, sec = split_main(DRAFT_MAIN.read_text(encoding="utf-8"))
    if list(sec) != ["4.1", "4.2", "4.3", "4.4", "4.5"]:
        raise ValueError(f"unexpected main sections: {list(sec)}")
    if sec["4.2"].count("(Figure 5d)") != 1:
        raise ValueError("expected exactly one forward reference '(Figure 5d)' in Section 4.2")
    sec["4.2"] = sec["4.2"].replace("(Figure 5d)", "(Section 4.4)")
    risk = strip_comments(DRAFT_RISK.read_text(encoding="utf-8")).strip()
    cal = strip_comments(DRAFT_CAL.read_text(encoding="utf-8")).strip()
    parts = ["# 4. Results", PREFACE, intro, sec["4.1"], sec["4.2"], risk, sec["4.3"], sec["4.4"], cal, sec["4.5"]]
    return relabel("\n\n".join(parts)) + "\n"


def check_order(text: str):
    seen = {"Table": [], "Figure": []}
    for mt in REF.finditer(text):
        kind = mt.group(1)
        for num in (mt.group(3), mt.group(6)):
            if num and not num.startswith("S") and num not in seen[kind]:
                seen[kind].append(num)
    problems = []
    for kind, order in seen.items():
        nums = [int(n) for n in order]
        expected = sorted(int(new) for k, new, _o, place, *_ in ITEMS if k == kind and place == "main" and new != "1")
        if nums != expected:
            problems.append(f"{kind} first-mention order {nums} != {expected}")
    return problems


def stem(kind: str, new: str, short: str) -> str:
    num = f"S{int(new[1:]):02d}" if new.startswith("S") else f"{int(new):02d}"
    return f"{kind}{num}_{short}"


def copy_tables():
    rows = []
    out_dir = PAPER / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, new, old, place, section, src, short in ITEMS:
        if kind != "Table":
            continue
        name = stem(kind, new, short)
        md = src.with_suffix(".md").read_text(encoding="utf-8")
        old_caption = f"**Table {old}."
        if not md.startswith(old_caption):
            raise ValueError(f"{src.name}: caption does not start with {old_caption!r}")
        (out_dir / f"{name}.md").write_text(relabel(md), encoding="utf-8")
        tex = src.with_suffix(".tex")
        if tex.exists():
            body = relabel(tex.read_text(encoding="utf-8"))
            body = LABEL.sub(lambda _m: f"\\label{{tab:{name}}}", body)
            (out_dir / f"{name}.tex").write_text(body, encoding="utf-8")
        shutil.copyfile(src.with_suffix(".csv"), out_dir / f"{name}.csv")
        full = src.parent / f"{src.name}_full_precision.csv"
        if full.exists():
            shutil.copyfile(full, out_dir / f"{name}_full_precision.csv")
        rows.append((kind, new, old, place, section, src, name))
    return rows


def copy_figures():
    rows = []
    out_dir = PAPER / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, new, old, place, section, src, short in ITEMS:
        if kind != "Figure":
            continue
        name = stem(kind, new, short)
        for ext in (".pdf", ".png", ".csv"):
            shutil.copyfile(src.with_suffix(ext), out_dir / f"{name}{ext}")
        rows.append((kind, new, old, place, section, src, name))
    return rows


def write_catalogue(table_rows, figure_rows):
    lines = ["# Tables and figures (AIESC 2027 Results)", "",
             "Numbering follows the order of first mention in `results_en_full.md` (Table 1 and Figure 1 belong to "
             "the Methods section). Supplementary items are grouped by section.", "", "## Main tables", ""]
    for kind, new, *_rest, name in table_rows:
        if not new.startswith("S"):
            lines += [(PAPER / "tables" / f"{name}.md").read_text(encoding="utf-8").strip(), ""]
    lines += ["## Figures", ""]
    for kind, new, *_rest, name in figure_rows:
        lines += [f"![Figure {new}](figures/{name}.png)", "", f"**Figure {new}.** {FIGURE_CAPTIONS[new]}", ""]
    lines += ["## Supplementary tables", ""]
    for kind, new, *_rest, name in table_rows:
        if new.startswith("S"):
            caption = (PAPER / "tables" / f"{name}.md").read_text(encoding="utf-8").splitlines()[0]
            lines.append(f"- {caption} (`tables/{name}.md`)")
    (PAPER / "tables_and_figures.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_map(rows):
    with open(PAPER / "numbering_map.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["new_label", "old_label", "placement", "section", "source_file", "paper_file"])
        for kind, new, old, place, section, src, name in rows:
            w.writerow([f"{kind} {new}", f"{kind} {old}", place, section, str(src.relative_to(ROOT)), name])


def residual_old_labels(text: str, rows) -> list[str]:
    """Labels that only exist in the old scheme must not survive (e.g. Table S7a, Table 1b)."""
    new_labels = {f"{k} {n}" for k, n, *_ in rows}
    found = set()
    for mt in REF.finditer(text):
        kind = mt.group(1)
        for num, letter in ((mt.group(3), mt.group(4)), (mt.group(6), mt.group(7))):
            if num is None:
                continue
            base = f"{kind} {num}{letter}" if num.startswith("S") else f"{kind} {num}"
            if base not in new_labels:
                found.add(base)
    return sorted(found)


def safety_checks(before) -> list[str]:
    import pandas as pd
    from ir_pipeline.privacy import privacy_gate, raw_identifiers

    from research.aiesc2027 import config as C
    from research.aiesc2027 import integrity

    ids, names = raw_identifiers([pd.read_csv(C.RAW_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")])
    gate = privacy_gate([PAPER], ids, names)
    cmp_ = integrity.compare(before, integrity.snapshot())
    (PAPER / "validation.json").write_text(json.dumps({"privacy_gate": gate, "integrity": cmp_}, ensure_ascii=False,
                                                      indent=2), encoding="utf-8")
    print(f"privacy gate: {gate['status']} ({gate['n_findings']} findings); "
          f"integrity: {cmp_['status']} ({cmp_['files_checked']} files)")
    failed = []
    if gate["status"] != "PASS":
        failed.append(f"privacy gate {gate['status']}")
    if cmp_["status"] != "PASS":
        failed.append(f"integrity {cmp_['status']}")
    return failed


def main() -> int:
    from research.aiesc2027 import integrity

    before = integrity.snapshot()
    PAPER.mkdir(parents=True, exist_ok=True)
    text = assemble_text()
    problems = check_order(text)
    table_rows, figure_rows = copy_tables(), copy_figures()
    rows = table_rows + figure_rows
    problems += [f"residual old label in text: {x}" for x in residual_old_labels(text, rows)]
    for kind, new, *_rest, name in table_rows:
        problems += [f"{name}: residual old label {x}"
                     for x in residual_old_labels((PAPER / "tables" / f"{name}.md").read_text(encoding="utf-8"), rows)]
    for heading in ("## 4.1", "## 4.2", "### 4.2.1", "### 4.2.2", "## 4.3", "## 4.4", "### 4.4.1", "## 4.5"):
        if text.count(f"\n{heading} ") != 1:
            problems.append(f"heading {heading} appears {text.count(chr(10) + heading + ' ')} times")
    if "<!--" in text or "draft for AIESC" in text:
        problems.append("draft markers left in text")
    if problems:
        print("\n".join(problems))
        return 1
    (PAPER / "results_en_full.md").write_text(text, encoding="utf-8")
    write_catalogue(table_rows, figure_rows)
    write_map(rows)
    print(f"wrote {PAPER.relative_to(ROOT)}: {len(table_rows)} tables, {len(figure_rows)} figures, "
          f"{len(text.split())} words")
    failed = safety_checks(before)
    if failed:
        print("\n".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
