"""Reproducible course-family groups built from normalised course names only (no grades).

family         = normalised name with the ordinal removed: 英文(二), 英文(四) -> 英文
family_linked  = connected components of families joined by a deterministic name-similarity link:
                 the shorter name (>= 3 chars) is a prefix of the longer and covers >= 50% of it,
                 OR the character-bigram Jaccard similarity is >= 0.6.
                 A deliberately conservative (over-merging) grouping for a stricter hold-out.

The audit lists every link and every family whose member courses disagree on registration
attributes (category, credits): these are the ambiguous assignments a human should review.
Row counts are enrolment counts (registration data), never grades.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

LINK_RULE = {"prefix_min_len": 3, "prefix_min_ratio": 0.5, "jaccard": 0.6}


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def family_links(families: list[str], rule: dict = LINK_RULE) -> pd.DataFrame:
    names = sorted(set(families))
    B = {n: _bigrams(n) for n in names}
    rows = []
    for a, b in itertools.combinations(names, 2):
        s, l = (a, b) if len(a) <= len(b) else (b, a)
        prefix = len(s) >= rule["prefix_min_len"] and l.startswith(s) and len(s) / len(l) >= rule["prefix_min_ratio"]
        jac = len(B[a] & B[b]) / len(B[a] | B[b])
        contained = len(s) >= 2 and s in l
        if prefix or jac >= rule["jaccard"] or contained:
            rows.append({"family_a": a, "family_b": b, "jaccard": round(jac, 3), "prefix_link": prefix,
                         "substring_only": contained and not prefix and jac < rule["jaccard"],
                         "used_for_linked_groups": bool(prefix or jac >= rule["jaccard"])})
    return pd.DataFrame(rows, columns=["family_a", "family_b", "jaccard", "prefix_link", "substring_only",
                                       "used_for_linked_groups"])


def linked_groups(families: list[str], links: pd.DataFrame) -> dict[str, str]:
    """Union-find; each component is named by its lexicographically smallest family (deterministic)."""
    names = sorted(set(families))
    endpoints = set(links["family_a"]) | set(links["family_b"]) if len(links) else set()
    parent = {n: n for n in sorted(set(names) | endpoints)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in links[links["used_for_linked_groups"]].itertuples():
        ra, rb = find(r.family_a), find(r.family_b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {n: find(n) for n in names}   # only families actually present are mapped


def add_groups(frame: pd.DataFrame, links: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = frame.copy()
    fams = sorted(f["course_family"].unique())
    links = family_links(fams) if links is None else links
    f["family_linked"] = f["course_family"].map(linked_groups(fams, links))
    return f, links


def audit_families(frame: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby("course_family")
    t = pd.DataFrame({
        "rows": g.size(),
        "n_courses": g["course_norm"].nunique(),
        "members": g["course_norm"].agg(lambda s: " / ".join(sorted(s.unique()))),
        "categories": g["course_category"].agg(lambda s: " / ".join(sorted(s.dropna().astype(str).unique()))),
        "credits": g["credits"].agg(lambda s: " / ".join(str(v) for v in sorted(s.dropna().unique()))),
        "family_linked": g["family_linked"].first(),
    })
    t["hetero_category"] = t["categories"].str.contains(" / ")
    t["hetero_credits"] = t["credits"].str.contains(" / ")
    linked = pd.concat([links["family_a"], links["family_b"]]) if len(links) else pd.Series(dtype=str)
    t["has_name_link"] = t.index.isin(set(linked))
    t["merged_in_linked_group"] = t.groupby("family_linked")["rows"].transform("size") > 1
    t["ambiguous"] = t["hetero_category"] | t["hetero_credits"] | t["has_name_link"]
    t["reason"] = np.select(
        [t["hetero_category"] & (t["n_courses"] > 1), t["hetero_credits"] & (t["n_courses"] > 1),
         t["hetero_category"] | t["hetero_credits"], t["has_name_link"]],
        ["member courses differ in category", "member courses differ in credits",
         "same course name offered with different category/credits", "name linked to another family"], default="")
    return t.reset_index().sort_values("rows", ascending=False)


def size_bins(frame: pd.DataFrame, col: str = "course_family") -> pd.DataFrame:
    """Family size by enrolment rows and by number of member courses (registration counts only)."""
    rows = frame.groupby(col)["row_id"].transform("size")
    courses = frame.groupby(col)["course_norm"].transform("nunique")
    return pd.DataFrame({
        "family_rows_bin": pd.cut(rows, [0, 99, 999, np.inf], labels=["<100", "100-999", ">=1000"]).astype(str),
        "family_courses_bin": pd.cut(courses, [0, 1, 3, np.inf], labels=["1", "2-3", ">=4"]).astype(str)},
        index=frame.index)
