"""Tests of the Results-chapter assembler (renumbering only; no data access)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.aiesc2027 import paper_package as P  # noqa: E402


def test_map_is_a_bijection_per_kind():
    for kind in ("Table", "Figure"):
        new = [n for k, n, *_ in P.ITEMS if k == kind]
        old = [o for k, _n, o, *_ in P.ITEMS if k == kind]
        assert len(set(new)) == len(new) and len(set(old)) == len(old)
    assert set(P.FIGURE_CAPTIONS) == {n for k, n, *_ in P.ITEMS if k == "Figure"}


def test_relabel_is_single_pass_and_keeps_panels():
    text = "Table 8 and Figure 7a; Table 4 and Figure 5d; Tables 6-7; Table S6b; Table S7a; Table 1b; Figure S1."
    assert P.relabel(text) == ("Table 4 and Figure 4a; Table 6 and Figure 6d; Tables 8–9; Table S11; Table S3; "
                               "Table S1; Figure S1.")


def test_unmapped_reference_aborts():
    with pytest.raises(KeyError):
        P.relabel("see Table S6")
    with pytest.raises(KeyError):
        P.relabel("see Table 12")
