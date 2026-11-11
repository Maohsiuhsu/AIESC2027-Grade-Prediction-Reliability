"""Course-name normalisation. Pure string processing: no grades or student data involved."""
from __future__ import annotations

import re
import unicodedata

ORDINALS = "一二三四五六七八九十"
TAG_RE = re.compile(r"\((核|通識)\)")
# section / group markers: "-A班", "-B組", "A班", "(A)", "(B)", trailing single latin letter after CJK
GROUP_RE = [re.compile(r"[-－]?[A-Za-z][班組]$"), re.compile(r"\([A-Za-z]\)$"),
            re.compile(r"(?<=[一-鿿])[A-Z]$")]
ORDINAL_RE = re.compile(rf"\(([{ORDINALS}]+)\)")


def normalize_name(name: str) -> str:
    """NFKC (full-width -> half-width), drop whitespace, lower-case latin letters, drop
    section/group markers and the (核)/(通識) tags. Two sections of the same course map together."""
    s = unicodedata.normalize("NFKC", str(name))
    s = re.sub(r"\s+", "", s)
    s = TAG_RE.sub("", s)
    for rx in GROUP_RE:
        s = rx.sub("", s)
    return s.lower()


def course_family(name: str) -> str:
    """Normalised name without the ordinal: 英文(二) and 英文(四) -> 英文."""
    return ORDINAL_RE.sub("", normalize_name(name)) or normalize_name(name)


def name_flags(name: str) -> dict:
    s = unicodedata.normalize("NFKC", str(name))
    m = ORDINAL_RE.search(s)
    return {"tag_core": int("(核)" in s), "tag_general": int("(通識)" in s),
            "ordinal": ORDINALS.index(m.group(1)[0]) + 1 if m else 0,
            "is_lab": int(bool(re.search(r"實習|實驗|實作", s))),
            "is_project": int(bool(re.search(r"專題", s))),
            "deep_bowl": int("深碗" in s)}
