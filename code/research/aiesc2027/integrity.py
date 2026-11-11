"""SHA-256 snapshot of every pre-existing artefact, taken before and after the run, to prove that no
existing data, code or result (IR study, thesis, earlier outputs) was modified."""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import config as C

PROTECTED = ["IR.csv", "outputs", "research", "src", "scripts", "tests", "thesis", "private",
             "requirements.txt", "run_pipeline.sh", "README.md"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", "aiesc2027"}


def _files():
    for name in PROTECTED:
        p = C.ROOT / name
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (SKIP_PARTS & set(f.relative_to(C.ROOT).parts)):
                    yield f


def snapshot() -> dict[str, str]:
    out = {}
    for f in _files():
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[str(f.relative_to(C.ROOT))] = h.hexdigest()
    return out


def compare(before: dict, after: dict) -> dict:
    changed = sorted(k for k in before if k in after and before[k] != after[k])
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    return {"files_checked": len(before), "changed": changed, "removed": removed, "added_outside_aiesc": added,
            "status": "PASS" if not (changed or removed or added) else "FAIL"}
