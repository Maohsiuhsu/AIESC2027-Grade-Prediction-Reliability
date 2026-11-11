"""Compute frozen multilingual sentence embeddings for every normalised course name.

Runs as a separate step (torch is only imported here) and writes a cache that the experiments
read. Only course names are embedded: no student data, no grades. The model is frozen (no
fine-tuning), so there is nothing to fit and nothing that can see a test fold.

Usage (torch needs Anaconda's libstdc++ on this machine):
    LD_LIBRARY_PATH=$HOME/anaconda3/lib .venv/bin/python research/advanced/embed_courses.py \
        [--model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.advanced.names import normalize_name  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE = ROOT / "outputs" / "research" / "advanced" / "cache"


def cache_path(model: str) -> Path:
    return CACHE / f"course_embeddings__{model.replace('/', '__')}.npz"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cleaned", type=Path, default=ROOT / "outputs" / "data" / "cleaned_grades.csv")
    a = p.parse_args(argv)
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    names = pd.read_csv(a.cleaned, usecols=["course_name"], dtype=str)["course_name"].dropna().unique()
    norm = sorted({normalize_name(n) for n in names})
    torch.manual_seed(0)
    t0 = time.time()
    model = SentenceTransformer(a.model, device="cpu")
    vec = model.encode(norm, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(a.model)
    np.savez_compressed(out, names=np.array(norm), vectors=vec.astype(np.float32))
    meta = {"model": a.model, "n_names": len(norm), "dim": int(vec.shape[1]), "seconds": round(time.time() - t0, 1),
            "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
            "names_sha256": hashlib.sha256("\n".join(norm).encode()).hexdigest(),
            "vectors_sha256": hashlib.sha256(vec.astype(np.float32).tobytes()).hexdigest()}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
