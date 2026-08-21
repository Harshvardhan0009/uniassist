"""Path resolution and import wiring for the evaluation framework.

The evaluation code lives at the repo root under ``evaluation/`` but needs to
import the production application package (``app...``) which lives under
``backend/``. This module centralises those paths and makes ``app`` importable
without changing the working directory or touching production code.

Run runners from the project root with the backend venv, e.g.::

    backend\\venv\\Scripts\\python.exe -m evaluation.runners.retrieval_eval --config baseline_v1
"""

from __future__ import annotations

import sys
from pathlib import Path

# evaluation/lib/paths.py -> evaluation/lib -> evaluation -> <project root>
EVAL_DIR: Path = Path(__file__).resolve().parent.parent
PROJECT_ROOT: Path = EVAL_DIR.parent
BACKEND_DIR: Path = PROJECT_ROOT / "backend"
DATA_DIR: Path = PROJECT_ROOT / "Data"

# Evaluation sub-directories
DATASET_DIR: Path = EVAL_DIR / "dataset"
CONFIGS_DIR: Path = EVAL_DIR / "configs"
EXPERIMENTS_DIR: Path = EVAL_DIR / "experiments"
RESULTS_DIR: Path = EXPERIMENTS_DIR / "results"
REPORTS_DIR: Path = EVAL_DIR / "reports"
SNAPSHOT_DIR: Path = EVAL_DIR / "snapshots"
# Local, isolated Chroma stores (gitignored — regenerated from committed snapshots)
CHROMA_DIR: Path = EVAL_DIR / ".chroma"

QUESTIONS_PATH: Path = DATASET_DIR / "questions.json"


def ensure_backend_on_path() -> None:
    """Make the production ``app`` package importable (adds ``backend/`` to sys.path)."""
    p = str(BACKEND_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_dirs() -> None:
    """Create the writable output directories if they don't exist."""
    for d in (RESULTS_DIR, REPORTS_DIR, SNAPSHOT_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8.

    The production ingestion/query modules log Unicode (→, ✓, ⚠, —). On Windows
    consoles that default to cp1252 this can crash logging, so mirror the guard
    used in ``app.ingestion.pipeline``.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
