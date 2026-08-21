"""Shared helpers for evaluation runners: logging, latency stats, result I/O."""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from statistics import mean

from ..lib import paths


def configure_logging(verbose: bool = False) -> None:
    """Concise logging; silence chatty production/library loggers unless verbose."""
    paths.force_utf8_stdio()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not verbose:
        for noisy in ("app", "httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def latency_stats(values) -> dict:
    """avg / p50 / p95 / max over non-null latency values (ms)."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {}
    s = sorted(vals)

    def pct(p: float) -> float:
        idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
        return s[idx]

    return {
        "avg": round(mean(vals), 2),
        "p50": round(pct(50), 2),
        "p95": round(pct(95), 2),
        "max": round(max(vals), 2),
        "n": len(vals),
    }


def dataset_meta(data: dict, evaluated: int) -> dict:
    qs = data.get("questions", [])
    answerable = sum(1 for q in qs if q.get("answerable", True))
    return {
        "name": data.get("dataset"),
        "version": data.get("version"),
        "phase": data.get("phase"),
        "questions": len(qs),
        "answerable": answerable,
        "unanswerable": len(qs) - answerable,
        "evaluated": evaluated,
    }


def environment_meta() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform()}


def write_result(out_path: Path, envelope: dict) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
