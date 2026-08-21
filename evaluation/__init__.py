"""UniAssist evaluation framework (Phases 4+).

Everything under `evaluation/` is offline tooling that measures the RAG system.
It never mutates production: experiments run against isolated *local* Chroma
collections built from a frozen corpus snapshot (see `evaluation/lib`).
"""
