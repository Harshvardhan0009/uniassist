"""
Ingestion Pipeline Orchestrator — Steps 3–6 combined.

Runs the full ingestion flow:
  partition documents → chunk → summarize → embed → store in ChromaDB

Can be run as a CLI script:
  python -m app.ingestion.pipeline
"""

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from app.config import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_and_store
from app.ingestion.partition import partition_directory
from app.ingestion.summarizer import summarize_chunks

# ── Logging setup ────────────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def run_pipeline(data_dir: Path | None = None) -> dict:
    """
    Run the full ingestion pipeline on all documents in the data directory.

    Args:
        data_dir: Override the default data directory from config.

    Returns:
        Summary dict with stats about the ingestion run.
    """
    data_dir = data_dir or settings.DATA_DIR

    console.rule("[bold blue]UniAssist — Ingestion Pipeline[/bold blue]")

    # ── Step 3: Partition ────────────────────────────────────────────
    console.print("\n[bold]Step 3:[/bold] Partitioning documents...")
    all_elements = partition_directory(data_dir)

    if not all_elements:
        console.print("[red]No documents found or processed. Aborting.[/red]")
        return {"status": "error", "reason": "no_documents"}

    # ── Step 4: Chunk ────────────────────────────────────────────────
    console.print("\n[bold]Step 4:[/bold] Chunking documents...")
    all_chunks = []
    for filename, elements in all_elements.items():
        chunks = chunk_documents(elements, source_file=filename)
        all_chunks.extend(chunks)

    console.print(f"  Total chunks: [green]{len(all_chunks)}[/green]")

    # ── Step 5: Summarize ────────────────────────────────────────────
    console.print("\n[bold]Step 5:[/bold] Summarizing chunks...")
    summarized = summarize_chunks(all_chunks)

    # ── Step 6: Embed and store ──────────────────────────────────────
    console.print("\n[bold]Step 6:[/bold] Embedding & storing in ChromaDB...")
    stored_count = embed_and_store(summarized)

    # ── Summary ──────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Ingestion Complete[/bold green]")

    summary_table = Table(title="Pipeline Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("PDFs processed", str(len(all_elements)))
    summary_table.add_row(
        "Total elements",
        str(sum(len(v) for v in all_elements.values())),
    )
    summary_table.add_row("Total chunks", str(len(all_chunks)))
    summary_table.add_row("Documents stored", str(stored_count))
    if settings.is_chroma_server:
        vector_store_desc = (
            f"{'https' if settings.CHROMA_SSL else 'http'}://"
            f"{settings.CHROMA_HOST}:{settings.CHROMA_PORT} (server)"
        )
    else:
        vector_store_desc = str(settings.CHROMA_PERSIST_DIR)
    summary_table.add_row("Vector store", vector_store_desc)
    summary_table.add_row("Collection", settings.CHROMA_COLLECTION)
    summary_table.add_row("Embedding model", settings.EMBEDDING_MODEL)
    summary_table.add_row(
        "Summarization",
        "✓ Enabled" if settings.has_llm else "⚠ Skipped (no API key)",
    )
    console.print(summary_table)

    return {
        "status": "success",
        "pdfs_processed": len(all_elements),
        "total_elements": sum(len(v) for v in all_elements.values()),
        "total_chunks": len(all_chunks),
        "documents_stored": stored_count,
    }


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UniAssist Ingestion Pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Path to data directory (default: {settings.DATA_DIR})",
    )
    args = parser.parse_args()

    result = run_pipeline(args.data_dir)

    if result["status"] != "success":
        sys.exit(1)
