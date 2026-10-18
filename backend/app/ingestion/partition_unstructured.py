"""
Document Partitioning via Unstructured.io — Phase 11 alternative parser.

Drop-in replacement for ``partition.py`` that uses Unstructured's ``partition``
to extract text and structured elements from PDF, DOCX, TXT, and Markdown files.

Preserves the same API and metadata contract as the current partition module:
  - ``partition_directory(data_dir)`` → ``dict[str, list[Document]]``
  - Each Document has metadata keys: source, filename, page_number, category, source_file

Unstructured classifies elements into types (Title, NarrativeText, Table,
ListItem, Header, etc.), which we map to the category tags the downstream
pipeline expects.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Supported extensions (same as current parser)
_SUPPORTED = {".pdf", ".docx", ".txt", ".md"}

# Unstructured element types that indicate table content
_TABLE_TYPES = {"Table"}


def _element_category(element) -> str:
    """Map an Unstructured element type to the category tag used downstream."""
    etype = type(element).__name__
    if etype in _TABLE_TYPES:
        return "TableAndText"
    return "NarrativeText"


def _table_to_markdown(element) -> str:
    """Convert an Unstructured Table element to a Markdown table string.

    Unstructured exposes table data via ``element.metadata.text_as_html``
    (HTML <table>) or as plain text. We prefer the HTML route and convert
    it to Markdown for consistency with the pdfplumber parser's output.
    """
    html = getattr(element.metadata, "text_as_html", None)
    if html:
        try:
            return _html_table_to_markdown(html)
        except Exception:
            pass
    # Fallback: use the element's text representation
    return str(element)


def _html_table_to_markdown(html: str) -> str:
    """Best-effort HTML <table> → Markdown table conversion (no extra deps)."""
    import re

    # Extract rows
    rows: list[list[str]] = []
    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        tr = tr_match.group(1)
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.DOTALL | re.IGNORECASE)
        # Strip inner HTML tags and whitespace
        clean = [re.sub(r"<[^>]+>", "", c).strip().replace("\n", " ") for c in cells]
        if any(clean):
            rows.append(clean)

    if not rows or len(rows) < 2:
        # Not a real table — return raw text
        return re.sub(r"<[^>]+>", " ", html).strip()

    # Normalise column count to the widest row
    max_cols = max(len(r) for r in rows)
    for r in rows:
        r.extend([""] * (max_cols - len(r)))

    header = rows[0]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * max_cols) + " |\n"
    for r in rows[1:]:
        md += "| " + " | ".join(r[:max_cols]) + " |\n"
    return md


def partition_file_unstructured(file_path: Path) -> list[Document]:
    """Partition a single file using Unstructured and return LangChain Documents.

    Groups consecutive elements by page number so the output granularity matches
    the current parser (one Document per PDF page; single Document for docx/txt).
    Tables are rendered as Markdown.
    """
    from unstructured.partition.auto import partition

    suffix = file_path.suffix.lower()
    if suffix not in _SUPPORTED:
        return []

    logger.info(f"Partitioning (Unstructured): {file_path.name}")

    try:
        # Use 'fast' strategy for PDFs — avoids needing system OCR packages.
        # 'fast' uses pdfminer under the hood (built in to unstructured[pdf]),
        # which is different from our current pdfplumber path.
        kwargs = {}
        if suffix == ".pdf":
            kwargs["strategy"] = "fast"

        elements = partition(filename=str(file_path), **kwargs)
    except Exception as e:
        logger.error(f"Unstructured partitioning failed for {file_path.name}: {e}")
        return []

    if not elements:
        logger.warning(f"No elements extracted from {file_path.name}")
        return []

    # Group elements by page number
    pages: dict[int, list] = {}
    for el in elements:
        page_num = getattr(el.metadata, "page_number", None) or 1
        pages.setdefault(page_num, []).append(el)

    documents: list[Document] = []
    for page_num in sorted(pages.keys()):
        content_parts: list[str] = []
        has_table = False

        for el in pages[page_num]:
            etype = type(el).__name__
            if etype in _TABLE_TYPES:
                has_table = True
                md_table = _table_to_markdown(el)
                content_parts.append("\n\n### Extracted Table Data:\n" + md_table)
            else:
                text = str(el).strip()
                if text:
                    content_parts.append(text)

        full_content = "\n\n".join(part for part in content_parts if part.strip())
        if not full_content.strip():
            continue

        doc = Document(
            page_content=full_content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "page_number": page_num,
                "category": "TableAndText" if has_table else "NarrativeText",
            },
        )
        documents.append(doc)

    logger.info(f"  → {len(documents)} pages from {file_path.name} (Unstructured)")
    return documents


def partition_directory(data_dir: Path) -> dict[str, list[Document]]:
    """Extract text from all supported files using Unstructured (recursive).

    Same API as ``partition.partition_directory`` — returns a dict keyed by
    the file path relative to ``data_dir``.
    """
    all_files = [
        f for f in sorted(data_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in _SUPPORTED and not f.name.startswith(".")
    ]

    if not all_files:
        logger.warning(f"No supported document files found in {data_dir}")
        return {}

    logger.info(f"Found {len(all_files)} document files in {data_dir}")

    results: dict[str, list[Document]] = {}
    for file_path in all_files:
        try:
            docs = partition_file_unstructured(file_path)
            if docs:
                rel_key = file_path.relative_to(data_dir).as_posix()
                for d in docs:
                    d.metadata["source_file"] = rel_key
                results[rel_key] = docs
        except Exception as e:
            logger.error(f"Failed to partition {file_path.name}: {e}")

    total = sum(len(v) for v in results.values())
    logger.info(f"Partitioning complete (Unstructured): {total} total document pages from {len(results)} files")
    return results
