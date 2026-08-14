"""
Document Partitioning — Step 3 of the pipeline.

Supports PDF, DOCX, TXT, and Markdown files.
Each file/page becomes a LangChain Document with metadata (source, filename, page_number).
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def partition_pdf(file_path: Path) -> list[Document]:
    """Extract text from a PDF file, one Document per page."""
    logger.info(f"Partitioning PDF: {file_path.name}")
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    for page in pages:
        page.metadata["filename"] = file_path.name
        page.metadata["page_number"] = page.metadata.get("page", 0) + 1
        page.metadata["category"] = "NarrativeText"

    logger.info(f"  → {len(pages)} pages extracted from {file_path.name}")
    return pages


def partition_docx(file_path: Path) -> list[Document]:
    """Extract text from a DOCX file."""
    logger.info(f"Partitioning DOCX: {file_path.name}")
    try:
        import docx

        doc = docx.Document(str(file_path))
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)

        # Also extract table text
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    full_text.append(row_text)

        content = "\n\n".join(full_text)
        if not content.strip():
            return []

        doc_obj = Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "page_number": 1,
                "category": "NarrativeText",
            },
        )
        return [doc_obj]
    except Exception as e:
        logger.error(f"Failed to read DOCX {file_path.name}: {e}")
        return []


def partition_text(file_path: Path) -> list[Document]:
    """Extract text from a TXT or Markdown file."""
    logger.info(f"Partitioning Text/MD: {file_path.name}")
    try:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["filename"] = file_path.name
            d.metadata["page_number"] = 1
            d.metadata["category"] = "NarrativeText"
        return docs
    except Exception as e:
        logger.error(f"Failed to read Text {file_path.name}: {e}")
        return []


def partition_file(file_path: Path) -> list[Document]:
    """Partition a single document based on extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return partition_pdf(file_path)
    elif suffix == ".docx":
        return partition_docx(file_path)
    elif suffix in (".txt", ".md"):
        return partition_text(file_path)
    else:
        return []


def partition_directory(data_dir: Path) -> dict[str, list[Document]]:
    """
    Extract text from all supported files in a directory (recursive).

    Supported: PDF, DOCX, TXT, MD.
    """
    supported_extensions = {".pdf", ".docx", ".txt", ".md"}
    all_files = [
        f for f in sorted(data_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in supported_extensions and not f.name.startswith(".")
    ]

    if not all_files:
        logger.warning(f"No supported document files found in {data_dir}")
        return {}

    logger.info(f"Found {len(all_files)} document files in {data_dir}")

    results: dict[str, list[Document]] = {}
    for file_path in all_files:
        try:
            docs = partition_file(file_path)
            if docs:
                results[file_path.name] = docs
        except Exception as e:
            logger.error(f"Failed to partition {file_path.name}: {e}")

    total = sum(len(v) for v in results.values())
    logger.info(f"Partitioning complete: {total} total document pages from {len(results)} files")
    return results
