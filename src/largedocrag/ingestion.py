"""PDF extraction and ingestion helpers matching the exploration notebook."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable

import ollama
import pymupdf

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL


logger = logging.getLogger(__name__)
EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 100


@dataclass(frozen=True)
class DocumentPage:
    """Extracted text from one PDF page, using one-based PDF page numbering."""

    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """A character-based text chunk and its source page and stable Chroma ID."""

    chunk_id: str
    page_number: int
    text: str


def extract_pdf_pages(pdf_path: Path) -> list[DocumentPage]:
    """Extract page text with PyMuPDF, preserving the notebook's page numbering."""
    pages: list[DocumentPage] = []
    try:
        with pymupdf.open(str(pdf_path)) as document:
            for page_index, page in enumerate(document):
                pages.append(
                    DocumentPage(page_number=page_index + 1, text=page.get_text())
                )
    except Exception:
        logger.exception("Could not extract text from PDF %s", pdf_path)
        raise
    logger.info("Extracted %d pages from %s", len(pages), pdf_path)
    return pages


def chunk_pages(
    pages: list[DocumentPage],
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split each page with the notebook's fixed-character size and overlap.

    Text is deliberately not cleaned: the notebook chunks the exact text returned
    by PyMuPDF. Chunk IDs follow its original ``chunk-{index}`` convention.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    for page in pages:
        start = 0
        while start < len(page.text):
            end = start + chunk_size
            chunk_text = page.text[start:end]
            if chunk_text.strip():
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chunk-{len(chunks)}",
                        page_number=page.page_number,
                        text=chunk_text,
                    )
                )
            if end >= len(page.text):
                break
            start = end - overlap
    return chunks


def ingest_chunks(
    chunks: list[DocumentChunk],
    collection: Any,
    *,
    embedding_model: str = EMBEDDING_MODEL,
    embed: Callable[..., dict[str, Any]] | None = None,
) -> str:
    """Add chunks to an explicitly supplied, existing Chroma collection.

    This function never creates, deletes, or recreates a collection. It mirrors
    the notebook's count guard: equal counts are skipped, empty collections are
    populated, and partial/mismatched counts raise an error. The function is
    not called on import; callers must deliberately invoke it to write data.
    """
    expected_count = len(chunks)
    stored_count = collection.count()
    if stored_count == expected_count:
        logger.info("Collection already has %d records; skipping ingestion", stored_count)
        return "skipped"
    if stored_count != 0:
        raise RuntimeError(
            "Chroma contains a partial or mismatched index. "
            "Check the stored collection before continuing."
        )
    if not chunks:
        logger.info("No chunks to ingest")
        return "skipped"

    embedding_fn = embed or ollama.embed
    texts = [chunk.text for chunk in chunks]
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        response = embedding_fn(
            model=embedding_model,
            input=texts[start : start + EMBEDDING_BATCH_SIZE],
        )
        embeddings.extend(response["embeddings"])
        logger.info("Embedded %d of %d chunks", min(start + EMBEDDING_BATCH_SIZE, len(texts)), len(texts))

    if len(embeddings) != expected_count:
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {expected_count} chunks"
        )

    for start in range(0, expected_count, UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        batch = chunks[start:end]
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=embeddings[start:end],
            metadatas=[{"page_number": chunk.page_number} for chunk in batch],
        )
    logger.info("Ingestion complete; collection now has %d records", collection.count())
    return "ingested"


def ingest_pdf(
    pdf_path: Path,
    collection: Any,
    *,
    embedding_model: str = EMBEDDING_MODEL,
    embed: Callable[..., dict[str, Any]] | None = None,
) -> str:
    """Extract, chunk, and ingest a PDF into a caller-provided collection."""
    chunks = chunk_pages(extract_pdf_pages(pdf_path))
    return ingest_chunks(
        chunks,
        collection,
        embedding_model=embedding_model,
        embed=embed,
    )

