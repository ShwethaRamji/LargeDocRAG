"""Dense ChromaDB retrieval and context formatting for LargeDocRAG."""

from dataclasses import dataclass
from typing import Any, Callable

import ollama

from .config import EMBEDDING_MODEL, TOP_K


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved ChromaDB record, including its stable ID and provenance."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def page_number(self) -> Any:
        """Return the stored PDF page number, if present."""
        return self.metadata.get("page_number")


def retrieve_chunks(
    question: str,
    collection: Any,
    *,
    top_k: int = TOP_K,
    embedding_model: str = EMBEDDING_MODEL,
    embed: Callable[..., dict[str, Any]] | None = None,
) -> list[RetrievedChunk]:
    """Embed a question and run the existing dense ChromaDB similarity query.

    ``embed`` is injectable so callers can test retrieval without starting Ollama.
    When omitted, this calls Ollama exactly as the original Streamlit app did.
    """
    embedding_fn = embed or ollama.embed
    embedding_response = embedding_fn(model=embedding_model, input=question)
    question_embedding = embedding_response["embeddings"][0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns each query's results as a nested list. An empty result set
    # is handled here without indexing an absent first result row.
    ids = _first_result_row(results.get("ids"))
    documents = _first_result_row(results.get("documents"))
    metadatas = _first_result_row(results.get("metadatas"))
    distances = _first_result_row(results.get("distances"))

    return [
        RetrievedChunk(
            chunk_id=str(chunk_id),
            text=document,
            metadata=metadata or {},
            distance=float(distance),
        )
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        )
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved text with page labels as the original app does."""
    return "\n\n---\n\n".join(
        f"[PDF page {chunk.page_number}]\n{chunk.text}" for chunk in chunks
    )


def format_sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Convert retrieved chunks into source records suitable for the UI."""
    return [
        {
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "distance": chunk.distance,
            "text": chunk.text,
        }
        for chunk in chunks
    ]


def _first_result_row(value: Any) -> list[Any]:
    """Unwrap one Chroma query result row, returning an empty list if absent."""
    if not value:
        return []
    return value[0] or []

