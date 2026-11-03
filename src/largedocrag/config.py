"""Shared paths and settings for the existing LargeDocRAG baseline."""

from pathlib import Path


# config.py lives at <project>/src/largedocrag/config.py.
BASE_DIR = Path(__file__).resolve().parents[2]
DOCUMENT_DIR = BASE_DIR / "data"
PDF_PATH = DOCUMENT_DIR / "bookL.pdf"
VECTOR_STORE_PATH = BASE_DIR / "vector_store"

COLLECTION_NAME = "bookl_chunks"
EMBEDDING_MODEL = "embeddinggemma"
CHAT_MODEL = "llama3.2"
TOP_K = 3

# These values match the page-by-page character chunking in rag_exploration.ipynb.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

