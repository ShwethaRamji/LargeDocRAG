"""Unit tests for notebook-compatible ingestion using mocked dependencies."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.largedocrag.ingestion import (
    DocumentPage,
    chunk_pages,
    extract_pdf_pages,
    ingest_chunks,
)


class ExtractionAndChunkingTests(unittest.TestCase):
    def test_extracts_one_based_page_numbers(self):
        first, second = Mock(), Mock()
        first.get_text.return_value = "First page"
        second.get_text.return_value = "Second page"
        document = Mock()
        document.__iter__ = Mock(return_value=iter([first, second]))
        document.__enter__ = Mock(return_value=document)
        document.__exit__ = Mock(return_value=False)

        with patch("src.largedocrag.ingestion.pymupdf.open", return_value=document):
            pages = extract_pdf_pages(Path("book.pdf"))

        self.assertEqual(pages, [
            DocumentPage(page_number=1, text="First page"),
            DocumentPage(page_number=2, text="Second page"),
        ])

    def test_chunking_preserves_notebook_size_overlap_and_page_metadata(self):
        page = DocumentPage(page_number=7, text="abcdefghij")
        chunks = chunk_pages([page], chunk_size=6, overlap=2)
        self.assertEqual([chunk.text for chunk in chunks], ["abcdef", "efghij"])
        self.assertEqual([chunk.chunk_id for chunk in chunks], ["chunk-0", "chunk-1"])
        self.assertEqual([chunk.page_number for chunk in chunks], [7, 7])


class IngestionWriteGuardTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            DocumentPage(page_number=1, text="one"),
        ]

    def test_empty_collection_upserts_exact_ids_text_embeddings_and_pages(self):
        chunks = chunk_pages(
            [DocumentPage(5, "some page text")], chunk_size=30, overlap=0
        )
        collection = Mock()
        collection.count.side_effect = [0, 1]
        embed = Mock(return_value={"embeddings": [[0.1, 0.2]]})

        status = ingest_chunks(chunks, collection, embed=embed)

        self.assertEqual(status, "ingested")
        embed.assert_called_once_with(
            model="embeddinggemma",
            input=["some page text"],
        )
        collection.upsert.assert_called_once_with(
            ids=["chunk-0"],
            documents=["some page text"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"page_number": 5}],
        )

    def test_equal_record_count_skips_without_embedding_or_write(self):
        collection = Mock()
        collection.count.return_value = 2
        embed = Mock()
        chunks = chunk_pages([DocumentPage(1, "text")])
        status = ingest_chunks(chunks * 2, collection, embed=embed)
        self.assertEqual(status, "skipped")
        embed.assert_not_called()
        collection.upsert.assert_not_called()

    def test_partial_index_raises_without_embedding_or_write(self):
        collection = Mock()
        collection.count.return_value = 1
        embed = Mock()
        chunks = chunk_pages([DocumentPage(1, "text")]) * 2
        with self.assertRaisesRegex(RuntimeError, "partial or mismatched index"):
            ingest_chunks(chunks, collection, embed=embed)
        embed.assert_not_called()
        collection.upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
