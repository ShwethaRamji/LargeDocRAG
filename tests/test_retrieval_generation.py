"""Focused unit tests for the first extracted application modules."""

import unittest
from unittest.mock import Mock

from src.largedocrag.config import (
    BASE_DIR,
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    VECTOR_STORE_PATH,
)
from src.largedocrag.generation import build_messages, generate_answer
from src.largedocrag.retrieval import (
    RetrievedChunk,
    format_context,
    format_sources,
    retrieve_chunks,
)


class ConfigTests(unittest.TestCase):
    def test_defaults_preserve_existing_baseline(self):
        self.assertEqual(BASE_DIR.name, "LargeDocRAG")
        self.assertEqual(VECTOR_STORE_PATH, BASE_DIR / "vector_store")
        self.assertEqual(COLLECTION_NAME, "bookl_chunks")
        self.assertEqual(EMBEDDING_MODEL, "embeddinggemma")
        self.assertEqual(CHAT_MODEL, "llama3.2")
        self.assertEqual(TOP_K, 3)
        self.assertEqual((CHUNK_SIZE, CHUNK_OVERLAP), (1000, 200))


class RetrievalTests(unittest.TestCase):
    def test_retrieve_preserves_ids_page_and_distance(self):
        collection = Mock()
        collection.query.return_value = {
            "ids": [["chunk-7"]],
            "documents": [["Relevant PDF text"]],
            "metadatas": [[{"page_number": 12}]],
            "distances": [[0.125]],
        }
        embed = Mock(return_value={"embeddings": [[0.1, 0.2]]})

        chunks = retrieve_chunks("question", collection, embed=embed)

        embed.assert_called_once_with(model="embeddinggemma", input="question")
        collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        self.assertEqual(chunks, [RetrievedChunk(
            chunk_id="chunk-7",
            text="Relevant PDF text",
            metadata={"page_number": 12},
            distance=0.125,
        )])

    def test_empty_results_are_safe(self):
        collection = Mock()
        collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        chunks = retrieve_chunks(
            "question", collection,
            embed=lambda **_: {"embeddings": [[0.1]]},
        )
        self.assertEqual(chunks, [])
        self.assertEqual(format_context(chunks), "")
        self.assertEqual(format_sources(chunks), [])

    def test_context_and_sources_keep_page_citation_format(self):
        chunk = RetrievedChunk("chunk-2", "Some text", {"page_number": 8}, 0.2)
        self.assertEqual(format_context([chunk]), "[PDF page 8]\nSome text")
        self.assertEqual(format_sources([chunk]), [{
            "chunk_id": "chunk-2",
            "page_number": 8,
            "distance": 0.2,
            "text": "Some text",
        }])


class GenerationTests(unittest.TestCase):
    def test_prompt_preserves_grounding_citations_and_history(self):
        history = [{"role": "user", "content": "Earlier question"}]
        messages = build_messages("question", "[PDF page 4]\ncontext", history)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("using only the supplied PDF context", messages[0]["content"])
        self.assertIn("PDF page numbers", messages[0]["content"])
        self.assertEqual(messages[1], history[0])
        self.assertIn("PDF context:\n[PDF page 4]\ncontext", messages[2]["content"])

    def test_generate_answer_calls_configured_chat_model(self):
        chat = Mock(return_value={"message": {"content": "Grounded answer"}})
        answer = generate_answer("question", "context", chat=chat)
        self.assertEqual(answer, "Grounded answer")
        self.assertEqual(chat.call_args.kwargs["model"], "llama3.2")


if __name__ == "__main__":
    unittest.main()
