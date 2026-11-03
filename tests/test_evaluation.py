"""Unit tests for page-level baseline metrics and manual response review."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

from src.largedocrag.evaluation import (
    RESPONSE_QUALITY_DIMENSIONS,
    evaluate_retrieval,
    load_evaluation_dataset,
    page_retrieval_metrics,
    response_review_template,
    write_response_review_template,
    write_question_results_csv,
)
from src.largedocrag.retrieval import RetrievedChunk


class PageMetricTests(unittest.TestCase):
    def test_metrics_deduplicate_pages_at_first_rank(self):
        metrics = page_retrieval_metrics([1, 1, 2], [2, 4], k=3)
        self.assertEqual(metrics, {
            "precision_at_k": 1 / 3,
            "recall_at_k": 1 / 2,
            "mrr": 1 / 2,
            "hit_rate_at_k": 1.0,
        })

    def test_empty_retrieval_returns_zero_metrics(self):
        metrics = page_retrieval_metrics([], [4], k=3)
        self.assertEqual(metrics, {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "hit_rate_at_k": 0.0,
        })

    def test_no_gold_page_labels_are_unscored(self):
        metrics = page_retrieval_metrics([4, 5], [], k=3)
        self.assertTrue(all(value is None for value in metrics.values()))

    def test_evaluation_calls_retrieval_and_returns_details(self):
        retrieve = Mock(return_value=[
            RetrievedChunk("chunk-9", "text", {"page_number": 9}, 0.3),
        ])
        records = [{
            "question_id": "Q001",
            "question": "question",
            "relevant_pdf_pages": [9],
        }]
        per_question, aggregate = evaluate_retrieval(
            records, Mock(), top_k=3, retrieve=retrieve
        )
        retrieve.assert_called_once()
        self.assertEqual(per_question[0]["retrieved"][0]["chunk_id"], "chunk-9")
        self.assertEqual(per_question[0]["retrieved_unique_pages_at_k"], [9])
        self.assertEqual(per_question[0]["hit_rate_at_k"], 1.0)
        self.assertGreaterEqual(per_question[0]["latency_seconds"], 0.0)
        self.assertEqual(aggregate["mrr"], 1.0)


class DatasetAndHumanReviewTests(unittest.TestCase):
    def test_dataset_contains_original_twenty_questions_and_page_labels(self):
        records = load_evaluation_dataset()
        self.assertEqual(len(records), 20)
        self.assertEqual(records[0]["question_id"], "Q001")
        self.assertEqual(records[0]["relevant_pdf_pages"], [3])
        self.assertEqual(records[0]["relevant_chunk_ids"], [])

    def test_response_review_starts_unscored(self):
        record = {
            "question_id": "Q001",
            "question": "question",
            "expected_answer": "expected",
        }
        review = response_review_template(record, generated_answer="answer")
        self.assertEqual(review["generated_answer"], "answer")
        self.assertEqual(set(review["human_review"]), set(RESPONSE_QUALITY_DIMENSIONS))
        self.assertTrue(all(score is None for score in review["human_review"].values()))

    def test_review_template_writer_serializes_blank_human_scores(self):
        record = {"question_id": "Q001", "question": "question"}
        target = MagicMock(spec=Path)
        target.parent = MagicMock(spec=Path)
        write_response_review_template([record], target)
        target.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        serialized = target.write_text.call_args.args[0]
        self.assertIn('"citation_quality": null', serialized)

    def test_question_csv_writer_creates_one_spreadsheet_row_per_question(self):
        target = MagicMock(spec=Path)
        target.parent = MagicMock(spec=Path)
        file_handle = MagicMock()
        file_handle.__enter__.return_value = file_handle
        target.open.return_value = file_handle
        rows = [{
            "question_id": "Q001",
            "question": "What is in the book?",
            "relevant_pdf_pages": [3],
            "retrieved_pdf_pages": [3, 8],
            "retrieved_unique_pages_at_k": [3, 8],
            "retrieved": [
                {"chunk_id": "chunk-4", "page_number": 3, "distance": 0.1},
                {"chunk_id": "chunk-5", "page_number": 8, "distance": 0.2},
            ],
            "precision_at_k": 1 / 3,
            "recall_at_k": 1.0,
            "mrr": 1.0,
            "hit_rate_at_k": 1.0,
            "latency_seconds": 0.05,
        }]

        write_question_results_csv(rows, target)

        target.open.assert_called_once_with("w", newline="", encoding="utf-8-sig")
        serialized = "".join(call.args[0] for call in file_handle.write.call_args_list)
        self.assertEqual(len(serialized.strip().splitlines()), 2)
        self.assertIn("Q001", serialized)
        self.assertIn("chunk-4;chunk-5", serialized)


if __name__ == "__main__":
    unittest.main()
