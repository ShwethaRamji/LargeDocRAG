"""Page-level retrieval metrics and human response-review helpers.

The available gold labels identify source PDF pages rather than relevant chunks.
Metrics therefore evaluate the first occurrence of each retrieved page. This is
an explicit page-level proxy and should not be interpreted as chunk-level truth.
"""

import argparse
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Iterable

from .config import (
    BASE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TOP_K,
    VECTOR_STORE_PATH,
)
from .retrieval import RetrievedChunk, retrieve_chunks
from .generation import generate_answer_with_usage


logger = logging.getLogger(__name__)
DATASET_PATH = BASE_DIR / "evals" / "evaluation_dataset.json"
RESULTS_PATH = BASE_DIR / "experiments" / "results.csv"
SUMMARY_FIELDS = [
    "Experiment name",
    "Date",
    "Retrieval method",
    "Embedding model",
    "Top-K",
    "Precision@K",
    "Recall@K",
    "MRR",
    "Hit Rate@K",
    "Average retrieval latency",
    "Notes",
]
QUESTION_RESULT_FIELDS = [
    "question_id",
    "question",
    "relevant_pdf_pages",
    "retrieved_pdf_pages",
    "retrieved_unique_pages_at_k",
    "retrieved_chunk_ids",
    "retrieved_distances",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "hit_rate_at_k",
    "latency_seconds",
]
RESPONSE_QUALITY_DIMENSIONS = (
    "correctness",
    "relevance",
    "faithfulness",
    "groundedness",
    "completeness",
    "citation_quality",
)


def load_evaluation_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Read and minimally validate the page-labeled JSON evaluation set."""
    with path.open("r", encoding="utf-8") as dataset_file:
        records = json.load(dataset_file)
    if not isinstance(records, list):
        raise ValueError("Evaluation dataset must be a JSON list of records")
    for index, record in enumerate(records):
        for required in ("question_id", "question", "relevant_pdf_pages"):
            if required not in record:
                raise ValueError(f"Record {index} is missing {required!r}")
    return records


def page_retrieval_metrics(
    retrieved_pages: Iterable[int | None],
    relevant_pages: Iterable[int],
    *,
    k: int = TOP_K,
) -> dict[str, float | None]:
    """Calculate Precision@K, Recall@K, MRR, and Hit Rate@K over unique pages.

    Repeated chunks from the same page count once, at their first rank. Precision
    uses the requested K as its denominator (missing unique pages count as
    non-relevant); recall uses the number of labeled relevant pages.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    unique_retrieved: list[int] = []
    seen: set[int] = set()
    for page in retrieved_pages:
        if page is not None and page not in seen:
            seen.add(page)
            unique_retrieved.append(page)
        if len(unique_retrieved) >= k:
            break

    relevant = set(relevant_pages)
    if not relevant:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "hit_rate_at_k": None,
        }

    relevant_retrieved = [page for page in unique_retrieved if page in relevant]
    reciprocal_rank = next(
        (1.0 / rank for rank, page in enumerate(unique_retrieved, start=1) if page in relevant),
        0.0,
    )
    return {
        "precision_at_k": len(relevant_retrieved) / k,
        "recall_at_k": len(relevant_retrieved) / len(relevant),
        "mrr": reciprocal_rank,
        "hit_rate_at_k": float(bool(relevant_retrieved)),
    }


def evaluate_retrieval(
    records: list[dict[str, Any]],
    collection: Any,
    *,
    top_k: int = TOP_K,
    retrieve: Callable[..., list[RetrievedChunk]] = retrieve_chunks,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """Run page-level retrieval evaluation and return per-question and mean data."""
    per_question: list[dict[str, Any]] = []
    for record in records:
        started = perf_counter()
        chunks = retrieve(record["question"], collection, top_k=top_k)
        latency = perf_counter() - started
        pages = [chunk.page_number for chunk in chunks]
        metrics = page_retrieval_metrics(
            pages,
            record.get("relevant_pdf_pages", []),
            k=top_k,
        )
        per_question.append({
            "question_id": record["question_id"],
            "question": record["question"],
            "relevant_pdf_pages": record.get("relevant_pdf_pages", []),
            "retrieved": [
                {
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "distance": chunk.distance,
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
            "retrieved_unique_pages_at_k": _unique_pages(pages, top_k),
            "latency_seconds": latency,
            **metrics,
        })

    aggregate = {
        key: _mean_defined(row[key] for row in per_question)
        for key in ("precision_at_k", "recall_at_k", "mrr", "hit_rate_at_k")
    }
    aggregate["average_retrieval_latency"] = _mean_defined(
        row["latency_seconds"] for row in per_question
    )
    return per_question, aggregate


def append_experiment_summary(
    aggregate: dict[str, float | None],
    *,
    path: Path = RESULTS_PATH,
    experiment_name: str = "dense_baseline",
    top_k: int = TOP_K,
    notes: str = "Page-level labels from the existing 20-question evaluation CSV.",
) -> None:
    """Append one real evaluation run to the summary CSV, creating headers if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {
        "Experiment name": experiment_name,
        "Date": datetime.now().astimezone().date().isoformat(),
        "Retrieval method": "dense cosine similarity",
        "Embedding model": EMBEDDING_MODEL,
        "Top-K": top_k,
        "Precision@K": aggregate.get("precision_at_k"),
        "Recall@K": aggregate.get("recall_at_k"),
        "MRR": aggregate.get("mrr"),
        "Hit Rate@K": aggregate.get("hit_rate_at_k"),
        "Average retrieval latency": aggregate.get("average_retrieval_latency"),
        "Notes": notes,
    }
    with path.open("a", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_question_results_csv(
    question_results: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write one Excel-friendly CSV row per evaluated question."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # A UTF-8 BOM helps desktop Excel recognize UTF-8 when opening a CSV directly.
    with path.open("w", newline="", encoding="utf-8-sig") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=QUESTION_RESULT_FIELDS)
        writer.writeheader()
        for result in question_results:
            retrieved = result.get("retrieved", [])
            writer.writerow({
                "question_id": result["question_id"],
                "question": result["question"],
                "relevant_pdf_pages": _join_values(result.get("relevant_pdf_pages", [])),
                "retrieved_pdf_pages": _join_values(
                    item.get("page_number") for item in retrieved
                ),
                "retrieved_unique_pages_at_k": _join_values(
                    result.get("retrieved_unique_pages_at_k", [])
                ),
                "retrieved_chunk_ids": _join_values(
                    item.get("chunk_id") for item in retrieved
                ),
                "retrieved_distances": _join_values(
                    item.get("distance") for item in retrieved
                ),
                "precision_at_k": result.get("precision_at_k"),
                "recall_at_k": result.get("recall_at_k"),
                "mrr": result.get("mrr"),
                "hit_rate_at_k": result.get("hit_rate_at_k"),
                "latency_seconds": result.get("latency_seconds"),
            })


def response_review_template(
    record: dict[str, Any],
    *,
    generated_answer: str = "",
    retrieved_pages: list[int] | None = None,
) -> dict[str, Any]:
    """Build a human-review record with unscored quality dimensions."""
    return {
        "question_id": record["question_id"],
        "question": record["question"],
        "expected_answer": record.get("expected_answer", ""),
        "generated_answer": generated_answer,
        "retrieved_pages": retrieved_pages or [],
        "human_review": {dimension: None for dimension in RESPONSE_QUALITY_DIMENSIONS},
        "reviewer_notes": "",
    }


def write_response_review_template(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """Save blank human-review records; no automatic judge or scores are used."""
    reviews = [response_review_template(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reviews, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _unique_pages(pages: Iterable[int | None], k: int) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for page in pages:
        if page is not None and page not in seen:
            seen.add(page)
            unique.append(page)
        if len(unique) >= k:
            break
    return unique


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return mean(defined) if defined else None


def _join_values(values: Iterable[Any]) -> str:
    """Serialize a list-valued CSV field with semicolon separators."""
    return ";".join("" if value is None else str(value) for value in values)


def main() -> None:
    """Run the dense baseline against the existing persistent collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--experiment-name", default="dense_baseline")
    args = parser.parse_args()

    # The runner only gets the existing collection and issues queries. It does
    # not create, modify, or re-ingest collection records.
    import chromadb

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    records = load_evaluation_dataset(args.dataset)
    results, aggregate = evaluate_retrieval(records, collection, top_k=args.top_k)

    # Generate an answer from the same retrieved chunks so the saved run can
    # be reviewed by an LLM judge without repeating retrieval or generation.
    for result, record in zip(results, records):
        context = "\n\n---\n\n".join(
            f"[PDF page {chunk['page_number']}]\n{chunk['text']}"
            for chunk in result["retrieved"]
        )
        answer, prompt_tokens, answer_tokens = generate_answer_with_usage(
            result["question"], context
        )
        result["expected_answer"] = record.get("expected_answer", "")
        result["question_type"] = record.get("question_type")
        result["answerable"] = record.get("answerable")
        result["generated_answer"] = answer
        result["token_usage"] = {
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "total_tokens": (
                prompt_tokens + answer_tokens
                if prompt_tokens is not None and answer_tokens is not None
                else None
            ),
        }

    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_path = BASE_DIR / "experiments" / "retrieval_runs" / f"{run_id}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "retrieval_method": "dense cosine similarity",
                "embedding_model": EMBEDDING_MODEL,
                "answer_model": CHAT_MODEL,
                "top_k": args.top_k,
                "aggregate": aggregate,
                "questions": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    question_csv_path = run_path.with_suffix(".csv")
    write_question_results_csv(results, question_csv_path)
    append_experiment_summary(
        aggregate,
        experiment_name=args.experiment_name,
        top_k=args.top_k,
    )
    logger.info("Saved per-question results to %s", run_path)
    logger.info("Saved spreadsheet view to %s", question_csv_path)
    logger.info("Aggregate metrics: %s", aggregate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

