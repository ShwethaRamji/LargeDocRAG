"""Grounded answer generation using the configured local Ollama model."""

from collections.abc import Callable, Sequence
from typing import Any

import ollama

from .config import CHAT_MODEL


SYSTEM_PROMPT = (
    "You answer questions using only the supplied PDF context. "
    "If the answer is not supported by the context, say that "
    "the PDF context does not provide enough information. "
    "Include the relevant PDF page numbers in your answer."
)


def build_messages(
    question: str,
    context: str,
    chat_history: Sequence[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the grounded prompt while retaining prior conversation turns."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history or [])
    messages.append(
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"PDF context:\n{context}\n\n"
                "Answer the question using the PDF context."
            ),
        }
    )
    return messages


def generate_answer(
    question: str,
    context: str,
    chat_history: Sequence[dict[str, str]] | None = None,
    *,
    model: str = CHAT_MODEL,
    chat: Callable[..., dict[str, Any]] | None = None,
) -> str:
    """Ask Ollama to answer from retrieved context and return its response."""
    chat_fn = chat or ollama.chat
    response = chat_fn(
        model=model,
        messages=build_messages(question, context, chat_history),
    )
    return response["message"]["content"]

