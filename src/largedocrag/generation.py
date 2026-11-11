"""Grounded answer generation using the configured local Ollama model."""

from collections.abc import Callable, Sequence
import json
import sys
from typing import Any

import ollama

from .config import CHAT_MODEL


SYSTEM_PROMPT = (
    "You answer questions using only the supplied PDF context. "
    "If the answer is not supported by the context, say that "
    "the PDF context does not provide enough information. "
    "Include the relevant PDF page numbers in your answer."
)


def _print_terminal(text: str, *, flush: bool = False) -> None:
    """Print prompt text safely even when the terminal encoding is limited."""
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None)
    if encoding:
        text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    print(text, file=stream, flush=flush)


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
    answer, _, _ = generate_answer_with_usage(
        question, context, chat_history, model=model, chat=chat
    )
    return answer


def generate_answer_with_usage(
    question: str,
    context: str,
    chat_history: Sequence[dict[str, str]] | None = None,
    *,
    model: str = CHAT_MODEL,
    chat: Callable[..., dict[str, Any]] | None = None,
) -> tuple[str, int | None, int | None]:
    """Return the answer plus Ollama's prompt and generated token counts."""
    chat_fn = chat or ollama.chat
    messages = build_messages(question, context, chat_history)

    # Print both the labeled prompt parts and the exact payload so the terminal
    # log makes it clear what Ollama receives, including system instructions
    # and prior turns.
    _print_terminal("\n========== LLM request ==========")
    _print_terminal(f"model - {model}")
    _print_terminal(f"system prompt -\n{SYSTEM_PROMPT}")
    _print_terminal("chat history -")
    history = list(chat_history or [])
    if history:
        for message in history:
            _print_terminal(f"[{message['role']}] {message['content']}")
    else:
        _print_terminal("(none)")
    _print_terminal(f"user question -\n{question}")
    _print_terminal(f"PDF context -\n{context if context else '(no context retrieved)'}")
    _print_terminal("complete messages sent to LLM -")
    _print_terminal(json.dumps(messages, ensure_ascii=False, indent=2))
    _print_terminal("======== end LLM request ========\n", flush=True)

    response = chat_fn(
        model=model,
        messages=messages,
    )
    answer = response["message"]["content"]
    if isinstance(response, dict):
        prompt_tokens = response.get("prompt_eval_count")
        answer_tokens = response.get("eval_count")
    else:
        prompt_tokens = getattr(response, "prompt_eval_count", None)
        answer_tokens = getattr(response, "eval_count", None)
    return answer, prompt_tokens, answer_tokens

