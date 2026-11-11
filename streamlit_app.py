import chromadb
import ollama
import streamlit as st

from src.largedocrag.config import (
    CHAT_MODEL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    VECTOR_STORE_PATH,
)
from src.largedocrag.generation import generate_answer_with_usage
from src.largedocrag.retrieval import (
    format_context,
    format_sources,
    retrieve_chunks,
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_token_usage" not in st.session_state:
    st.session_state.last_token_usage = None


st.set_page_config(
    page_title="Book RAG",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource
def load_collection():
    """Open the existing persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def retrieve_context(question: str, collection):
    """Embed a question and retrieve the most relevant chunks."""
    embedding_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question,
    )
    question_embedding = embedding_response["embeddings"][0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    sources = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        page_number = metadata["page_number"]
        context_parts.append(
            f"[PDF page {page_number}]\n{document}"
        )
        sources.append(
            {
                "page_number": page_number,
                "distance": distance,
                "text": document,
            }
        )

    return "\n\n---\n\n".join(context_parts), sources


def generate_answer(question: str, context: str, chat_history):
    """Ask Llama to answer using only the retrieved PDF context."""
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using only the supplied PDF context. "
                "If the answer is not supported by the context, say that "
                "the PDF context does not provide enough information. "
                "Include the relevant PDF page numbers in your answer."
            ),
        }
    ]

    messages.extend(chat_history)

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

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
    )
    return response["message"]["content"]


st.title("📚 Ask your book")
st.caption("Answers are generated from the indexed PDF using Ollama.")

with st.sidebar:
    st.subheader("Index information")
    st.write(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.write(f"Answer model: `{CHAT_MODEL}`")
    st.write(f"Retrieved chunks per question: `{TOP_K}`")

    try:
        collection = load_collection()
        st.success(f"ChromaDB records: {collection.count()}")
    except Exception as error:
        collection = None
        st.error("Could not open the ChromaDB collection.")
        st.caption(str(error))

    token_usage_display = st.empty()
    last_usage = st.session_state.last_token_usage
    if last_usage is None:
        token_usage_display.caption("Token usage will appear after your first question.")
    elif last_usage["total"] is None:
        token_usage_display.metric("Tokens for last question", "Unavailable")
    else:
        token_usage_display.metric(
            "Tokens for last question", f"{last_usage['total']:,}"
        )
        token_usage_display.caption(
            f"Input: {last_usage['prompt']:,} · Answer: {last_usage['answer']:,}"
        )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("token_usage"):
            usage = message["token_usage"]
            if usage["total"] is None:
                st.caption("Token usage unavailable")
            else:
                st.caption(
                    f"Tokens: {usage['total']:,} total "
                    f"(input {usage['prompt']:,} · answer {usage['answer']:,})"
                )


question = st.chat_input("Ask a question about the PDF")


if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if collection is None:
            st.error("The ChromaDB collection is unavailable.")
        else:
            try:
                with st.spinner(
                    "Searching the PDF and generating an answer..."
                ):
                    search_question = question
                    chunks = retrieve_chunks(search_question, collection)
                    context = format_context(chunks)
                    sources = format_sources(chunks)
                    answer, prompt_tokens, answer_tokens = generate_answer_with_usage(
                        question,
                        context,
                        st.session_state.chat_history,
                    )

                total_tokens = (
                    prompt_tokens + answer_tokens
                    if prompt_tokens is not None and answer_tokens is not None
                    else None
                )
                token_usage = {
                    "prompt": prompt_tokens,
                    "answer": answer_tokens,
                    "total": total_tokens,
                }
                st.session_state.last_token_usage = token_usage
                if total_tokens is None:
                    token_usage_display.metric(
                        "Tokens for last question", "Unavailable"
                    )
                else:
                    token_usage_display.metric(
                        "Tokens for last question", f"{total_tokens:,}"
                    )
                    token_usage_display.caption(
                        f"Input: {prompt_tokens:,} · Answer: {answer_tokens:,}"
                    )

                st.markdown(answer)
                if total_tokens is None:
                    st.caption("Token usage unavailable")
                else:
                    st.caption(
                        f"Tokens: {total_tokens:,} total "
                        f"(input {prompt_tokens:,} · answer {answer_tokens:,})"
                    )

                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "token_usage": token_usage,
                    }
                )

                if sources:
                    with st.expander("Retrieved sources"):
                        for source in sources:
                            st.markdown(
                                f"**PDF page {source['page_number']}** "
                                f"(distance: {source['distance']:.4f})"
                            )
                            st.caption(source["text"][:500])

            except Exception as error:
                st.error(f"Request failed: {error}")
