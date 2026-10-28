
"""RAG indexing, retrieval, and grounded answer generation."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = Path(
    os.getenv("RAG_INDEX_DIR", PROJECT_ROOT / "data" / "indexes")
)


def transcript_fingerprint(
    text: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Create a stable cache key for a transcript and its indexing settings."""
    payload = (
        f"{embedding_model}|{chunk_size}|{chunk_overlap}|{text}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def make_embeddings(
    model_name: str | None = None,
) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=model_name
        or os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
    )


def make_chat_model(
    model_name: str | None = None,
) -> ChatGoogleGenerativeAI:
    # Keep the chat model configurable for different API accounts.
    return ChatGoogleGenerativeAI(
        model=model_name
        or os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
    )


def get_or_create_vector_store(
    transcript: str,
    source: str,
    embeddings: GoogleGenerativeAIEmbeddings,
    embedding_model: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 16,
    delay_seconds: float = 1.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[FAISS, int, bool, str]:
    """Load a cached FAISS index or create and save one in batches."""
    if not transcript.strip():
        raise ValueError("Transcript text is empty.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    cache_key = transcript_fingerprint(
        transcript, embedding_model, chunk_size, chunk_overlap
    )
    index_path = INDEX_ROOT / cache_key
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)

    if (
        (index_path / "index.faiss").exists()
        and (index_path / "index.pkl").exists()
    ):
        # Only load index files created and trusted by this application.
        store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return store, store.index.ntotal, True, cache_key

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    docs = splitter.create_documents(
        [transcript],
        metadatas=[
            {"source": source, "transcript_hash": cache_key}
        ],
    )

    if not docs:
        raise ValueError("No text chunks were created from the transcript.")

    store = None

    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]

        if store is None:
            store = FAISS.from_documents(batch, embeddings)
        else:
            store.add_documents(batch)

        completed = min(start + len(batch), len(docs))

        if progress_callback:
            progress_callback(completed, len(docs))

        if completed < len(docs) and delay_seconds > 0:
            time.sleep(delay_seconds)

    if store is None:
        raise RuntimeError("The vector store could not be created.")

    store.save_local(str(index_path))
    return store, len(docs), False, cache_key


def build_rag_chain(
    vector_store: FAISS,
    chat_model: ChatGoogleGenerativeAI,
    k: int = 4,
):
    """Build an MMR-based RAG chain with grounded answers and source documents."""

    # MMR considers a wider set of candidates and selects relevant,
    # less-redundant chunks for the final context.
    final_k = max(1, min(int(k), 10))
    fetch_k = max(20, final_k)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": final_k,
            "fetch_k": fetch_k,
            "lambda_mult": 0.5,
        },
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a careful assistant answering questions about a source transcript.

Use only the context below. Do not rely on outside knowledge or invent facts.
If the context does not answer the question, clearly say that the
transcript does not provide enough information.
Keep the answer clear and structured. When useful, refer to the source chunk numbers.

<context>
{context}
</context>

Question: {question}

Answer:"""
    )

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[Source chunk {i}] {doc.page_content}"
            for i, doc in enumerate(docs, start=1)
        )

    # The normal chain retrieves MMR-selected chunks and formats them
    # as context for the LLM.
    answer_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )

    # This helper returns both the generated answer and the exact
    # documents used to generate it, avoiding a second retrieval.
    def answer_with_sources(question: str) -> dict:
        docs = retriever.invoke(question)
        context = format_docs(docs)

        answer = (
            prompt
            | chat_model
            | StrOutputParser()
        ).invoke(
            {"context": context, "question": question}
        )

        return {
            "answer": answer,
            "source_documents": docs,
        }

    return retriever, answer_chain, answer_with_sources
