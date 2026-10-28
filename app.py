"""Streamlit UI for a YouTube Transcript RAG Assistant."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag_engine import build_rag_chain, get_or_create_vector_store, make_chat_model, make_embeddings
from src.transcript import fetch_youtube_transcript

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(128,128,128,.25); border-radius: 18px; margin-bottom: 1rem;}
.hero h1 {margin-bottom: .3rem;}
.small-muted {opacity: .75; font-size: .92rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎬 YouTube RAG Assistant</h1>
  <p>Ask questions about a video transcript. Retrieve relevant passages, generate grounded answers with Gemini, and inspect the sources used.</p>
  <p class="small-muted">LangChain · Gemini · FAISS · Streamlit</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    embedding_model = st.text_input("Embedding model", value=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
    chat_model = st.text_input("Chat model", value=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash"))
    chunk_size = st.slider("Chunk size", 500, 1800, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, step=50)
    top_k = st.slider("Retrieved chunks (k)", 1, 8, 4)
    st.caption("Changing the embedding model or chunk settings creates a separate local index.")

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning("Gemini API key not found. Add GOOGLE_API_KEY to your local .env file or set it as a deployment secret.")

if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "source_name" not in st.session_state:
    st.session_state.source_name = "uploaded transcript"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "index_key" not in st.session_state:
    st.session_state.index_key = None
if "messages" not in st.session_state:
    st.session_state.messages = []

source_tab, upload_tab, paste_tab = st.tabs(["YouTube URL", "Upload transcript", "Paste transcript"])
with source_tab:
    st.write("Try fetching captions directly. Some networks and cloud hosts block YouTube transcript requests; if that happens, upload or paste the transcript instead.")
    video_input = st.text_input("YouTube video URL or video ID", placeholder="https://www.youtube.com/watch?v=...")
    language = st.text_input("Caption language codes (comma-separated)", value="en", help="Example: en or hi,en")
    if st.button("Fetch transcript", type="primary", disabled=not video_input.strip()):
        try:
            with st.spinner("Fetching YouTube captions..."):
                transcript, video_id = fetch_youtube_transcript(video_input, tuple(x.strip() for x in language.split(",") if x.strip()))
            st.session_state.transcript_text = transcript
            st.session_state.source_name = f"YouTube video {video_id}"
            st.session_state.vector_store = None
            st.session_state.index_key = None
            st.session_state.messages = []
            st.success(f"Transcript fetched ({len(transcript):,} characters).")
        except Exception as exc:
            st.error(f"Could not fetch captions: {exc}")
            st.info("This can happen when YouTube blocks requests from your IP or captions are unavailable. Use the Upload transcript or Paste transcript tab.")

with upload_tab:
    uploaded_file = st.file_uploader("Upload a transcript text file", type=["txt", "md", "srt", "vtt"])
    if uploaded_file is not None and st.button("Use uploaded transcript"):
        raw = uploaded_file.getvalue().decode("utf-8", errors="replace")
        # Remove common SRT/VTT timestamp and sequence lines; keep caption text.
        cleaned_lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.upper() == "WEBVTT" or line.isdigit() or " --> " in line:
                continue
            cleaned_lines.append(line)
        st.session_state.transcript_text = " ".join(cleaned_lines)
        st.session_state.source_name = uploaded_file.name
        st.session_state.vector_store = None
        st.session_state.index_key = None
        st.session_state.messages = []
        st.success(f"Loaded {len(st.session_state.transcript_text):,} characters from {uploaded_file.name}.")

with paste_tab:
    pasted = st.text_area("Paste transcript text", height=180, placeholder="Paste the video transcript here...")
    if st.button("Use pasted transcript", disabled=not pasted.strip()):
        st.session_state.transcript_text = pasted.strip()
        st.session_state.source_name = "pasted transcript"
        st.session_state.vector_store = None
        st.session_state.index_key = None
        st.session_state.messages = []
        st.success(f"Loaded {len(pasted):,} characters.")

if st.session_state.transcript_text:
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Transcript characters", f"{len(st.session_state.transcript_text):,}")
    col2.metric("Source", st.session_state.source_name[:32])
    col3.metric("Index status", "Ready" if st.session_state.vector_store is not None else "Not built")
    with st.expander("Preview transcript"):
        st.text(st.session_state.transcript_text[:5000] + ("…" if len(st.session_state.transcript_text) > 5000 else ""))

    if st.button("Build / load searchable index", type="primary", disabled=not api_key):
        try:
            embeddings = make_embeddings(embedding_model)
            progress = st.progress(0, text="Preparing transcript chunks...")
            def update_progress(done: int, total: int) -> None:
                progress.progress(min(done / total, 1.0), text=f"Embedding chunks: {done}/{total}")
            with st.spinner("Checking for a saved index or creating embeddings..."):
                store, count, cached, key = get_or_create_vector_store(
                    transcript=st.session_state.transcript_text,
                    source=st.session_state.source_name,
                    embeddings=embeddings,
                    embedding_model=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    progress_callback=update_progress,
                )
            progress.progress(1.0, text="Index ready")
            st.session_state.vector_store = store
            st.session_state.index_key = key
            st.session_state.retriever, st.session_state.rag_chain, st.session_state.answer_fn = build_rag_chain(
                store, make_chat_model(chat_model), top_k
            )
            st.success(f"{'Loaded cached' if cached else 'Created'} FAISS index with {count} vectors.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            st.caption("Check the API key, model access, quota, and internet connection. A 429 quota error cannot be fixed just by reducing batch size.")

    if st.session_state.vector_store is not None:
        st.subheader("Ask the transcript")
        st.caption("Answers are grounded in retrieved transcript chunks. Open Sources below an answer to inspect the evidence.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Sources used"):
                        for i, doc in enumerate(message["sources"], 1):
                            st.markdown(f"**Chunk {i}** · `{doc.metadata.get('source', st.session_state.source_name)}`")
                            st.write(doc.page_content)
        question = st.chat_input("Ask a question about this video...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Retrieving context and asking Gemini..."):
                        result = st.session_state.answer_fn(question)
                    st.markdown(result["answer"])
                    with st.expander("Sources used"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"**Chunk {i}** · `{doc.metadata.get('source', st.session_state.source_name)}`")
                            st.write(doc.page_content)
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["source_documents"]})
                except Exception as exc:
                    st.error(f"Answer generation failed: {exc}")
                    st.caption("Check chat-model availability and Gemini quota. Your transcript index remains saved if indexing succeeded.")
else:
    st.info("Start by fetching captions, uploading a transcript file, or pasting transcript text.")

st.divider()
st.caption("Privacy: transcripts and FAISS indexes are stored on the app's local filesystem. Hosted containers may have ephemeral storage unless a persistent volume is configured. Do not upload private transcripts unless you are comfortable sending retrieved text to the Gemini API.")
