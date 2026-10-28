# YouTube RAG Assistant 🎬

A Streamlit application that lets users ask questions about YouTube video transcripts. It uses **LangChain**, **Google Gemini**, **Gemini embeddings**, and **FAISS** for Retrieval-Augmented Generation (RAG). The application displays retrieved source passages so users can verify answers against the transcript.

## Features

- Fetch captions from a YouTube URL or video ID when network access permits.
- Upload `.txt`, `.md`, `.srt`, or `.vtt` transcript files, or paste transcript text.
- Split transcript text into overlapping chunks and embed them in batches.
- Cache FAISS indexes by transcript content, embedding model, chunk size, and overlap.
- Use **Maximal Marginal Relevance (MMR)** to retrieve relevant chunks while reducing redundant context.
- Configure the number of retrieved chunks and other retrieval settings.
- Ask questions through a chat interface and inspect the source chunks used to generate answers.
- Configure Gemini chat and embedding model names through environment variables or the sidebar.
- Run locally or with Docker and Docker Compose.
- Support health checks, a non-root container user, and documented deployment options.

## Architecture

```text
YouTube URL / transcript upload / pasted text
                    |
                    v
              Transcript text
                    |
                    v
       RecursiveCharacterTextSplitter
                    |
                    v
       Gemini embedding model (batched)
                    |
                    v
           Persistent FAISS index
                    |
Question ---> MMR retriever
                    |
          Candidate chunks (fetch_k)
                    |
                    v
     Relevant, less-redundant chunks (k)
                    |
                    v
        Context + question prompt
                    |
                    v
            Gemini chat model
                    |
                    v
       Answer + inspectable sources
```

## 1. Requirements

- Python 3.11 recommended
- A Gemini API key from Google AI Studio
- Internet access to call the Gemini API
- Docker Desktop (optional, for container deployment)

YouTube caption fetching may fail when YouTube blocks a network/IP or when captions are disabled. This is not necessarily an application bug. Use the transcript upload or paste options as a fallback.

## 2. Run locally in VS Code (Conda `aiml-env`)

Open PowerShell in the project directory:

```powershell
conda activate aiml-env
python --version
python -m pip install -r requirements.txt
python -m pip check
```

Create `.env` from the example:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and add your real API key. Never commit `.env` to Git.

```env
GOOGLE_API_KEY=your_real_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
RAG_INDEX_DIR=data/indexes
```

Model availability can vary by API account and region. If `gemini-3.5-flash-lite` is unavailable to your API key, configure `GEMINI_CHAT_MODEL` with a currently supported model.

Start the application:

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal, normally `http://localhost:8501`.

### Typical workflow

1. Fetch captions, upload a transcript, or paste transcript text.
2. Configure the chunk settings and number of retrieved chunks in the sidebar.
3. Click **Build / load searchable index**.
4. Wait for indexing to complete. A cached index is reused when the transcript and indexing settings match.
5. Ask questions and expand **Sources used** to inspect the supporting passages.

## 3. Retrieval strategy: MMR

The application uses **Maximal Marginal Relevance (MMR)** retrieval with FAISS.

Traditional similarity search selects chunks based primarily on their semantic similarity to the question. MMR considers a wider set of candidates and balances relevance with diversity, helping reduce repetitive passages in the final context.

The current configuration in `src/rag_engine.py` is:

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": final_k,
        "fetch_k": fetch_k,
        "lambda_mult": 0.5,
    },
)
```

| Parameter | Purpose |
|---|---|
| `search_type="mmr"` | Enables MMR retrieval |
| `k` | Number of final chunks returned |
| `fetch_k` | Number of candidate chunks considered before selecting the final results |
| `lambda_mult` | Balances relevance and diversity |

The default final chunk count is 4 when `build_rag_chain()` is called without a different value. The candidate pool is at least 20 chunks, and `lambda_mult=0.5` provides a balanced starting point.

- Increase `lambda_mult` toward `1.0` to prioritize relevance.
- Decrease it toward `0.0` to prioritize diversity.
- Increase `k` when answers need context from more parts of a transcript, but remember that larger contexts can increase token usage and latency.

MMR does not guarantee better answers for every question. Evaluate it against similarity search using factual questions, broad summaries, exact terms, and questions whose answers are absent from the transcript.

**Existing indexes:** Switching from similarity search to MMR does not require generating embeddings again, provided the existing FAISS index is compatible with the current transcript and embedding configuration.

## 4. Docker

Ensure Docker Desktop is running and create `.env` as described above.

Build and start:

```powershell
docker compose up --build
```

Open `http://localhost:8501`.

Stop:

```powershell
docker compose down
```

The Compose configuration mounts `./data` into the container, allowing locally generated indexes to persist across container restarts. To rebuild an index from scratch, stop the application and delete the relevant subfolder under `data/indexes`.

Build and run without Compose:

```powershell
docker build -t youtube-rag-assistant:1.0.0 .
docker run --rm -p 8501:8501 --env-file .env -v "${PWD}/data:/app/data" -e RAG_INDEX_DIR=/app/data/indexes youtube-rag-assistant:1.0.0
```

## 5. Push to GitHub

Create an empty repository on GitHub, then run these commands from the project directory:

```powershell
git init
git add .
git status
git commit -m "Add MMR retrieval to YouTube RAG assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/youtube-rag-assistant.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If the repository already exists locally, do not repeat `git init` or add another remote unnecessarily. Instead, stage and commit your changes:

```powershell
git add src/rag_engine.py README.md
git status
git commit -m "Add MMR retrieval and update documentation"
git push
```

Before committing, inspect `git status` and ensure `.env` is not listed. Never commit API keys, private transcripts, or generated FAISS cache files.

## 6. Deploy it live

### Option A — Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud and create a new app from the repository.
3. Set the main file path to `app.py` and deploy.
4. Add `GOOGLE_API_KEY`, `GEMINI_EMBEDDING_MODEL`, and `GEMINI_CHAT_MODEL` in the app's Secrets/settings area using the host's supported format.
5. Test with a small transcript first.

Hosted filesystem storage may be ephemeral. If the app restarts or is redeployed, its local FAISS cache may disappear. Users can rebuild the index from the transcript. YouTube caption fetching may also be blocked on hosted infrastructure, so transcript upload is an important fallback.

### Option B — Docker host (for example, Render or another container platform)

1. Create a web service from the GitHub repository and select Docker as the runtime.
2. Set `GOOGLE_API_KEY` as a secret/environment variable. Set the model variables if needed.
3. Configure the service to use the container's port `8501`, or its injected `PORT` environment variable if the application and container configuration support it.
4. If persistent disks or volumes are supported, mount one at `/app/data` and set `RAG_INDEX_DIR=/app/data/indexes`.
5. Deploy and test the public URL.

Free-tier instances may sleep, have resource limits, or lose local files on redeploy. Do not assume that a local FAISS index persists unless persistent storage is configured.

## 7. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API credential |
| `GEMINI_API_KEY` | — | Alternative key name, if supported by the application |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Text embedding model |
| `GEMINI_CHAT_MODEL` | `gemini-3.5-flash-lite` | Chat generation model |
| `RAG_INDEX_DIR` | `data/indexes` | Local FAISS cache directory |

## 8. Security and production notes

- Store secrets in `.env` locally and in the hosting provider's secret manager in production. Never hard-code or commit them.
- FAISS pickle deserialization is enabled for the application's local index cache. Never load untrusted index files.
- Transcript content is sent to Gemini through embedding and answer-generation API calls. Avoid processing sensitive content unless you have permission and understand the provider's terms.
- This project is suitable as a portfolio/demo deployment. For higher-traffic production use, consider authentication, request quotas, observability, persistent managed storage, and explicit data-retention controls.
- API quotas and pricing are controlled by Google and can change. Smaller embedding batches help manage request pressure but do not bypass exhausted quotas.
- MMR changes retrieval behavior, not the underlying embedding vectors. Answer quality still depends on transcript quality, chunking, retrieval settings, and the language model.

## 9. Troubleshooting

**`GOOGLE_API_KEY` missing**

Check the `.env` location and confirm it contains `GOOGLE_API_KEY=...`.

**Model not found (404)**

Set `GEMINI_CHAT_MODEL` to a model currently available to your API key, then restart the application.

**Quota / `429 RESOURCE_EXHAUSTED`**

Check Google AI Studio/API limits and wait for quota reset or use a plan with sufficient quota. Avoid repeatedly rebuilding the same index.

**YouTube `RequestBlocked`**

Hosted/cloud IPs can be blocked. Upload or paste a transcript instead.

**FAISS loads but retrieval seems wrong**

Ensure the transcript, embedding model, and chunk settings match the intended index. The cache key changes when the transcript or indexing configuration changes