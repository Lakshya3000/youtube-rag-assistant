# YouTube RAG Assistant 🎬

**Ask questions about YouTube videos using Retrieval-Augmented Generation (RAG).**

🔴 **Live Demo:** https://youtube-rag-assistant-p8he.onrender.com  
💻 **GitHub Repository:** https://github.com/Lakshya3000/youtube-rag-assistant

A Streamlit application that lets users ask questions about YouTube video transcripts. It uses **LangChain, Google Gemini, Gemini embeddings, FAISS, and Maximal Marginal Relevance (MMR)** to retrieve relevant transcript passages and generate grounded answers with inspectable sources.

> **Note:** Direct YouTube caption fetching may fail on hosted infrastructure because YouTube can block cloud-hosting IP addresses. You can upload a transcript file or paste transcript text as an alternative.

## ✨ Features

- **YouTube transcript input:** Fetch captions using a YouTube URL or video ID when network access permits.
- **Multiple transcript options:** Upload `.txt`, `.md`, `.srt`, or `.vtt` files, or paste transcript text.
- **Retrieval-Augmented Generation:** Generate answers grounded in transcript content.
- **Gemini integration:** Use Google Gemini for embeddings and answer generation.
- **FAISS vector search:** Store and retrieve transcript embeddings efficiently.
- **MMR retrieval:** Balance relevance and diversity to reduce repetitive context.
- **Configurable retrieval:** Adjust chunk size, overlap, embedding model, chat model, and the number of retrieved chunks.
- **Index caching:** Reuse compatible indexes to avoid unnecessary embedding API calls.
- **Source inspection:** Expand the sources used to inspect the retrieved transcript passages.
- **Docker support:** Run the application using Docker or Docker Compose.

## 🖥️ Live Demo

Try the deployed application here:

**[Open YouTube RAG Assistant](https://youtube-rag-assistant-p8he.onrender.com)**

To test it:
1. Paste a transcript or upload a transcript file.
2. Click **Use pasted transcript** or submit the uploaded transcript.
3. Click **Build / load searchable index**.
4. Ask a question about the video.
5. Expand **Sources used** to inspect the supporting passages.

**Deployment note:** Render's free instance may spin down when inactive, so the first request after inactivity can take longer. Local FAISS indexes may also need to be rebuilt if their files are lost after a restart or redeployment.

## 🏗️ Architecture

```text
YouTube URL / Transcript Upload / Pasted Text
                    |
                    v
              Transcript Text
                    |
                    v
       RecursiveCharacterTextSplitter
                    |
                    v
          Gemini Embedding Model
                    |
                    v
               FAISS Index
                    |
Question ----------> MMR Retriever
                    |
                    v
         Relevant Transcript Chunks
                    |
                    v
            Context + Question
                    |
                    v
            Gemini Chat Model
                    |
                    v
          Answer + Source Passages
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Application development |
| Streamlit | Interactive web interface |
| LangChain | RAG pipeline and retrieval |
| Google Gemini | Answer generation |
| Gemini Embeddings | Convert text into vector representations |
| FAISS | Vector storage and similarity search |
| MMR | Relevant and diverse chunk retrieval |
| Docker | Containerization |
| Docker Compose | Local container orchestration |
| GitHub | Version control and source code hosting |
| Render | Cloud deployment |

## 📋 Requirements

- Python 3.11
- Google Gemini API key
- Internet connection for Gemini API requests
- Docker Desktop (optional, for containerized execution)

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Lakshya3000/youtube-rag-assistant.git
cd youtube-rag-assistant
```

### 2. Activate your Python environment

For Conda:

```powershell
conda activate aiml-env
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

### 3. Configure environment variables

Create your local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure your API key and model names:

```env
GOOGLE_API_KEY=your_real_api_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
RAG_INDEX_DIR=data/indexes
```

**Security:** Replace the placeholder with your own API key locally. Never commit `.env` or expose your API key in source code, screenshots, or public repositories.

Model availability may vary by API account and region. Use a currently supported model if your configured model is unavailable.

### 4. Start the application

```powershell
streamlit run app.py
```

Open the URL shown in your terminal, normally:

http://localhost:8501

### 5. Use the application

1. Fetch captions, upload a transcript, or paste transcript text.
2. Adjust chunk settings and the number of retrieved chunks if required.
3. Click **Build / load searchable index**.
4. Wait for indexing to complete.
5. Ask questions about the transcript.
6. Expand **Sources used** to inspect the evidence behind the answer.

## 🧠 Retrieval Strategy: Maximal Marginal Relevance (MMR)

This project uses **Maximal Marginal Relevance (MMR)** to select transcript chunks for answering questions.

Traditional similarity search prioritizes chunks most similar to the query. MMR also considers diversity, helping reduce redundant passages in the retrieved context.

The retrieval configuration in `src/rag_engine.py` uses:

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
| `k` | Number of final chunks returned |
| `fetch_k` | Number of candidate chunks considered |
| `lambda_mult` | Balance between relevance and diversity |

The default final chunk count is 4, the candidate pool is at least 20 chunks, and `lambda_mult=0.5` provides a balanced starting point.

MMR does not guarantee better answers for every question. Answer quality also depends on transcript quality, chunking, retrieval settings, and the language model.

## 🐳 Run with Docker

Ensure Docker Desktop is running and your `.env` file is configured.

Build and start the application:

```powershell
docker compose up --build
```

Open:

http://localhost:8501

Stop the application:

```powershell
docker compose down
```

The local Docker Compose configuration mounts the `data` directory into the container so generated indexes can persist across container restarts.

## ☁️ Deployment

This application is deployed on **Render using Docker**.

- **Live application:** https://youtube-rag-assistant-p8he.onrender.com
- **Source code:** https://github.com/Lakshya3000/youtube-rag-assistant

The deployment uses environment variables configured in Render for the Gemini API key and model settings.

### Deployment considerations

- Hosted services may have limited resources and may spin down when inactive.
- Do not assume local FAISS index files will survive every restart or redeployment.
- If the index is missing, the transcript may need to be indexed again, consuming Gemini embedding quota.
- YouTube may block automatic transcript requests from cloud-hosting IP addresses.
- Uploading or pasting a transcript is the available fallback when direct caption fetching fails.

## 🔐 Environment Variables

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Google Gemini API credential |
| `GEMINI_API_KEY` | Alternative API-key name, if supported by the application |
| `GEMINI_EMBEDDING_MODEL` | Gemini embedding model |
| `GEMINI_CHAT_MODEL` | Gemini chat model |
| `RAG_INDEX_DIR` | Directory used to store FAISS indexes |

Keep API keys in environment variables or your hosting provider's secret settings. Never commit real credentials to GitHub.

## ⚠️ Limitations

- YouTube caption fetching can fail when requests are blocked or captions are unavailable.
- Hosted filesystems may not preserve FAISS indexes across restarts or redeployments.
- Gemini API quotas, latency, and availability depend on the selected model and account.
- Generated answers are based on retrieved transcript passages and may not capture information missing from the transcript.

## 🔧 Troubleshooting

**YouTube transcript fetch fails**

Try uploading a transcript file or pasting transcript text instead.

**Gemini API key missing**

Verify that `GOOGLE_API_KEY` is configured in your local `.env` file or the Render environment settings.

**Model not found**

Configure `GEMINI_CHAT_MODEL` with a model currently available to your API account.

**Quota error (`429 RESOURCE_EXHAUSTED`)**

Check your Google AI Studio quota and avoid repeatedly rebuilding an index when a compatible cached index is available.

**FAISS index missing**

Build the searchable index again from the transcript. Re-indexing may consume additional embedding quota.

## 🔒 Privacy and Security

- Transcript text is sent to Gemini for embedding and answer generation.
- Do not submit private transcripts unless you have permission to process them.
- Never load untrusted FAISS pickle files.
- For production use, consider authentication, rate limiting, persistent storage, monitoring, and data-retention controls.

## 👨‍💻 Author

**Lakshya Pratap Singh**

- GitHub: https://github.com/Lakshya3000

---

⭐ If you find this project useful, consider giving the repository a star.

