# Sonar Core

Sonar Core is a full-service audio processing system that includes ASR, speaker diarization, semantic search, and summarization. 
This service processes meeting and call recordings to extract meaningful information and provide an interactive chat interface over the transcription data.

## Technologies Used

- **Framework**: FastAPI for asynchronous REST API handling.
- **ASR (Speech Recognition)**: `faster-whisper`.
- **Speaker Diarization**: `pyannote.audio`.
- **LLM Orchestration**: LangGraph and LangChain for multi-agent workflows (QA and Summarization).
- **RAG & Indexing**: LlamaIndex for document chunking and semantic search.
- **Vector Database**: **Qdrant** (for persistent storage and fast retrieval).
- **Server**: Uvicorn.

## RAG Pipeline Architecture
The system uses a Multi-Agent RAG (Retrieval-Augmented Generation) approach based on LangGraph. 
When text or audio is processed, it is chopped into smaller semantic chunks and vectorized into a Qdrant collection managed by LlamaIndex. 
Instead of fitting the entire transcription into the LLM context window, **the LLM is equipped with Semantic Search and Summarization Tools**.
When you ask a question during a chat session, the LLM calls these tools to search the index and fetch exactly the relevant pieces of the transcript it needs to answer you.

## Quick Start (Docker)

The fastest way to run the entire stack (API + Qdrant) is using Docker Compose.

1. **Configure Environment**
   Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Fill in your `LLM_API_KEY` and `HF_TOKEN` (required for Pyannote).

2. **Launch Stack**
   ```bash
   docker-compose up -d --build
   ```
   - API will be available at: `http://localhost:8000`
   - Qdrant UI will be available at: `http://localhost:6333/dashboard`
   - The API uses **hot-reload**, so any changes to the source code will be applied immediately.

## Local Development

If you prefer to run services manually:

1. **Install Dependencies**
   Ensure you have Python 3.12+ installed:
   ```bash
   pip install -e .
   ```

2. **Run Qdrant**
   The system requires a running Qdrant instance:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Start the API**
   ```bash
   export LLM_API_KEY="your_key"
   export HF_TOKEN="your_token"
   uvicorn Api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Interactive CLI Interface

In addition to the core API, an interactive Python command-line interface is available for direct terminal usage.

To launch the CLI menu, run:
```bash
sonar-cli
```
*(or `python -m Cli.main` if not installed via pip)*

**Features of the CLI**:
- **Option 1**: Standalone audio transcription and diarization.
- **Options 2 & 3**: Processes text/audio and starts an interactive AI chat loop.
- Supports elegant session disposal upon exiting.

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
