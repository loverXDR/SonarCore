# Sonar Core

Sonar Core is a full-service audio processing system that includes ASR, speaker diarization, semantic search, and summarization. 
This service processes meeting and call recordings to extract meaningful information and provide an interactive chat interface over the transcription data.

## Technologies Used

- Framework: FastAPI for asynchronous REST API handling.
- ASR (Speech Recognition): faster-whisper.
- Speaker Diarization: pyannote.audio.
- LLM Orchestration: LangGraph and LangChain for multi-agent workflows (QA and Summarization).
- RAG & Indexing: LlamaIndex for document chunking, indexing, and semantic search.
- Server: Uvicorn.

## RAG Pipeline Architecture
The system uses a Multi-Agent RAG (Retrieval-Augmented Generation) approach based on LangGraph. 
When text or audio is processed, it is chopped into smaller semantic chunks and vectorized into a LlamaIndex query engine database. 
Instead of fitting the entire transcription into the LLM context window, **the LLM is equipped with Semantic Search and Summarization Tools**.
When you ask a question during a chat session, the LLM calls these tools to search the index and fetch exactly the relevant pieces of the transcript it needs to answer you.

## Interactive CLI Interface

In addition to the core API, an interactive Python command-line interface is available for direct terminal usage without running the web server.

To launch the CLI menu, run:
```bash
sonar-cli
```
*(or `python -m Cli.main` if not installed via pip)*

**Features of the CLI**:
- **Option 1**: Standalone audio translation and diarization (prints to terminal).
- **Options 2 & 3**: Processes text/audio and drops you straight into an interactive, infinite AI chat loop (`You:` -> `Agent:`).
- Supports elegant session disposal upon exiting the chat loop.

## How to Run

1. Install Dependencies
Ensure you have Python 3.12+ installed, then install the project and its core dependencies:
```bash
pip install -e .
```

2. Set Environment Variables
Export necessary API keys and configurations:
```bash
export LLM_API_BASE="https://api.openai.com/v1/"
export LLM_API_KEY="your_api_key"
export HF_TOKEN="your_huggingface_token" # Required for Pyannote diarization
```

3. Start the Server
Run the application using Uvicorn (add --reload for development):
```bash
uvicorn Api.main:app --host 0.0.0.0 --port 8000
```

The OpenAPI documentation (Swagger UI) will automatically be available at `http://0.0.0.0:8000/docs`.