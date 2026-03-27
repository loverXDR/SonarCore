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

## API Endpoints

The API provides endpoints for transcription and interactive agent sessions.

### Direct Transcription
**POST /transcribe**
Transcribes audio directly and returns raw text along with segmented timestamps.
Supports exactly one of the following inputs:
- file: Multipart file upload.
- url: Direct link to an audio file.
- file_path: Local file path on the server.
- use_diarization: Boolean flag to enable speaker separation.

### Session Management
**POST /sessions/audio**
Processes an audio file, builds a RAG index, and initializes an interactive agent session.
Accepts audio via file, url, or file_path.

**POST /sessions/text**
Initializes an agent session from a raw text payload.

**DELETE /sessions/{session_id}**
Deletes an active session and frees memory.

### Agent Chat
**POST /sessions/{session_id}/chat**
Sends a text message to an active agent session to ask questions about the transcript or request a summary.

## How to Run

1. Install Dependencies
Ensure you have Python 3.12+ installed, then install the required packages:
```bash
pip install -r requirements.txt
```

2. Set Environment Variables
Export necessary API keys and configurations:
```bash
export LLM_API_BASE="https://gptproxy.recdev.ru:444/v1/"
export LLM_API_KEY="your_api_key"
export HF_TOKEN="your_huggingface_token" # Required for Pyannote diarization
```

3. Start the Server
Run the application using Uvicorn (add --reload for development):
```bash
uvicorn Api.main:app --host 0.0.0.0 --port 8000
```

The OpenAPI documentation (Swagger UI) will automatically be available at `http://0.0.0.0:8000/docs`.