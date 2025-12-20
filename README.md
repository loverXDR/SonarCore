# Sonar Core

Sonar Core - service full audio processing: ASR, diarization, semantic search and summarization via LLM

this service are useful for meetings, calls when you need to understand information without dirty context

## Key features

* **High performance ASR** - Faster-Whisper/Sherpa-ONNX
* **Diarization** -Pyannote/speaker embeddings
* **RAG** - Fast semantic search of useful data from speech
* **LLM** - Response generation and summarization pipelines

## Technical stack

* **Core** - Python3.12, Asyncio, FastAPI
* **ASR** - Faster-Whisper/Sherpa-onnx
* **LLM+RAG** - LangChain + Qdrant
* **Diarization** - Pyannote
* **Infra** - Docker/Docker Compose
* 