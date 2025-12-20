# Sonar Core

*Sonar Core* is a full-service audio processing system that includes ASR, speaker diarization, semantic search, and summarization via LLMs.  
This service is especially useful for meetings and calls where you need to extract meaningful information without unnecessary or irrelevant context.

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