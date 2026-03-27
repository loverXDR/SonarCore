# SonarCore API Documentation

The SonarCore API provides a unified interface for interacting with multi-agent RAG (Retrieval-Augmented Generation) workflows. It supports audio transcription, speaker diarization, and context-isolated chat sessions.

## Base URL
Default development URL: `http://localhost:8000`

---

## Sessions

### Create Session from Audio
`POST /sessions/audio`
Create a new agent session by processing an audio file.

**Request Body:**
- `audio_path` (string, required): Local path to the audio file on the server.
- `use_diarization` (boolean, optional): Enable speaker identification.
- `config` (object, optional): Override agent configuration.

**Response Example:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session created successfully"
}
```

---

### Create Session from Text
`POST /sessions/text`
Create a new agent session by processing raw text contents.

**Request Body:**
- `text` (string, required): Raw text content to process.
- `config` (object, optional): Override agent configuration.

**Response Example:**
```json
{
  "session_id": "a7b2c3d4-e5f6-7890-a1b2-c3d4e5f67890",
  "message": "Session created successfully"
}
```

---

### List Sessions
`GET /sessions`
Retrieve a list of all active/managed sessions.

**Response Example:**
```json
{
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "created_at": "2026-03-27T10:00:00Z",
      "last_active": "2026-03-27T10:05:22Z",
      "message_count": 4,
      "config": { ... }
    }
  ]
}
```

---

### Get Session Metadata
`GET /sessions/{session_id}`
Retrieve detailed information about a specific session.

**Response Example:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-03-27T10:00:00Z",
  "last_active": "2026-03-27T10:05:22Z",
  "message_count": 4,
  "config": { ... }
}
```

---

### Delete Session
`DELETE /sessions/{session_id}`
Permanently delete a session from memory.

**Response Example:**
```json
{
  "message": "Session deleted"
}
```

---

## Chat

### Chat with Agent
`POST /sessions/{session_id}/chat`
Send a message to an active agent and get a context-aware response.

**Request Body:**
- `message` (string, required): User's query.

**Response Example:**
```json
{
  "answer": "Based on the transcript, the speaker mentioned X...",
  "sources": [
    { "text": "Segment of transcript text...", "metadata": { "start": 12.5, "end": 15.0 } }
  ],
  "history": [
    { "role": "user", "content": "What was discussed?" },
    { "role": "assistant", "content": "Based on the transcript..." }
  ]
}
```

---

### Get Conversation History
`GET /sessions/{session_id}/history`
Retrieve the full list of messages in the session.

**Response Example:**
```json
[
  { "role": "user", "content": "Hello" },
  { "role": "assistant", "content": "Hi there! How can I help you with your transcription today?" }
]
```

---

## Transcription

### Direct Transcription
`POST /transcribe`
Convert audio to text directly.

**Response Example:**
```json
{
  "text": "Full transcribed text here...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a test.",
      "speaker": "Speaker 1"
    }
  ]
}
```

---

## System Health

### Health Check
`GET /health`
Returns the status of the API and its core dependencies.

**Response Example:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 1240.5,
  "dependencies": {
    "vector_store": "reachable",
    "llm_provider": "active"
  }
}
```

---

## Architecture Note
The API follows a **Repository Pattern** for session management. By default, it uses `InMemorySessionRepository`. This can be swapped for persistent storage (e.g., SQLite or JSON) in `Api/dependencies.py`.
