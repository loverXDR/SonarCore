"""Main FastAPI application for SonarCore"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Optional, List

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi import UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .schemas import (
    ProcessAudioRequest,
    ProcessTextRequest,
    SessionResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    TranscriptionResponse,
    SessionInfo,
    SessionListResponse,
    HealthCheckResponse,
    AsyncTaskResponse,
    SummaryRequest,
    SummaryResponse,
)
from Core.Schemas import AgentMessage
from .services import AgentManager
from .dependencies import get_agent_manager
from .utils import handle_audio_input

logger = logging.getLogger(__name__)

_STREAM_CHUNK_WORDS = 3
_STREAM_CHUNK_DELAY = 0.02
_PROGRESS_POLL_INTERVAL = 0.5
_PROGRESS_TIMEOUT = 600

app = FastAPI(
    title="SonarCore Agent API",
    description="Async API to interact with the multi-agent RAG workflow",
    version="1.0.0",
)

# Configure CORS with explicit origins (wildcard + credentials is invalid per browser policy)
allow_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Cache-Control", "X-Accel-Buffering", "Content-Type"],
)

start_time = time.time()


@app.post("/sessions/audio", response_model=SessionResponse, tags=["Sessions"])
async def create_session_from_audio(
    request: ProcessAudioRequest,
    manager: AgentManager = Depends(get_agent_manager),
):
    """Create a new agent session by processing an audio file.

    Args:
        request (ProcessAudioRequest): HTTP Request containing audio details.
        manager (AgentManager, optional): Injected dependency. Defaults to Depends(get_agent_manager).

    Raises:
        HTTPException: On internal processing failure.

    Returns:
        SessionResponse: The created session details.
    """
    try:
        session_id = await manager.create_session_from_audio(
            audio_path=request.audio_path,
            use_diarization=request.use_diarization,
            config=request.config,
        )
        return SessionResponse(
            session_id=session_id, message="Session created from audio."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=SessionListResponse, tags=["Sessions"])
async def list_sessions(manager: AgentManager = Depends(get_agent_manager)):
    """List all active agent sessions."""
    return SessionListResponse(sessions=manager.get_all_sessions())


@app.get("/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def get_session(
    session_id: str, manager: AgentManager = Depends(get_agent_manager)
):
    """Get metadata for a specific session."""
    try:
        return manager.get_session_details(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/sessions/text", response_model=SessionResponse, tags=["Sessions"])
async def create_session_from_text(
    request: ProcessTextRequest,
    manager: AgentManager = Depends(get_agent_manager),
):
    """Create a new agent session by processing raw text.

    Args:
        request (ProcessTextRequest): HTTP request containing raw text.
        manager (AgentManager, optional): Dependency. Defaults to Depends(get_agent_manager).

    Raises:
        HTTPException: On internal processing failure.

    Returns:
        SessionResponse: The created session details.
    """
    try:
        session_id = await manager.create_session_from_text(
            text=request.text, config=request.config
        )
        return SessionResponse(
            session_id=session_id, message="Session created from text."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/sessions/{session_id}/chat",
    response_model=ChatMessageResponse,
    tags=["Chat"],
)
async def chat_with_agent(
    session_id: str,
    request: ChatMessageRequest,
    manager: AgentManager = Depends(get_agent_manager),
):
    """Send a message to an active agent session.

    Args:
        session_id (str): Unique identifier for the active session.
        request (ChatMessageRequest): Request containing the user payload.
        manager (AgentManager, optional): Dependency. Defaults to Depends(get_agent_manager).

    Raises:
        HTTPException: If the session fails or cannot be found.

    Returns:
        ChatMessageResponse: Result containing text and conversation history.
    """
    try:
        response = await manager.chat(session_id=session_id, message=request.message)
        return ChatMessageResponse(
            answer=response.answer,
            sources=response.sources,
            history=response.history,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/sessions/{session_id}/summary",
    response_model=SummaryResponse,
    tags=["Summary"],
)
async def summarize_session(
    session_id: str,
    request: SummaryRequest,
    manager: AgentManager = Depends(get_agent_manager),
):
    """Generate a summary of the session document with explicit parameters.

    Args:
        session_id (str): Unique identifier for the active session.
        request (SummaryRequest): Summary language, style and size.
        manager (AgentManager, optional): Dependency. Defaults to Depends(get_agent_manager).

    Raises:
        HTTPException: If the session is not found or generation fails.

    Returns:
        SummaryResponse: Generated summary with the applied parameters.
    """
    try:
        summary = await manager.summarize(session_id=session_id, options=request)
        return SummaryResponse(
            summary=summary,
            language=request.language,
            style=request.style,
            size=request.size,
            comment=request.comment,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/stream", tags=["Chat"])
async def stream_chat(
    session_id: str,
    message: str = Query(..., description="User message to send to the agent"),
    manager: AgentManager = Depends(get_agent_manager),
):
    """Stream a chat response as Server-Sent Events.

    The agent is invoked in a background thread. Once the full answer is
    available it is split into word-level chunks and streamed to the client
    with a short delay between each chunk for smooth UX.

    SSE event types:
        * ``token``  -- partial answer text
        * ``done``   -- final payload with sources and history
        * ``error``  -- on failure
    """

    async def _event_generator():
        try:
            # Run the synchronous agent.chat in a thread so we don't block
            response = await manager.chat(session_id=session_id, message=message)

            # Split answer into small chunks for fake token streaming
            words = response.answer.split()
            for i in range(0, len(words), _STREAM_CHUNK_WORDS):
                chunk = " ".join(words[i : i + _STREAM_CHUNK_WORDS])
                # Add trailing space unless this is the last chunk
                if i + _STREAM_CHUNK_WORDS < len(words):
                    chunk += " "
                payload = json.dumps({"type": "token", "content": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(_STREAM_CHUNK_DELAY)

            # Final event with metadata
            done_payload = json.dumps(
                {
                    "type": "done",
                    "sources": response.sources,
                    "history": [m.model_dump() for m in response.history],
                },
                ensure_ascii=False,
            )
            yield f"data: {done_payload}\n\n"

        except ValueError as exc:
            error_payload = json.dumps({"type": "error", "detail": str(exc)})
            yield f"data: {error_payload}\n\n"
        except Exception as exc:
            logger.exception("Streaming chat error for session %s", session_id)
            error_payload = json.dumps({"type": "error", "detail": str(exc)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/sessions/{session_id}/history",
    response_model=List[AgentMessage],
    tags=["Chat"],
)
async def get_session_history(
    session_id: str, manager: AgentManager = Depends(get_agent_manager)
):
    """Retrieve full conversation history for a session."""
    try:
        return manager.get_session_history(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(
    session_id: str,
    manager: AgentManager = Depends(get_agent_manager),
):
    """Delete an active agent session.

    Args:
        session_id (str): Session identifier to delete.
        manager (AgentManager, optional): Dependency. Defaults to Depends(get_agent_manager).

    Raises:
        HTTPException: If the session is not found.

    Returns:
        dict: Success message.
    """
    success = manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}


@app.post(
    "/sessions/audio/async",
    response_model=AsyncTaskResponse,
    tags=["Sessions"],
)
async def create_session_from_audio_async(
    request: ProcessAudioRequest,
    background_tasks: BackgroundTasks,
    manager: AgentManager = Depends(get_agent_manager),
):
    """Start audio session creation as a background task.

    Returns a ``task_id`` immediately. Poll ``GET /sessions/progress/{task_id}``
    via SSE to monitor progress.
    """
    task_id = str(uuid.uuid4())

    async def _run_task() -> None:
        try:
            await manager.create_session_from_audio(
                audio_path=request.audio_path,
                use_diarization=request.use_diarization,
                config=request.config,
                task_id=task_id,
            )
        except Exception:
            logger.exception("Background audio processing failed for task %s", task_id)

    background_tasks.add_task(_run_task)
    return AsyncTaskResponse(task_id=task_id, message="Processing started. Poll progress via SSE.")


@app.get("/sessions/progress/{task_id}", tags=["Sessions"])
async def get_session_progress(task_id: str):
    """Stream progress updates for an async session creation task via SSE.

    Emits a JSON event every ~500 ms until the task reaches ``ready`` or
    ``error`` status, or until the timeout is exceeded.

    SSE event types:
        * ``progress`` -- intermediate status with ``pct`` field
        * ``ready``    -- final event containing the ``session_id``
        * ``error``    -- task failed; ``detail`` field has info
        * ``timeout``  -- server gave up waiting
    """

    async def _progress_generator():
        elapsed = 0.0
        while elapsed < _PROGRESS_TIMEOUT:
            progress = AgentManager.get_task_progress(task_id)

            if progress is None:
                payload = json.dumps({"type": "progress", "status": "queued", "pct": 0})
                yield f"data: {payload}\n\n"
            elif progress.get("status") == "ready":
                payload = json.dumps(
                    {
                        "type": "ready",
                        "status": "ready",
                        "pct": 100,
                        "session_id": progress["session_id"],
                    }
                )
                yield f"data: {payload}\n\n"
                return
            elif progress.get("status") == "error":
                payload = json.dumps(
                    {
                        "type": "error",
                        "status": "error",
                        "detail": progress.get("detail", "Unknown error"),
                    }
                )
                yield f"data: {payload}\n\n"
                return
            else:
                payload = json.dumps(
                    {
                        "type": "progress",
                        "status": progress.get("status", "unknown"),
                        "pct": progress.get("pct", 0),
                    }
                )
                yield f"data: {payload}\n\n"

            await asyncio.sleep(_PROGRESS_POLL_INTERVAL)
            elapsed += _PROGRESS_POLL_INTERVAL

        # Timeout reached
        payload = json.dumps({"type": "timeout", "detail": "Progress tracking timed out."})
        yield f"data: {payload}\n\n"

    return StreamingResponse(
        _progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Transcription"])
async def transcribe_audio(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    file_path: Optional[str] = Form(None),
    use_diarization: bool = Form(False),
    manager: AgentManager = Depends(get_agent_manager),
):
    """
    Directly transcribe audio to text.

    Accepts exactly one input method from (file, url, file_path).

    Args:
        file (UploadFile, optional): Uploaded local file bytes. Defaults to File(None).
        url (str, optional): Remote url to download. Defaults to Form(None).
        file_path (str, optional): Pre-existing local server path. Defaults to Form(None).
        use_diarization (bool, optional): Enable speaker identification. Defaults to Form(False).
        manager (AgentManager, optional): Dependency. Defaults to Depends(get_agent_manager).

    Raises:
        HTTPException: On bad input combinations or processing failures.

    Returns:
        TranscriptionResponse: Raw parsed text and segment blocks.
    """
    try:
        async with handle_audio_input(file=file, url=url, file_path=file_path) as path:
            result = await manager.transcribe_audio(
                audio_path=path, use_diarization=use_diarization, config=None
            )
            return TranscriptionResponse(
                text=result["text"], segments=result["segments"]
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check(manager: AgentManager = Depends(get_agent_manager)):
    """Check API health and status of real dependencies."""
    dependency_status = await manager.check_health()
    
    # Overall status is 'error' if any critical dependency is unreachable
    status = "ok"
    for dep_status in dependency_status.values():
        if "unreachable" in dep_status:
            status = "error"
            break

    return HealthCheckResponse(
        status=status,
        version="1.0.0",
        uptime_seconds=time.time() - start_time,
        dependencies=dependency_status,
    )


if __name__ == "__main__":
    uvicorn.run("Api.main:app", host="0.0.0.0", port=8000, reload=True)
