"""Main FastAPI application for SonarCore"""

import uvicorn
from fastapi import FastAPI, Depends, HTTPException

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
)
from Core.Schemas import AgentMessage
from .services import AgentManager
from .dependencies import get_agent_manager
from .utils import handle_audio_input
from typing import Optional, List
import time
from fastapi import UploadFile, File, Form, Query

app = FastAPI(
    title="SonarCore Agent API",
    description="Async API to interact with the multi-agent RAG workflow",
    version="1.0.0",
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
