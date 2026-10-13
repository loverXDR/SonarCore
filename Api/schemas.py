"""Pydantic schemas for the API"""

from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from Core.Schemas import (
    AgentConfig,
    AgentMessage,
    SummaryOptions,
    SummarySize,
    SummaryStyle,
)


class ProcessAudioRequest(BaseModel):
    """Request schema for creating a session from an audio or video file.

    Attributes:
        audio_path (str): The local path to the audio or video file to process.
        use_diarization (bool): Flag indicating whether to perform speaker diarization.
        config (Optional[AgentConfig]): Optional override configuration for the pipeline.
    """

    audio_path: str = Field(description="Path to the audio or video file on the server")
    use_diarization: bool = Field(
        default=False, description="Whether to use speaker diarization"
    )
    config: Optional[AgentConfig] = Field(
        default=None, description="Optional custom agent configuration"
    )


class ProcessTextRequest(BaseModel):
    """Request schema for creating a session from raw text.

    Attributes:
        text (str): The raw text content to process.
        config (Optional[AgentConfig]): Optional override configuration for the pipeline.
    """

    text: str = Field(description="Raw text content to process")
    config: Optional[AgentConfig] = Field(
        default=None, description="Optional custom agent configuration"
    )


class SessionResponse(BaseModel):
    """Response schema returned upon successfully creating a session.

    Attributes:
        session_id (str): The unique identifier for the newly created session.
        message (str): A descriptive success message.
    """

    session_id: str = Field(description="Unique identifier for the created session")
    message: str = Field(description="Status message")


class ChatMessageRequest(BaseModel):
    """Request schema for sending a message to an active chat session.

    Attributes:
        message (str): The message content from the user.
    """

    message: str = Field(description="User message text")


class ChatMessageResponse(BaseModel):
    """Response from the agent."""

    answer: str = Field(description="Agent's response text")
    sources: List[dict] = Field(default_factory=list, description="List of sources used")
    history: List[AgentMessage] = Field(
        default_factory=list, description="Conversation history"
    )


class SummaryRequest(SummaryOptions):
    """Request schema for generating a document summary.

    Attributes:
        language (str): ISO 639-1 language code of the summary.
        style (SummaryStyle): Summary presentation style.
        size (SummarySize): Target summary length.
        comment (Optional[str]): Additional user instructions for the summary.
    """


class SummaryResponse(BaseModel):
    """Response schema containing the generated summary.

    Attributes:
        summary (str): The generated summary text.
        language (str): Language code used for generation.
        style (SummaryStyle): Style used for generation.
        size (SummarySize): Size used for generation.
        comment (Optional[str]): Additional instructions used for generation.
    """

    summary: str = Field(description="Generated summary text")
    language: str = Field(description="Language code of the summary")
    style: SummaryStyle = Field(description="Summary presentation style")
    size: SummarySize = Field(description="Target summary length")
    comment: Optional[str] = Field(
        default=None, description="Additional user instructions for the summary"
    )


class TranscriptionSegment(BaseModel):
    """A segment of transcribed audio with timestamps."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """Response schema for the direct transcription endpoint.

    Attributes:
        text (str): The complete transcribed raw text.
        segments (List[TranscriptionSegment]): A detailed list of transcribed pieces with timestamps and optional speakers.
    """
    
    text: str = Field(description="The transcribed text")
    segments: List[TranscriptionSegment] = Field(
        default_factory=list, description="Transcription segments with timestamps"
    )


class SessionInfo(BaseModel):
    """Detailed information about an agent session."""

    session_id: str = Field(description="Unique session identifier")
    created_at: datetime = Field(description="Timestamp of session creation")
    last_active: datetime = Field(description="Timestamp of last activity")
    config: AgentConfig = Field(description="Configuration used for this session")
    message_count: int = Field(default=0, description="Number of messages in history")


class SessionListResponse(BaseModel):
    """Response containing a list of sessions."""

    sessions: List[SessionInfo] = Field(description="List of active sessions")


class HealthCheckResponse(BaseModel):
    """System health status response."""

    status: str = Field(description="Overall system status (e.g., 'ok')")
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="System uptime")
    dependencies: Dict[str, str] = Field(
        default_factory=dict, description="Status of major dependencies"
    )


class AsyncTaskResponse(BaseModel):
    """Response returned when an async background task is started.

    Attributes:
        task_id (str): Unique identifier to poll progress via SSE.
        message (str): Human-readable status message.
    """

    task_id: str = Field(description="Unique task identifier for progress tracking")
    message: str = Field(description="Status message")
