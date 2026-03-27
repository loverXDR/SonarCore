"""Pydantic schemas for the API"""

from typing import List, Optional
from pydantic import BaseModel, Field

from Core.Schemas import AgentConfig, AgentMessage


class ProcessAudioRequest(BaseModel):
    """Request schema for creating a session from an audio file.

    Attributes:
        audio_path (str): The local path to the audio file to process.
        use_diarization (bool): Flag indicating whether to perform speaker diarization.
        config (Optional[AgentConfig]): Optional override configuration for the pipeline.
    """

    audio_path: str = Field(description="Path to the audio file on the server")
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
