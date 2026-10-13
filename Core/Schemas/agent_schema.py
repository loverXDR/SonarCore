"""Schemas for Agent Workflow"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .rag_schema import LLMConfig, IndexConfig
from .asr_schema import ASRConfig, WhisperConfig
from .diarization_schema import PyannoteConfig


class AgentConfig(BaseModel):
    """Combined config for agent pipeline"""

    llm: LLMConfig
    index: IndexConfig = Field(default_factory=IndexConfig)
    asr: Optional[ASRConfig] = WhisperConfig()
    diarization: Optional[PyannoteConfig] = None
    system_prompt: str = (
        "Ты — интеллектуальный ассистент, работающий "
        "с транскрипцией аудиозаписи. "
        "Используй инструменты поиска и суммаризации, "
        "чтобы отвечать на вопросы пользователя "
        "по содержимому документа. "
        "Отвечай на языке пользователя."
    )


class AgentMessage(BaseModel):
    """Single message in agent conversation"""

    role: str = Field(
        ...,
        description="Role: user, assistant, or system",
    )
    content: str


class AgentResponse(BaseModel):
    """Response from agent chat"""

    answer: str
    sources: List[dict] = []
    history: List[AgentMessage] = []


class SummaryStyle(str, Enum):
    """Presentation style of a generated summary"""

    BRIEF = "brief"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    MEETING_MINUTES = "meeting_minutes"


class SummarySize(str, Enum):
    """Target length of a generated summary"""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SummaryOptions(BaseModel):
    """Parameters controlling summary generation"""

    language: str = Field(
        default="ru",
        min_length=2,
        max_length=5,
        description="ISO 639-1 language code of the summary (e.g. 'ru', 'en')",
    )
    style: SummaryStyle = Field(
        default=SummaryStyle.BRIEF,
        description="Summary presentation style",
    )
    size: SummarySize = Field(
        default=SummarySize.MEDIUM,
        description="Target summary length",
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Additional user instructions for the summary",
    )
