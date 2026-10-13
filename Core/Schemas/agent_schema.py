"""Schemas for Agent Workflow"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .rag_schema import LLMConfig, IndexConfig
from .asr_schema import ASRConfig, default_asr_config
from .diarization_schema import DiarizationConfig, default_diarization_config


class AgentConfig(BaseModel):
    """Combined config for agent pipeline"""

    llm: LLMConfig
    index: IndexConfig = Field(default_factory=IndexConfig)
    asr: Optional[ASRConfig] = Field(default_factory=default_asr_config)
    diarization: Optional[DiarizationConfig] = Field(
        default_factory=default_diarization_config
    )
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


class MinutesParticipant(BaseModel):
    """Meeting participant identified in the transcript"""

    speaker: str = Field(
        ...,
        description="Speaker label from the transcript (e.g. 'SPEAKER_00') or the name if there are no labels",
    )
    name: Optional[str] = Field(
        default=None,
        description="Real name if it is mentioned or can be inferred from the transcript",
    )
    role: Optional[str] = Field(
        default=None,
        description="Role or position in the meeting if it follows from the transcript",
    )


class MinutesAgendaItem(BaseModel):
    """Single discussed agenda item"""

    title: str = Field(..., description="Short title of the discussed item")
    discussion: str = Field(
        ...,
        description="Course of discussion: who proposed what and the key arguments",
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Key facts, figures and positions on this item",
    )


class MinutesActionItem(BaseModel):
    """Assigned task with an owner and a deadline"""

    task: str = Field(..., description="What has to be done")
    owner: Optional[str] = Field(default=None, description="Responsible person or speaker")
    deadline: Optional[str] = Field(default=None, description="Deadline as stated in the meeting")


class MeetingMinutes(BaseModel):
    """Structured meeting minutes extracted from a transcript"""

    title: str = Field(..., description="Meeting topic in a few words")
    summary: str = Field(
        ...,
        description="Overview of the meeting in 2-4 sentences: purpose, main topics and outcome",
    )
    participants: List[MinutesParticipant] = Field(default_factory=list)
    agenda: List[MinutesAgendaItem] = Field(
        default_factory=list,
        description="Discussed items in the order they appear in the meeting",
    )
    decisions: List[str] = Field(default_factory=list, description="Decisions made")
    action_items: List[MinutesActionItem] = Field(default_factory=list)
    open_questions: List[str] = Field(
        default_factory=list, description="Questions that remain unresolved"
    )
