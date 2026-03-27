"""Schemas for Agent Workflow"""

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
