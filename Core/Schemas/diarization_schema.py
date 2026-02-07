from typing import List, Optional

from pydantic import BaseModel, Field


class PyannoteConfig(BaseModel):
    """Config for Pyannote Diarization"""

    auth_token: str = Field(..., description="HuggingFace auth token")
    model_name: str = "pyannote/speaker-diarization-community-1"
    device: str = "cpu"


class DiarizationSegment(BaseModel):
    """Single segment of diarization result"""

    start: float
    end: float
    speaker: str


class DiarizationResult(BaseModel):
    """Result of diarization"""

    segments: List[DiarizationSegment]
    info: dict = {}
