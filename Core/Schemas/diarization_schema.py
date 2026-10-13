"""Schemas for Diarization utilities"""

import os
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class BaseDiarizationConfig(BaseModel):
    """common config fields for diarization backends"""

    model_type: str
    device: str = "cpu"


class PyannoteConfig(BaseDiarizationConfig):
    """Config for Pyannote Diarization"""

    model_type: Literal["pyannote"] = "pyannote"
    auth_token: str = Field(
        default=os.getenv("HF_TOKEN", ""), description="HuggingFace auth token"
    )
    model_name: str = "pyannote/speaker-diarization-community-1"
    device: str = "cpu"


class NemotronDiarizationConfig(BaseDiarizationConfig):
    """Config for Nemotron 3 Diarization (transformers backend)"""

    model_type: Literal["nemotron"] = "nemotron"
    model_name: str = os.getenv("NEMOTRON_DIAR_MODEL", "nvidia/Nemotron-3-Diarization")
    device: str = os.getenv("NEMOTRON_DIAR_DEVICE", "cpu")  # "cpu" | "cuda"
    dtype: str = os.getenv("NEMOTRON_DIAR_DTYPE", "float32")  # bfloat16 for cuda
    threshold: float = float(
        os.getenv("NEMOTRON_DIAR_THRESHOLD", "0.5")
    )  # speaker activity probability counted as speech


class DiarizationSegment(BaseModel):
    """Single segment of diarization result"""

    start: float
    end: float
    speaker: str


class DiarizationResult(BaseModel):
    """Result of diarization"""

    segments: List[DiarizationSegment]
    info: dict = {}


DiarizationConfig = Union[PyannoteConfig, NemotronDiarizationConfig]


def default_diarization_config() -> Optional[DiarizationConfig]:
    """Build default diarization config based on DIARIZATION_BACKEND env variable"""
    backend = os.getenv("DIARIZATION_BACKEND", "none").lower()
    if backend == "none":
        return None
    elif backend == "pyannote":
        return PyannoteConfig()
    elif backend == "nemotron":
        return NemotronDiarizationConfig()
    else:
        raise ValueError(f"Unsupported DIARIZATION_BACKEND: {backend}")
