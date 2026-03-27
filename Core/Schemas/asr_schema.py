"""Schemas for ASR utilities"""

import os
from typing import List, Union

from pydantic import BaseModel

from .sherpa_schema import (
    BaseSherpaConfig,
    SherpaParaformerConfig,
    SherpaTransducerConfig,
)


class WhisperConfig(BaseModel):
    """config class for faster-whisper"""

    model_type: str = "whisper"
    model_size: str = os.getenv("WHISPER_MODEL_SIZE", "small")
    device: str = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    beam_size: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    lang: str = os.getenv("WHISPER_LANG", None)


class ASRSegment(BaseModel):
    """single segment of asr result"""

    start: float
    end: float
    text: str


class ASRResult(BaseModel):
    """result of asr transcription"""

    text: str
    segments: List[ASRSegment]
    info: dict = {}


ASRConfig = Union[
    SherpaParaformerConfig, SherpaTransducerConfig, WhisperConfig, BaseSherpaConfig
]
