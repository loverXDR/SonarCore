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
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    lang: str = None


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
