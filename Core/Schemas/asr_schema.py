"""Schemas for ASR utilities"""

import os
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, field_validator

from .sherpa_schema import (
    BaseSherpaConfig,
    SherpaParaformerConfig,
    SherpaTransducerConfig,
)


class BaseASRConfig(BaseModel):
    """common config fields for model-based ASR backends (whisper, qwen)"""

    model_type: str
    device: str = "cpu"
    lang: Optional[str] = None  # ISO 639-1 code, None for auto-detection
    prompt: Optional[str] = None  # context / terminology hint for the model
    max_new_tokens: Optional[int] = None


class WhisperConfig(BaseASRConfig):
    """config class for faster-whisper"""

    model_type: Literal["whisper"] = "whisper"
    model_size: str = os.getenv("WHISPER_MODEL_SIZE", "small")
    device: str = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    cpu_threads: int = int(os.getenv("WHISPER_CPU_THREADS", "0"))
    num_workers: int = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
    download_root: Optional[str] = os.getenv("WHISPER_DOWNLOAD_ROOT", None)
    beam_size: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    best_of: int = 5
    temperature: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lang: Optional[str] = os.getenv("WHISPER_LANG", None)
    task: Literal["transcribe", "translate"] = os.getenv("WHISPER_TASK", "transcribe")
    prompt: Optional[str] = os.getenv("WHISPER_PROMPT", None)
    hotwords: Optional[str] = os.getenv("WHISPER_HOTWORDS", None)
    max_new_tokens: Optional[int] = (
        int(os.getenv("WHISPER_MAX_NEW_TOKENS"))
        if os.getenv("WHISPER_MAX_NEW_TOKENS")
        else None
    )
    condition_on_previous_text: bool = (
        os.getenv("WHISPER_CONDITION_ON_PREV", "true").lower() == "true"
    )
    no_speech_threshold: Optional[float] = 0.6
    vad_filter: bool = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"
    vad_parameters: Optional[dict] = None
    word_timestamps: bool = (
        os.getenv("WHISPER_WORD_TIMESTAMPS", "false").lower() == "true"
    )


class QwenASRConfig(BaseASRConfig):
    """config class for Qwen3-ASR (qwen-asr transformers backend)"""

    model_type: Literal["qwen"] = "qwen"
    model_name: str = os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
    device: str = os.getenv("QWEN_ASR_DEVICE", "cpu")  # "cpu" | "cuda:0"
    dtype: str = os.getenv("QWEN_ASR_DTYPE", "float32")  # bfloat16 for cuda
    attn_implementation: Optional[str] = os.getenv("QWEN_ASR_ATTN", None)
    max_new_tokens: Optional[int] = int(os.getenv("QWEN_ASR_MAX_NEW_TOKENS", "4096"))
    max_inference_batch_size: int = int(os.getenv("QWEN_ASR_BATCH_SIZE", "8"))
    lang: Optional[str] = os.getenv("QWEN_ASR_LANG", None)
    prompt: Optional[str] = os.getenv("QWEN_ASR_PROMPT", None)
    forced_aligner: Optional[str] = os.getenv(
        "QWEN_ASR_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B"
    )  # None disables timestamps (and segments)
    segment_gap: float = float(os.getenv("QWEN_ASR_SEGMENT_GAP", "0.8"))

    @field_validator("forced_aligner", "attn_implementation")
    @classmethod
    def _empty_to_none(cls, value: Optional[str]) -> Optional[str]:
        return value or None


class ASRWord(BaseModel):
    """single word of asr result with timestamps"""

    start: float
    end: float
    text: str  # keeps its leading space: concatenated words give the segment text


class ASRSegment(BaseModel):
    """single segment of asr result"""

    start: float
    end: float
    text: str
    words: List[ASRWord] = []  # filled when the backend returns word timestamps


class ASRResult(BaseModel):
    """result of asr transcription"""

    text: str
    segments: List[ASRSegment]
    info: dict = {}


ASRConfig = Union[
    SherpaParaformerConfig,
    SherpaTransducerConfig,
    WhisperConfig,
    QwenASRConfig,
    BaseSherpaConfig,
]


def default_asr_config() -> ASRConfig:
    """Build default ASR config based on ASR_BACKEND env variable"""
    backend = os.getenv("ASR_BACKEND", "whisper").lower()
    if backend == "whisper":
        return WhisperConfig()
    elif backend == "qwen":
        return QwenASRConfig()
    else:
        raise ValueError(f"Unsupported ASR_BACKEND: {backend}")
