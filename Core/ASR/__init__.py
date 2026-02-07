from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Tuple

from Core.Schemas.asr_schema import ASRResult


class BaseASR(ABC):
    """Base class for ASR implementations"""

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> ASRResult:
        """Transcribe the given audio file"""
        pass
