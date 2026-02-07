from abc import ABC, abstractmethod
from Core.Schemas import ASRResult


class BaseASR(ABC):
    """Base class for ASR implementations"""

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> ASRResult:
        """Transcribe the given audio file"""
        pass
