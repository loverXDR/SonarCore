from abc import ABC, abstractmethod
from Core.Schemas import DiarizationResult


class BaseDiarization(ABC):
    """Base class for Diarization implementations"""

    @abstractmethod
    def diarize(self, audio_path: str) -> DiarizationResult:
        """Diarize the given audio file"""
        pass
