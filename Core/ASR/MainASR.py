from .SherpaASR import SherpaOfflineASR
from .WhisperASR import WhisperASR
from Core.Schemas import ASRConfig, ASRResult, BaseSherpaConfig, WhisperConfig


class MainASR:
    """Main ASR class that wraps specific implementations"""

    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self.backend = self._build_backend()

    def _build_backend(self):
        if isinstance(self.config, WhisperConfig):
            return WhisperASR(self.config)
        elif isinstance(self.config, BaseSherpaConfig):
            return SherpaOfflineASR(self.config)
        else:
            raise ValueError(f"Unsupported config type: {type(self.config)}")

    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe audio file using the configured backend"""
        return self.backend.transcribe(audio_path)
