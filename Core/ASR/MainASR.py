import threading

from .SherpaASR import SherpaOfflineASR
from .WhisperASR import WhisperASR
from Core.Schemas import (
    ASRConfig,
    ASRResult,
    BaseSherpaConfig,
    QwenASRConfig,
    WhisperConfig,
)


# Loaded backends keyed by config, so each model is loaded once per process
_backend_cache: dict = {}
_backend_lock = threading.Lock()


class MainASR:
    """Main ASR class that wraps specific implementations"""

    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self.backend = self._get_backend()

    def _get_backend(self):
        key = (type(self.config).__name__, self.config.model_dump_json())
        with _backend_lock:
            if key not in _backend_cache:
                _backend_cache[key] = self._build_backend()
            return _backend_cache[key]

    def _build_backend(self):
        if isinstance(self.config, WhisperConfig):
            return WhisperASR(self.config)
        elif isinstance(self.config, BaseSherpaConfig):
            return SherpaOfflineASR(self.config)
        elif isinstance(self.config, QwenASRConfig):
            from .QwenASR import QwenASR  # optional dependency: pip install sonarcore[qwen]

            return QwenASR(self.config)
        else:
            raise ValueError(f"Unsupported config type: {type(self.config)}")

    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe audio file using the configured backend"""
        return self.backend.transcribe(audio_path)
