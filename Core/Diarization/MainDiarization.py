import threading

from .PyannoteDiarization import PyannoteDiarization
from Core.Schemas import (
    DiarizationConfig,
    DiarizationResult,
    NemotronDiarizationConfig,
    PyannoteConfig,
)


# Loaded backends keyed by config, so each model is loaded once per process
_backend_cache: dict = {}
_backend_lock = threading.Lock()


class MainDiarization:
    """Main Diarization class that wraps specific implementations"""

    def __init__(self, config: DiarizationConfig) -> None:
        self.config = config
        self.backend = self._get_backend()

    def _get_backend(self):
        key = (type(self.config).__name__, self.config.model_dump_json())
        with _backend_lock:
            if key not in _backend_cache:
                _backend_cache[key] = self._build_backend()
            return _backend_cache[key]

    def _build_backend(self):
        if isinstance(self.config, PyannoteConfig):
            return PyannoteDiarization(self.config)
        elif isinstance(self.config, NemotronDiarizationConfig):
            from .NemotronDiarization import NemotronDiarization  # optional dependency: pip install sonarcore[nemotron]

            return NemotronDiarization(self.config)
        else:
            raise ValueError(f"Unsupported config type: {type(self.config)}")

    def diarize(self, audio_path: str) -> DiarizationResult:
        """Diarize audio file using the configured backend"""
        return self.backend.diarize(audio_path)
