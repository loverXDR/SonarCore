"""Whisper ASR class based on faster-whisper"""

import dataclasses
from typing import Iterable, Tuple
from faster_whisper import WhisperModel
from Core.Schemas import ASRResult, ASRSegment, WhisperConfig


from .base import BaseASR


class WhisperASR(BaseASR):
    """Whisper ASR class based on faster-whisper"""

    def __init__(self, config: WhisperConfig) -> None:
        self.config = config
        self.model = WhisperModel(
            model_size_or_path=self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe the given audio file

        Args:
            audio_path (str): Path to the audio file

        returns:
            ASRResult: Transcription result
        """

        transcription_list, transcription_info = self.model.transcribe(
            audio_path,
            beam_size=self.config.beam_size,
            language=self.config.lang,
        )

        segments = []
        full_text = ""
        for segment in transcription_list:
            segments.append(
                ASRSegment(start=segment.start, end=segment.end, text=segment.text)
            )
            full_text += segment.text

        return ASRResult(
            text=full_text,
            segments=segments,
            info=dataclasses.asdict(transcription_info),
        )


if __name__ == "__main__":
    config = WhisperConfig()
    model = WhisperASR(config)

    result = model.transcribe("Path/to/audio.wav")

    for segment in result.segments:
        print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")
    print(result.info)
