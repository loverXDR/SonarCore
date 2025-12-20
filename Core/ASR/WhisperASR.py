"""Whisper ASR class based on faster-whisper"""

from typing import Iterable, Tuple
from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionInfo, Segment


class WhisperASR:
    """Whisper ASR class based on faster-whisper"""

    def __init__(
        self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"
    ) -> None:
        self.model = WhisperModel(
            model_size_or_path=model_size, device=device, compute_type=compute_type
        )

    def transcribe(
        self, audio_path: str, beam_size: int = 5, lang: str = None
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribe the given audio file

        Args:
            audio_path (str): Path to the audio file
            beam_size (int, optional): Beam size. Defaults to 5.
            lang (str, optional): Language. Defaults to None.

        returns:
            Tuple[Iterable[Segment], TranscriptionInfo]: Transcription and transcription info
        """

        transcription_list, transcription_info = self.model.transcribe(
            audio_path, beam_size=beam_size, language=lang
        )
        return transcription_list, transcription_info


if __name__ == "__main__":
    model = WhisperASR()

    transcription, info = model.transcribe("Path/to/audio.wav")

    for segment in transcription:
        print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")
    print(info)
