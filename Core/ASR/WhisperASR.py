"""Whisper ASR class based on faster-whisper"""

import dataclasses
from typing import Iterable, Tuple
from faster_whisper import WhisperModel
from Core.Schemas import ASRResult, ASRSegment, ASRWord, WhisperConfig


from .base import BaseASR


class WhisperASR(BaseASR):
    """Whisper ASR class based on faster-whisper"""

    def __init__(self, config: WhisperConfig) -> None:
        self.config = config
        self.model = WhisperModel(
            model_size_or_path=self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads,
            num_workers=self.config.num_workers,
            download_root=self.config.download_root,
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
            language=self.config.lang,
            task=self.config.task,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            temperature=self.config.temperature,
            initial_prompt=self.config.prompt,
            hotwords=self.config.hotwords,
            max_new_tokens=self.config.max_new_tokens,
            condition_on_previous_text=self.config.condition_on_previous_text,
            no_speech_threshold=self.config.no_speech_threshold,
            vad_filter=self.config.vad_filter,
            vad_parameters=self.config.vad_parameters,
            word_timestamps=self.config.word_timestamps,
        )

        segments = []
        full_text = ""
        for segment in transcription_list:
            words = [
                ASRWord(start=word.start, end=word.end, text=word.word)
                for word in segment.words or []
            ]
            segments.append(
                ASRSegment(
                    start=segment.start, end=segment.end, text=segment.text, words=words
                )
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
