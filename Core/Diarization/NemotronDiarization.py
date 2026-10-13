"""Nemotron Diarization class based on transformers (nvidia/Nemotron-3-Diarization)"""

import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

from Core.Schemas import (
    DiarizationResult,
    DiarizationSegment,
    NemotronDiarizationConfig,
)

from .base import BaseDiarization


class NemotronDiarization(BaseDiarization):
    """Nemotron 3 Diarization implementation (offline inference, up to 8 speakers)"""

    def __init__(self, config: NemotronDiarizationConfig) -> None:
        self.config = config
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(
            self.config.model_name,
            dtype=getattr(torch, self.config.dtype),
        )
        self.model.to(self.config.device).eval()

    def diarize(self, audio_path: str) -> DiarizationResult:
        """
        Diarize the given audio file

        Args:
            audio_path (str): Path to the audio file

        Returns:
            DiarizationResult: Diarization result
        """
        sampling_rate = self.processor.feature_extractor.sampling_rate
        audio = load_audio(audio_path, sampling_rate=sampling_rate)
        inputs = self.processor(audio, sampling_rate=sampling_rate).to(
            self.model.device, dtype=self.model.dtype
        )

        with torch.inference_mode():
            logits = self.model(**inputs).logits

        speaker_segments = self.processor.extract_speaker_dict(
            logits, inputs.attention_mask, threshold=self.config.threshold
        )[0]

        segments = [
            DiarizationSegment(
                start=segment["Start"],
                end=segment["End"],
                speaker=f"SPEAKER_{segment['Speaker']:02d}",
            )
            for segment in speaker_segments
        ]
        num_speakers = len({segment.speaker for segment in segments})

        return DiarizationResult(segments=segments, info={"num_speakers": num_speakers})


if __name__ == "__main__":
    config = NemotronDiarizationConfig()
    model = NemotronDiarization(config)

    result = model.diarize("Path/to/audio.wav")

    for segment in result.segments:
        print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.speaker}")
    print(result.info)
