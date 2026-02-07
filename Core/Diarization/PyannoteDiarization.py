import torch
from pyannote.audio import Pipeline

from Core.Schemas import (
    DiarizationResult,
    DiarizationSegment,
    PyannoteConfig,
)

from .base import BaseDiarization


class PyannoteDiarization(BaseDiarization):
    """Pyannote Diarization implementation"""

    def __init__(self, config: PyannoteConfig) -> None:
        self.config = config
        self.pipeline = Pipeline.from_pretrained(
            self.config.model_name,
            token=self.config.auth_token,
        )
        if self.config.device != "cpu":
            self.pipeline.to(torch.device(self.config.device))

    def diarize(self, audio_path: str) -> DiarizationResult:
        """
        Diarize the given audio file

        Args:
            audio_path (str): Path to the audio file

        Returns:
            DiarizationResult: Diarization result
        """
        diarization = self.pipeline(audio_path)
        segments = []

        for turn, speaker in diarization.speaker_diarization:
            segments.append(
                DiarizationSegment(start=turn.start, end=turn.end, speaker=speaker)
            )

        return DiarizationResult(segments=segments)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Pyannote Diarization Client")
    parser.add_argument(
        "audio_path", type=str, help="Path to the audio file", default="input.wav"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace Auth Token",
    )

    args = parser.parse_args()

    if not args.token:
        print(
            "Error: HuggingFace token is required. Pass it via --token or set HF_TOKEN env var."
        )
        exit(1)

    config = PyannoteConfig(auth_token=args.token)

    try:
        model = PyannoteDiarization(config)
        print(f"Processing: {args.audio_path}")
        result = model.diarize(args.audio_path)

        for segment in result.segments:
            print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.speaker}")

    except Exception as e:
        print(f"Error: {e}")
