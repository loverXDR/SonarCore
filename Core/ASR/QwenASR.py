"""Qwen ASR class based on qwen-asr (Qwen3-ASR)"""

from typing import Iterable, List, Optional

import torch
from qwen_asr import Qwen3ASRModel
from Core.Schemas import ASRResult, ASRSegment, ASRWord, QwenASRConfig


from .base import BaseASR

# ISO 639-1 codes -> language names expected by Qwen3-ASR
QWEN_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}


class QwenASR(BaseASR):
    """Qwen ASR class based on qwen-asr transformers backend"""

    def __init__(self, config: QwenASRConfig) -> None:
        self.config = config
        dtype = getattr(torch, self.config.dtype)

        model_kwargs = dict(
            dtype=dtype,
            device_map=self.config.device,
            max_inference_batch_size=self.config.max_inference_batch_size,
            max_new_tokens=self.config.max_new_tokens,
        )
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        if self.config.forced_aligner:
            model_kwargs["forced_aligner"] = self.config.forced_aligner
            model_kwargs["forced_aligner_kwargs"] = dict(
                dtype=dtype,
                device_map=self.config.device,
            )

        self.model = Qwen3ASRModel.from_pretrained(
            self.config.model_name, **model_kwargs
        )

    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe the given audio file

        Args:
            audio_path (str): Path to the audio file

        returns:
            ASRResult: Transcription result
        """

        result = self.model.transcribe(
            audio=audio_path,
            context=self.config.prompt or "",
            language=self._to_qwen_language(self.config.lang),
            return_time_stamps=bool(self.config.forced_aligner),
        )[0]

        segments = (
            self._build_segments(result.time_stamps)
            if result.time_stamps is not None
            else []
        )

        return ASRResult(
            text=result.text,
            segments=segments,
            info={"language": result.language},
        )

    @staticmethod
    def _to_qwen_language(lang: Optional[str]) -> Optional[str]:
        """Map ISO 639-1 code to Qwen language name, pass unknown values as is"""
        if not lang:
            return None
        return QWEN_LANGUAGES.get(lang.lower(), lang)

    def _build_segments(self, time_stamps: Iterable) -> List[ASRSegment]:
        """Group word-level forced aligner items into segments split by pauses

        Args:
            time_stamps (Iterable): ForcedAlignResult items with text, start_time, end_time

        returns:
            List[ASRSegment]: Segments with timestamps
        """
        segments = []
        words, start, end = [], None, None
        for item in time_stamps:
            if words and item.start_time - end > self.config.segment_gap:
                segments.append(self._make_segment(start, end, words))
                words, start = [], None
            if start is None:
                start = item.start_time
            words.append(
                ASRWord(start=item.start_time, end=item.end_time, text=" " + item.text)
            )
            end = item.end_time

        if words:
            segments.append(self._make_segment(start, end, words))
        return segments

    @staticmethod
    def _make_segment(start: float, end: float, words: List[ASRWord]) -> ASRSegment:
        """Build a segment from its words"""
        return ASRSegment(
            start=start,
            end=end,
            text="".join(word.text for word in words).strip(),
            words=words,
        )


if __name__ == "__main__":
    config = QwenASRConfig()
    model = QwenASR(config)

    result = model.transcribe("Path/to/audio.wav")

    for segment in result.segments:
        print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")
    print(result.info)
