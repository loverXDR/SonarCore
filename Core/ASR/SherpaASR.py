"""Sherpa offline ASR class based on sherpa-onnx"""

from typing import Union
from sherpa_onnx import OfflineRecognizer
from Core.Schemas.sherpa_schema import SherpaParaformerConfig, SherpaTransducerConfig
from Core.Schemas.asr_schema import ASRResult
from Core.Utils.audio_utils import wav_to_data


from . import BaseASR


class SherpaOfflineASR(BaseASR):
    """Sherpa ASR class based on sherpa-onnx"""


    def __init__(
        self, config: Union[SherpaParaformerConfig, SherpaTransducerConfig]
    ) -> None:
        self.config = config
        self.recognizer = self._build_recognizer()

    def _build_recognizer(self) -> OfflineRecognizer:
        """Build sherpa recognizer based on sherpa-onnx"""
        if self.config.model_type == "transducer":
            return OfflineRecognizer.from_transducer(
                decoder=self.config.decoder,
                encoder=self.config.encoder,
                joiner=self.config.joiner,
                tokens=self.config.tokens_path,
                num_threads=self.config.num_threads,
                debug=self.config.debug_mode,
                provider=self.config.provider,
            )
        elif self.config.model_type == "paraformer":
            return OfflineRecognizer.from_paraformer(
                paraformer=self.config.model,
                tokens=self.config.tokens_path,
                num_threads=self.config.num_threads,
                debug=self.config.debug_mode,
                provider=self.config.provider,
            )
        else:
            raise ValueError("Invalid model type")

    def transcribe(self, audio_path, sample_rate: int = 16000) -> ASRResult:
        """Transcribe the given audio data
        Args:
            audio_path (str): path to audio data
            sample_rate (int, optional): sample rate of the audio to be transcribed. Defaults to 16000.
        returns:
            ASRResult: transcribed audio data
        """
        sample_rate_src, audio_data = wav_to_data(audio_path)
        s = self.recognizer.create_stream()
        s.accept_waveform(sample_rate_src, audio_data)
        self.recognizer.decode_stream(s)
        text = s.result.text
        return ASRResult(text=text, segments=[])
