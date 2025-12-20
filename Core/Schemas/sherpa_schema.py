"""config module for sherpa-asr"""

from pydantic import BaseModel


class BaseSherpaConfig(BaseModel):
    model_type: str
    tokens_path: str
    num_threads: int = 4
    sample_rate: int = 16000
    debug_mode: bool = False
    provider: str = "cpu"


class SherpaParaformerConfig(BaseSherpaConfig):
    """config class for sherpa-asr paraformer"""

    model_type: str = "paraformer"
    model: str  # path to model.onnx file


class SherpaTransducerConfig(BaseSherpaConfig):
    """config class for sherpa-asr transformer"""

    model_type: str = "transducer"
    encoder: str
    decoder: str
    joiner: str
