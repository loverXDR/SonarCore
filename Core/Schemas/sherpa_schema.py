"""config module for sherpa-asr"""

from pydantic import BaseModel


class BaseSherpaConfig(BaseModel):
    model_type: str
    tokens_path: str
    num_threads: int = 4
    sample_rate: int = 16000
    feature_dim: int = 80
    decoding_method: str = "greedy_search"  # or "modified_beam_search"
    rule_fsts: str = ""  # comma-separated ITN rule fst paths
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
    max_active_paths: int = 4  # used with modified_beam_search
    hotwords_file: str = ""
    hotwords_score: float = 1.5
    blank_penalty: float = 0.0
    modeling_unit: str = "cjkchar"  # for hotwords: cjkchar | bpe | cjkchar+bpe
    bpe_vocab: str = ""
    lm: str = ""  # path to rnn lm onnx model
    lm_scale: float = 0.1
