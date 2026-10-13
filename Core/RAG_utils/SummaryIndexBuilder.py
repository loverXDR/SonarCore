"""Summary index builder based on llama-index"""

from typing import Dict, List

from llama_index.core import Document, SummaryIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from Core.Schemas import IndexConfig
from .base import BaseIndexBuilder

# Process-level cache: avoids rebuilding in-memory summary indices for
# content that was already processed during this server lifetime.
_summary_cache: Dict[str, SummaryIndex] = {}


class SummaryIndexBuilder(BaseIndexBuilder):
    """Index builder for summarization tasks using llama-index SummaryIndex"""

    def __init__(self, config: IndexConfig) -> None:
        self.config = config
        self._configure_settings()

    def _configure_settings(self) -> None:
        """Apply embedding model to llama-index Settings"""
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
        )

    def build(self, documents: List[Document], text_hash: str) -> SummaryIndex:
        """Build or return cached SummaryIndex for the given content.

        Args:
            documents (List[Document]): Documents or nodes to index.
            text_hash (str): Stable content identifier used as cache key.

        Returns:
            SummaryIndex: In-memory summary index.

        Raises:
            RuntimeError: If index building fails.
        """
        if text_hash in _summary_cache:
            return _summary_cache[text_hash]

        try:
            index = SummaryIndex(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to build summary index: {str(e)}")

        _summary_cache[text_hash] = index
        return index


if __name__ == "__main__":
    from Core.Schemas import EmbeddingConfig

    config = IndexConfig(embedding=EmbeddingConfig(device="cpu"))
    builder = SummaryIndexBuilder(config)

    doc = Document(text="Example document text for testing.")
    index = builder.build([doc])
    print(f"Summary index built: {type(index)}")
