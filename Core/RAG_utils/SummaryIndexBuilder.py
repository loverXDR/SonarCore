"""Summary index builder based on llama-index"""

from typing import List

from llama_index.core import Document, SummaryIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from Core.Schemas import IndexConfig
from .base import BaseIndexBuilder


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

    def build(self, documents: List[Document]) -> SummaryIndex:
        """Build SummaryIndex from documents

        Args:
            documents (List[Document]): List of llama-index Documents or nodes

        Returns:
            SummaryIndex: Built summary index for summarization queries
        """
        index = SummaryIndex(documents)
        return index


if __name__ == "__main__":
    from Core.Schemas import EmbeddingConfig

    config = IndexConfig(embedding=EmbeddingConfig(device="cpu"))
    builder = SummaryIndexBuilder(config)

    doc = Document(text="Example document text for testing.")
    index = builder.build([doc])
    print(f"Summary index built: {type(index)}")
