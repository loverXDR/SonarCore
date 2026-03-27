"""QA index builder based on llama-index with Qdrant vector store"""

from typing import List

import qdrant_client
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from Core.Schemas import IndexConfig
from .base import BaseIndexBuilder


class QAIndexBuilder(BaseIndexBuilder):
    """Index builder for QA tasks using VectorStoreIndex + Qdrant"""

    def __init__(self, config: IndexConfig) -> None:
        self.config = config
        self._embedding = self._build_embedding()
        self._vector_store = self._build_vector_store()
        self._configure_settings()

    def _build_embedding(self) -> HuggingFaceEmbedding:
        """Build embedding model"""
        return HuggingFaceEmbedding(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
        )

    def _build_vector_store(self) -> QdrantVectorStore:
        """Build Qdrant vector store"""
        client = qdrant_client.QdrantClient(
            url=self.config.vector_store.url,
        )
        return QdrantVectorStore(
            client=client,
            collection_name=self.config.vector_store.collection_name,
        )

    def _configure_settings(self) -> None:
        """Apply embedding model to llama-index Settings"""
        Settings.embed_model = self._embedding

    def build(self, documents: List[Document]) -> VectorStoreIndex:
        """Build VectorStoreIndex with Qdrant from documents

        Args:
            documents (List[Document]): List of llama-index Documents

        Returns:
            VectorStoreIndex: Built vector index for QA queries
        """
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.config.parser.chunk_size,
                    chunk_overlap=self.config.parser.chunk_overlap,
                ),
                self._embedding,
            ],
            vector_store=self._vector_store,
        )
        pipeline.run(documents=documents)

        index = VectorStoreIndex.from_vector_store(self._vector_store)
        return index

    def load_index(self) -> VectorStoreIndex:
        """Load existing index from Qdrant vector store

        Returns:
            VectorStoreIndex: Loaded index from existing collection
        """
        return VectorStoreIndex.from_vector_store(self._vector_store)


if __name__ == "__main__":
    from Core.Schemas import EmbeddingConfig, VectorStoreConfig

    config = IndexConfig(
        embedding=EmbeddingConfig(device="cpu"),
        vector_store=VectorStoreConfig(
            url="http://localhost:6333",
            collection_name="test_collection",
        ),
    )
    builder = QAIndexBuilder(config)

    doc = Document(text="Example document text for QA testing.")
    index = builder.build([doc])
    print(f"QA index built: {type(index)}")
