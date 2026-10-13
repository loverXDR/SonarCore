"""QA index builder based on llama-index with Qdrant vector store"""

from typing import List

import qdrant_client
from qdrant_client.models import FieldCondition, Filter, MatchValue
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
        self._client = qdrant_client.QdrantClient(url=config.vector_store.url)
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
        return QdrantVectorStore(
            client=self._client,
            collection_name=self.config.vector_store.collection_name,
        )

    def _configure_settings(self) -> None:
        """Apply embedding model to llama-index Settings"""
        Settings.embed_model = self._embedding

    def _is_indexed(self, text_hash: str) -> bool:
        """Return True if Qdrant already contains points for this text hash.

        Args:
            text_hash (str): SHA-256-derived content identifier.

        Returns:
            bool: True if at least one matching point exists.
        """
        try:
            results, _ = self._client.scroll(
                collection_name=self.config.vector_store.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="text_hash", match=MatchValue(value=text_hash))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(results) > 0
        except Exception:
            return False

    def build(self, documents: List[Document], text_hash: str) -> VectorStoreIndex:
        """Build or reuse VectorStoreIndex for the given content.

        Checks Qdrant for existing points tagged with *text_hash* before
        running the ingestion pipeline. If the content is already indexed,
        the pipeline is skipped entirely — no embeddings are recomputed.

        Args:
            documents (List[Document]): Source documents to index.
            text_hash (str): Stable content identifier (SHA-256 hex prefix).

        Returns:
            VectorStoreIndex: Index backed by the Qdrant collection.

        Raises:
            RuntimeError: If ingestion fails.
        """
        if self._is_indexed(text_hash):
            return VectorStoreIndex.from_vector_store(self._vector_store)

        for doc in documents:
            doc.metadata["text_hash"] = text_hash

        try:
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
        except Exception as e:
            raise RuntimeError(f"Failed to build QA index in Qdrant: {str(e)}")

        return VectorStoreIndex.from_vector_store(self._vector_store)

    def load_index(self) -> VectorStoreIndex:
        """Load existing index from Qdrant vector store

        Returns:
            VectorStoreIndex: Loaded index from existing collection
        """
        return VectorStoreIndex.from_vector_store(self._vector_store)


if __name__ == "__main__":
    import hashlib
    from Core.Schemas import EmbeddingConfig, VectorStoreConfig

    config = IndexConfig(
        embedding=EmbeddingConfig(device="cpu"),
        vector_store=VectorStoreConfig(
            url="http://localhost:6333",
            collection_name="test_collection",
        ),
    )
    builder = QAIndexBuilder(config)

    text = "Example document text for QA testing."
    doc = Document(text=text)
    h = hashlib.sha256(text.encode()).hexdigest()[:16]
    index = builder.build([doc], text_hash=h)
    print(f"QA index built: {type(index)}")
