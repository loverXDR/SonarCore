"""Schemas for RAG utilities"""

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentParserConfig(BaseModel):
    """Config for document parser"""

    chunk_size: int = 256
    chunk_overlap: int = 32


class EmbeddingConfig(BaseModel):
    """Config for embedding model"""

    model_name: str = "intfloat/multilingual-e5-small"
    device: str = "cpu"


class LLMConfig(BaseModel):
    """Config for LLM"""

    api_base: str
    api_key: str
    model: str = "gpt-4o-mini"


class VectorStoreConfig(BaseModel):
    """Config for Qdrant vector store"""

    url: str = "http://localhost:6333"
    collection_name: str = "SonarPlace"


class IndexConfig(BaseModel):
    """Combined config for index building"""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    parser: DocumentParserConfig = Field(default_factory=DocumentParserConfig)
    llm: Optional[LLMConfig] = None


class QueryResult(BaseModel):
    """Result of a RAG query"""

    response: str
    source_nodes: List[dict] = []
