"""Schemas for RAG utilities"""

import os
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentParserConfig(BaseModel):
    """Config for document parser"""

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "256"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "32"))


class EmbeddingConfig(BaseModel):
    """Config for embedding model"""

    model_name: str = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-small")
    device: str = os.getenv("EMBED_DEVICE", "cpu")


class LLMConfig(BaseModel):
    """Config for LLM"""

    api_base: str
    api_key: str
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")


class VectorStoreConfig(BaseModel):
    """Config for Qdrant vector store"""

    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "SonarPlace")


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
