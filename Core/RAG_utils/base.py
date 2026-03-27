"""Base interfaces for RAG utilities"""

from abc import ABC, abstractmethod
from typing import List

from llama_index.core import Document, VectorStoreIndex
from Core.Schemas import QueryResult


class BaseDocumentParser(ABC):
    """Base class for document parsing implementations"""

    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """Parse file into list of llama-index Documents"""
        pass

    @abstractmethod
    def parse_text(self, text: str) -> List[Document]:
        """Parse raw text into list of llama-index Documents"""
        pass


class BaseIndexBuilder(ABC):
    """Base class for index building implementations"""

    @abstractmethod
    def build(self, documents: List[Document]) -> VectorStoreIndex:
        """Build index from list of Documents"""
        pass


class BaseQueryEngine(ABC):
    """Base class for query engine implementations"""

    @abstractmethod
    def query(self, query_text: str, **kwargs) -> QueryResult:
        """Execute a query against the index"""
        pass
