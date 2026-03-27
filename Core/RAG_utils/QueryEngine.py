"""Query engine based on llama-index"""

from typing import Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.llms.openai import OpenAI

from Core.Schemas import LLMConfig, QueryResult
from .base import BaseQueryEngine


class LlamaQueryEngine(BaseQueryEngine):
    """Query engine using llama-index for QA over indexed documents"""

    def __init__(self, index: VectorStoreIndex, llm_config: LLMConfig) -> None:
        self.index = index
        self.llm_config = llm_config
        self._configure_llm()

    def _configure_llm(self) -> None:
        """Configure LLM in llama-index Settings"""
        Settings.llm = OpenAI(
            model=self.llm_config.model,
            api_base=self.llm_config.api_base,
            api_key=self.llm_config.api_key,
        )

    def query(
        self,
        query_text: str,
        doc_id: Optional[str] = None,
        **kwargs,
    ) -> QueryResult:
        """Execute a query against the index

        Args:
            query_text (str): Question to ask
            doc_id (Optional[str]): Filter results by document ID

        Returns:
            QueryResult: Response with source nodes
        """
        engine_kwargs = {}

        if doc_id:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="doc_id", value=doc_id)]
            )
            engine_kwargs["filters"] = filters

        query_engine = self.index.as_query_engine(**engine_kwargs)
        response = query_engine.query(query_text)

        source_nodes = [
            {
                "text": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata,
            }
            for node in response.source_nodes
        ]

        return QueryResult(
            response=str(response),
            source_nodes=source_nodes,
        )


if __name__ == "__main__":
    from Core.Schemas import (
        IndexConfig,
        EmbeddingConfig,
        VectorStoreConfig,
    )
    from .QAIndexBuilder import QAIndexBuilder

    index_config = IndexConfig(
        embedding=EmbeddingConfig(device="cpu"),
        vector_store=VectorStoreConfig(
            url="http://localhost:6333",
            collection_name="SonarPlace",
        ),
    )
    builder = QAIndexBuilder(index_config)
    index = builder.load_index()

    llm_config = LLMConfig(
        api_base="https://gptproxy.recdev.ru:444/v1/",
        api_key="your-api-key",
    )
    engine = LlamaQueryEngine(index, llm_config)

    result = engine.query("Что обсуждалось на совещании?")
    print(result.response)
    for node in result.source_nodes:
        print(f"  [{node['score']:.3f}] {node['text'][:80]}...")
