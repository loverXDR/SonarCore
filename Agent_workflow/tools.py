"""LangChain tools wrapping RAG_utils for agent usage"""

from langchain_core.tools import tool

from Core.RAG_utils import LlamaQueryEngine


def create_search_tool(engine: LlamaQueryEngine, session_id: str = None):
    """Create a search tool bound to a specific query engine

    Args:
        engine (LlamaQueryEngine): Configured query engine
        session_id (str, optional): Session ID for metadata filtering

    Returns:
        Callable: LangChain tool for document search
    """

    @tool
    def search_documents(query: str) -> str:
        """Search through the document to find relevant
        information. Use this tool when the user asks
        a specific question about the document content.

        Args:
            query: Search query in natural language
        """
        result = engine.query(query, doc_id=session_id)
        sources = "\n".join(
            f"[{s.get('score', 0):.2f}] {s['text'][:200]}"
            for s in result.source_nodes
        )
        return f"{result.response}\n\nSources:\n{sources}"

    return search_documents


def create_summarize_tool(engine: LlamaQueryEngine, session_id: str = None):
    """Create a summarization tool bound to a query engine

    Args:
        engine (LlamaQueryEngine): Configured query engine
            with SummaryIndex
        session_id (str, optional): Session ID for metadata filtering

    Returns:
        Callable: LangChain tool for document summarization
    """

    @tool
    def summarize_documents(query: str) -> str:
        """Summarize the document or a specific aspect of it.
        Use this tool when the user asks for a summary,
        overview, or key points from the document.

        Args:
            query: What aspect to summarize
        """
        result = engine.query(query, doc_id=session_id)
        return result.response

    return summarize_documents
