"""Document parser based on llama-index"""

from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from Core.Schemas import DocumentParserConfig
from .base import BaseDocumentParser


class LlamaDocumentParser(BaseDocumentParser):
    """Document parser using llama-index SentenceSplitter"""

    def __init__(self, config: DocumentParserConfig) -> None:
        self.config = config
        self.splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def parse(self, file_path: str) -> List[Document]:
        """Parse file into list of llama-index Documents

        Args:
            file_path (str): Path to the text file

        Returns:
            List[Document]: Parsed documents with nodes
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.parse_text(text)

    def parse_text(self, text: str) -> List[Document]:
        """Parse raw text into list of llama-index Documents

        Args:
            text (str): Raw text content

        Returns:
            List[Document]: Parsed documents with nodes
        """
        document = Document(text=text)
        nodes = self.splitter.get_nodes_from_documents([document])
        return nodes


if __name__ == "__main__":
    config = DocumentParserConfig(chunk_size=256, chunk_overlap=32)
    parser = LlamaDocumentParser(config)

    nodes = parser.parse("path/to/document.txt")
    print(f"Parsed {len(nodes)} nodes")
    for node in nodes[:3]:
        print(f"  - {node.text[:80]}...")
