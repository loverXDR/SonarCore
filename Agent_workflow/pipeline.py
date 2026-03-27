"""Pipeline orchestrator: audio -> ASR -> index -> agent"""

from llama_index.core import Document

from Core.Schemas import (
    AgentConfig,
    DocumentParserConfig,
    LLMConfig,
)
from Core.ASR import MainASR
from Core.Diarization import PyannoteDiarization
from Core.RAG_utils import (
    LlamaDocumentParser,
    QAIndexBuilder,
    SummaryIndexBuilder,
    LlamaQueryEngine,
)

from .tools import create_search_tool, create_summarize_tool
from .agent import SonarAgent


class SonarPipeline:
    """Full pipeline: audio/text -> RAG index -> chat agent

    Orchestrates ASR, optional diarization, document parsing,
    index building, and agent creation.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def process_audio(
        self,
        audio_path: str,
        use_diarization: bool = False,
    ) -> SonarAgent:
        """Process audio file through full pipeline

        Args:
            audio_path (str): Path to audio file
            use_diarization (bool): Whether to apply
                speaker diarization

        Returns:
            SonarAgent: Ready-to-chat agent
        """
        asr = MainASR(self.config.asr)
        asr_result = asr.transcribe(audio_path)
        text = asr_result.text

        if use_diarization and self.config.diarization:
            diarizer = PyannoteDiarization(
                self.config.diarization,
            )
            diar_result = diarizer.diarize(audio_path)

            segments_text = []
            for seg in diar_result.segments:
                seg_text = self._find_text_for_segment(
                    seg, asr_result.segments,
                )
                segments_text.append(
                    f"[{seg.speaker}]: {seg_text}"
                )
            text = "\n".join(segments_text)

        return self.process_text(text)

    def process_text(self, text: str) -> SonarAgent:
        """Process raw text through RAG pipeline

        Args:
            text (str): Input text (transcript or document)

        Returns:
            SonarAgent: Ready-to-chat agent
        """
        parser = LlamaDocumentParser(
            DocumentParserConfig(
                chunk_size=self.config.index.parser.chunk_size,
                chunk_overlap=(
                    self.config.index.parser.chunk_overlap
                ),
            )
        )
        nodes = parser.parse_text(text)

        qa_builder = QAIndexBuilder(self.config.index)
        documents = [Document(text=text)]
        qa_index = qa_builder.build(documents)

        summary_builder = SummaryIndexBuilder(
            self.config.index,
        )
        summary_index = summary_builder.build(nodes)

        search_engine = LlamaQueryEngine(
            qa_index, self.config.llm,
        )
        summary_engine = LlamaQueryEngine(
            summary_index, self.config.llm,
        )

        tools = [
            create_search_tool(search_engine),
            create_summarize_tool(summary_engine),
        ]

        return SonarAgent(
            config=self.config,
            tools=tools,
        )

    @staticmethod
    def _find_text_for_segment(diar_seg, asr_segments):
        """Match diarization segment to ASR text

        Args:
            diar_seg: Diarization segment with timing
            asr_segments: List of ASR segments

        Returns:
            str: Combined text for the time range
        """
        texts = []
        for asr_seg in asr_segments:
            overlap_start = max(
                diar_seg.start, asr_seg.start,
            )
            overlap_end = min(diar_seg.end, asr_seg.end)
            if overlap_end > overlap_start:
                texts.append(asr_seg.text)
        return " ".join(texts) if texts else ""
