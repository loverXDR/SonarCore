"""Pipeline orchestrator: audio -> ASR -> index -> agent"""

from typing import Optional
from llama_index.core import Document

from Core.Schemas import (
    AgentConfig,
    DocumentParserConfig,
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

    def process_audio_segments(
        self,
        audio_path: str,
        use_diarization: bool = False,
    ) -> tuple[str, list[dict]]:
        """Transcribe and return (text, segments) with timestamps.

        Args:
            audio_path (str): Path to audio file.
            use_diarization (bool, optional): Whether to apply speaker diarization. Defaults to False.

        Returns:
            tuple[str, list[dict]]: A tuple containing the full raw text and a list of segment dictionaries.
        """
        asr = MainASR(self.config.asr)
        asr_result = asr.transcribe(audio_path)
        
        segments_out = []
        if use_diarization and self.config.diarization:
            diarizer = PyannoteDiarization(
                self.config.diarization,
            )
            diar_result = diarizer.diarize(audio_path)

            for seg in diar_result.segments:
                seg_text = self._find_text_for_segment(
                    seg, asr_result.segments,
                )
                segments_out.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg_text,
                    "speaker": seg.speaker
                })
            text = "\n".join(f"[{s['speaker']}]: {s['text']}" for s in segments_out)
        else:
            for seg in asr_result.segments:
                segments_out.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": None
                })
            text = asr_result.text
            
        return text, segments_out

    def transcribe_audio_to_text(
        self,
        audio_path: str,
        use_diarization: bool = False,
    ) -> str:
        """Transcribe and optionally diarize audio to get raw text.

        Args:
            audio_path (str): Path to audio file.
            use_diarization (bool, optional): Whether to apply speaker diarization. Defaults to False.

        Returns:
            str: The final transcribed text block.
        """
        text, _ = self.process_audio_segments(audio_path, use_diarization)
        return text

    def process_audio(
        self,
        audio_path: str,
        use_diarization: bool = False,
        session_id: Optional[str] = None,
    ) -> SonarAgent:
        """Process audio file through full pipeline

        Args:
            audio_path (str): Path to audio file
            use_diarization (bool): Whether to apply speaker diarization
            session_id (str, optional): Session ID for metadata tagging

        Returns:
            SonarAgent: Ready-to-chat agent
        """
        text = self.transcribe_audio_to_text(audio_path, use_diarization)
        return self.process_text(text, session_id)

    def process_text(self, text: str, session_id: Optional[str] = None) -> SonarAgent:
        """Process raw text through RAG pipeline

        Args:
            text (str): Input text (transcript or document)
            session_id (str, optional): Session ID for metadata tagging

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
        
        # Tag nodes with session_id for isolation
        if session_id:
            for node in nodes:
                node.metadata["session_id"] = session_id

        qa_builder = QAIndexBuilder(self.config.index)
        documents = [Document(text=text, metadata={"session_id": session_id} if session_id else {})]
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
            create_search_tool(search_engine, session_id=session_id),
            create_summarize_tool(summary_engine, session_id=session_id),
        ]

        return SonarAgent(
            config=self.config,
            search_tool=tools[0],
            summarize_tool=tools[1],
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
