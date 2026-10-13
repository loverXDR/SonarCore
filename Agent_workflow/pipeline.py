"""Pipeline orchestrator: audio -> ASR -> index -> agent"""

import hashlib
from typing import Optional

from llama_index.core import Document

from Core.Schemas import (
    AgentConfig,
    DocumentParserConfig,
)
from Core.ASR import MainASR
from Core.Diarization import MainDiarization
from Core.Utils import prepared_audio
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
            audio_path (str): Path to audio or video file.
            use_diarization (bool, optional): Whether to apply speaker diarization. Defaults to False.

        Returns:
            tuple[str, list[dict]]: A tuple containing the full raw text and a list of segment dictionaries.

        Raises:
            RuntimeError: If audio extraction, ASR transcription or diarization fails.
        """
        # Every backend gets the same 16 kHz mono wav, whatever the input container
        with prepared_audio(audio_path) as wav_path:
            return self._process_wav_segments(wav_path, use_diarization)

    def _process_wav_segments(
        self,
        audio_path: str,
        use_diarization: bool,
    ) -> tuple[str, list[dict]]:
        """Run ASR and optional diarization on a prepared wav file.

        Args:
            audio_path (str): Path to 16 kHz mono wav file.
            use_diarization (bool): Whether to apply speaker diarization.

        Returns:
            tuple[str, list[dict]]: A tuple containing the full raw text and a list of segment dictionaries.
        """
        try:
            asr = MainASR(self.config.asr)
            asr_result = asr.transcribe(audio_path)
        except Exception as e:
            raise RuntimeError(f"ASR transcription failed: {str(e)}")

        segments_out = []
        if use_diarization and self.config.diarization:
            try:
                diarizer = MainDiarization(
                    self.config.diarization,
                )
                diar_result = diarizer.diarize(audio_path)
            except Exception as e:
                raise RuntimeError(f"Diarization failed: {str(e)}")

            # Assign speakers word by word when the ASR backend returns word
            # timestamps, otherwise fall back to whole segments
            # (words keep their own leading spaces, segments are joined with one)
            units = [
                (unit, piece)
                for seg in asr_result.segments
                for unit, piece in (
                    [(word, word.text) for word in seg.words]
                    or [(seg, " " + seg.text.strip())]
                )
            ]
            for unit, piece in units:
                speaker = self._find_speaker_for_segment(
                    unit, diar_result.segments,
                )
                # Merge consecutive units of the same speaker into one turn
                if segments_out and segments_out[-1]["speaker"] == speaker:
                    segments_out[-1]["end"] = unit.end
                    segments_out[-1]["text"] += piece
                    continue
                segments_out.append({
                    "start": unit.start,
                    "end": unit.end,
                    "text": piece.strip(),
                    "speaker": speaker
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
        # Stable content identifier: same text always yields the same hash,
        # so Qdrant embeddings and the in-memory summary cache can be reused.
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        parser = LlamaDocumentParser(
            DocumentParserConfig(
                chunk_size=self.config.index.parser.chunk_size,
                chunk_overlap=self.config.index.parser.chunk_overlap,
            )
        )
        nodes = parser.parse_text(text)

        for node in nodes:
            node.metadata["text_hash"] = text_hash

        qa_builder = QAIndexBuilder(self.config.index)
        documents = [Document(text=text, metadata={"text_hash": text_hash})]
        qa_index = qa_builder.build(documents, text_hash=text_hash)

        summary_builder = SummaryIndexBuilder(self.config.index)
        summary_index = summary_builder.build(nodes, text_hash=text_hash)

        search_engine = LlamaQueryEngine(qa_index, self.config.llm)
        summary_engine = LlamaQueryEngine(summary_index, self.config.llm)

        tools = [
            create_search_tool(search_engine, text_hash=text_hash),
            create_summarize_tool(summary_engine, text_hash=text_hash),
        ]

        return SonarAgent(
            config=self.config,
            search_tool=tools[0],
            summarize_tool=tools[1],
        )

    @staticmethod
    def _find_speaker_for_segment(asr_seg, diar_segments):
        """Match ASR segment (or word) to the speaker who talks most during it

        Args:
            asr_seg: ASR segment or word with timing
            diar_segments: List of diarization segments

        Returns:
            Optional[str]: Speaker with the largest time overlap, or the speaker
                of the nearest diarization segment if none overlaps
        """
        if not diar_segments:
            return None

        overlaps = {}
        for diar_seg in diar_segments:
            overlap_start = max(
                diar_seg.start, asr_seg.start,
            )
            overlap_end = min(diar_seg.end, asr_seg.end)
            if overlap_end > overlap_start:
                overlaps[diar_seg.speaker] = (
                    overlaps.get(diar_seg.speaker, 0.0) + overlap_end - overlap_start
                )
        if overlaps:
            return max(overlaps, key=overlaps.get)

        center = (asr_seg.start + asr_seg.end) / 2
        nearest = min(
            diar_segments,
            key=lambda diar_seg: min(
                abs(center - diar_seg.start), abs(center - diar_seg.end),
            ),
        )
        return nearest.speaker
