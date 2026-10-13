"""Agent Session Manager Service"""

import asyncio
import logging
import traceback
import uuid
from typing import List, Optional, Dict

import httpx
import qdrant_client
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict

from Core.Schemas import (
    AgentConfig,
    AgentResponse,
    AgentMessage,
    SummaryOptions,
    SummaryStyle,
)
from Agent_workflow.pipeline import SonarPipeline
from Agent_workflow.agent import SonarAgent
from Agent_workflow.docx_report import build_minutes_docx, build_summary_docx
from .repositories import BaseSessionRepository, BaseTaskRepository, SessionRecord
from .schemas import SessionInfo

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages SonarAgent sessions.

    Session state lives in the repository; live agents are cached in memory
    and rebuilt from the stored text on first access after a restart.
    """

    def __init__(
        self,
        repository: BaseSessionRepository,
        task_repository: BaseTaskRepository,
        default_config: AgentConfig,
    ):
        self.repository = repository
        self.task_repository = task_repository
        self.default_config = default_config
        self._agents: Dict[str, SonarAgent] = {}
        self._restore_lock = asyncio.Lock()

    async def init(self) -> None:
        """Prepare the underlying storages."""
        await self.repository.init()
        await self.task_repository.init()

    async def _get_record(self, session_id: str) -> SessionRecord:
        record = await self.repository.get(session_id)
        if not record:
            raise ValueError(f"Session '{session_id}' not found")
        return record

    async def _get_agent(self, record: SessionRecord) -> SonarAgent:
        """Return the cached agent or rebuild it from the stored session."""
        agent = self._agents.get(record.session_id)
        if agent:
            return agent
        async with self._restore_lock:
            agent = self._agents.get(record.session_id)
            if agent:
                return agent
            logger.info("Restoring agent for session %s", record.session_id)
            pipeline = SonarPipeline(config=record.config)
            agent = await run_in_threadpool(
                pipeline.process_text, record.text, record.session_id
            )
            agent._history = messages_from_dict(record.history)
            self._agents[record.session_id] = agent
            return agent

    async def _set_progress(self, task_id: Optional[str], progress: dict) -> None:
        if task_id:
            await self.task_repository.set(task_id, progress)

    async def create_session_from_audio(
        self,
        audio_path: str,
        use_diarization: bool = False,
        config: Optional[AgentConfig] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Process audio file and create a new agent session asynchronously.

        When *task_id* is supplied the method publishes progress updates into
        the task repository so that the SSE progress endpoint can stream them
        to the client.

        Args:
            audio_path (str): Path to the audio file on the server.
            use_diarization (bool, optional): Whether to use speaker diarization. Defaults to False.
            config (AgentConfig, optional): Optional override for agent config. Defaults to None.
            task_id (str, optional): If provided, progress is written to the task repository.

        Returns:
            str: Processed session ID.
        """
        pipeline_config = config or self.default_config
        pipeline = SonarPipeline(config=pipeline_config)
        session_id = str(uuid.uuid4())

        await self._set_progress(task_id, {"status": "transcribing", "pct": 10})

        try:
            text = await run_in_threadpool(
                pipeline.transcribe_audio_to_text, audio_path, use_diarization
            )

            await self._set_progress(task_id, {"status": "indexing", "pct": 60})

            agent = await run_in_threadpool(pipeline.process_text, text, session_id)
            await self.repository.add(session_id, pipeline_config, text)
            self._agents[session_id] = agent

            await self._set_progress(
                task_id, {"status": "ready", "pct": 100, "session_id": session_id}
            )

            return session_id

        except Exception:
            await self._set_progress(
                task_id, {"status": "error", "pct": 0, "detail": traceback.format_exc()}
            )
            raise

    async def create_session_from_text(
        self, 
        text: str, 
        config: Optional[AgentConfig] = None
    ) -> str:
        """Process text and create a new agent session asynchronously.

        Args:
            text (str): Raw text content to parse and index.
            config (AgentConfig, optional): Optional override for agent config. Defaults to None.

        Returns:
            str: Processed session ID.
        """
        pipeline_config = config or self.default_config
        pipeline = SonarPipeline(config=pipeline_config)
        session_id = str(uuid.uuid4())
        agent = await run_in_threadpool(pipeline.process_text, text, session_id)
        await self.repository.add(session_id, pipeline_config, text)
        self._agents[session_id] = agent
        return session_id

    async def chat(self, session_id: str, message: str) -> AgentResponse:
        """Send a message to an active chat session asynchronously.

        Args:
            session_id (str): Unique identifier of the session.
            message (str): User message.

        Raises:
            ValueError: If the session ID doesn't exist.

        Returns:
            AgentResponse: Complete response from the agent.
        """
        record = await self._get_record(session_id)
        agent = await self._get_agent(record)
        response = await run_in_threadpool(agent.chat, message)
        await self.repository.save_history(session_id, messages_to_dict(agent._history))
        return response

    async def summarize(self, session_id: str, options: SummaryOptions) -> str:
        """Generate a summary of the session document asynchronously.

        Args:
            session_id (str): Unique identifier of the session.
            options (SummaryOptions): Summary language, style and size.

        Raises:
            ValueError: If the session ID doesn't exist.

        Returns:
            str: Generated summary text.
        """
        record = await self._get_record(session_id)
        agent = await self._get_agent(record)
        response = await run_in_threadpool(agent.summarize, options)
        return response.answer

    async def export_summary_docx(self, session_id: str, options: SummaryOptions) -> bytes:
        """Generate a summary of the session document as a Word file.

        Meeting minutes are extracted as structured data and rendered with
        tables and the transcript appendix; other styles are converted from Markdown.

        Args:
            session_id (str): Unique identifier of the session.
            options (SummaryOptions): Summary language, style and size.

        Raises:
            ValueError: If the session ID doesn't exist.

        Returns:
            bytes: .docx file content.
        """
        record = await self._get_record(session_id)
        agent = await self._get_agent(record)
        if options.style == SummaryStyle.MEETING_MINUTES:
            minutes = await run_in_threadpool(agent.meeting_minutes, options, record.text)
            return await run_in_threadpool(
                build_minutes_docx,
                minutes,
                options.language,
                record.created_at,
                record.session_id,
                record.text,
            )
        response = await run_in_threadpool(agent.summarize, options)
        return await run_in_threadpool(
            build_summary_docx, response.answer, options, record.created_at
        )

    async def get_all_sessions(self) -> List[SessionInfo]:
        """List all managed sessions as DTOs.

        Returns:
            List[SessionInfo]: A list containing metadata for all active sessions.
        """
        records = await self.repository.list_all()
        return [
            SessionInfo(
                session_id=r.session_id,
                created_at=r.created_at,
                last_active=r.last_active,
                config=r.config,
                message_count=len(r.history),
            )
            for r in records
        ]

    async def get_session_details(self, session_id: str) -> SessionInfo:
        """Get detailed metadata for a single session.

        Args:
            session_id (str): Unique identifier of the session.

        Raises:
            ValueError: If the session ID doesn't exist.

        Returns:
            SessionInfo: Detailed metadata DTO for the session.
        """
        record = await self._get_record(session_id)
        return SessionInfo(
            session_id=record.session_id,
            created_at=record.created_at,
            last_active=record.last_active,
            config=record.config,
            message_count=len(record.history),
        )

    async def get_session_history(self, session_id: str) -> List[AgentMessage]:
        """Get the full message history for a session.

        Args:
            session_id (str): Unique identifier of the session.

        Raises:
            ValueError: If the session ID doesn't exist.

        Returns:
            List[AgentMessage]: The complete conversation history.
        """
        record = await self._get_record(session_id)

        # Map stored LangChain messages back to DTOs
        history = []
        for msg in messages_from_dict(record.history):
             role = "user" if isinstance(msg, HumanMessage) else "assistant"
             history.append(AgentMessage(role=role, content=msg.content))
        return history

    async def transcribe_audio(
        self, 
        audio_path: str, 
        use_diarization: bool = False, 
        config: Optional[AgentConfig] = None
    ) -> dict:
        """Transcribe audio to text without creating an agent session.

        Args:
            audio_path (str): Local path to audio file.
            use_diarization (bool, optional): Include speaker diarization. Defaults to False.
            config (AgentConfig, optional): Override pipeline configuration. Defaults to None.

        Returns:
            dict: Containing 'text' (str) and 'segments' (list[dict]).
        """
        pipeline_config = config or self.default_config
        pipeline = SonarPipeline(config=pipeline_config)
        text, segments = await run_in_threadpool(
            pipeline.process_audio_segments, audio_path, use_diarization
        )
        return {"text": text, "segments": segments}

    async def delete_session(self, session_id: str) -> bool:
        """Delete an active session from the repository.

        Args:
            session_id (str): Unique session identifier.

        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        self._agents.pop(session_id, None)
        return await self.repository.delete(session_id)

    async def get_task_progress(self, task_id: str) -> Optional[dict]:
        """Return the current progress dict for *task_id*, or ``None``."""
        return await self.task_repository.get(task_id)

    async def check_health(self) -> Dict[str, str]:
        """Check availability of system dependencies (Vector DB and LLM).

        Returns:
            Dict[str, str]: Dictionary containing status of each dependency.
        """
        health_status = {
            "vector_store": "unknown",
            "llm_provider": "unknown"
        }

        # 1. Check Qdrant
        try:
            client = qdrant_client.QdrantClient(url=self.default_config.index.vector_store.url)
            # Simple operation to check connection
            client.get_collections()
            health_status["vector_store"] = "reachable"
        except Exception as e:
            health_status["vector_store"] = f"unreachable: {str(e)}"

        # 2. Check LLM (OpenAI/Base)
        try:
            async with httpx.AsyncClient() as client:
                # Ping the API base or a simple endpoint
                response = await client.get(self.default_config.llm.api_base, timeout=5.0)
                # 200 (OK), 401 (Unauthorized), 404 (Not Found - root of API), 
                # 405 (Method Not Allowed) are all signs that the server is reachable
                if response.status_code in [200, 401, 404, 405]:
                    health_status["llm_provider"] = "active"
                else:
                    health_status["llm_provider"] = f"error: {response.status_code}"
        except Exception as e:
            health_status["llm_provider"] = f"unreachable: {str(e)}"

        return health_status
