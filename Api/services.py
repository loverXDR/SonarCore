"""Agent Session Manager Service"""

import logging
import traceback
import uuid
from typing import List, Optional, Dict

import httpx
import qdrant_client
from fastapi.concurrency import run_in_threadpool

from Core.Schemas import AgentConfig, AgentResponse, AgentMessage
from Agent_workflow.pipeline import SonarPipeline
from Agent_workflow.agent import SonarAgent
from .repositories import BaseSessionRepository
from .schemas import SessionInfo

logger = logging.getLogger(__name__)

# In-process progress store for async tasks (not persisted across restarts)
_task_progress: Dict[str, dict] = {}


class AgentManager:
    """Manages active SonarAgent sessions."""

    def __init__(self, repository: BaseSessionRepository, default_config: AgentConfig):
        self.repository = repository
        self.default_config = default_config

    async def create_session_from_audio(
        self,
        audio_path: str,
        use_diarization: bool = False,
        config: Optional[AgentConfig] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Process audio file and create a new agent session asynchronously.

        When *task_id* is supplied the method publishes progress updates into
        the module-level ``_task_progress`` dict so that the SSE progress
        endpoint can stream them to the client.

        Args:
            audio_path (str): Path to the audio file on the server.
            use_diarization (bool, optional): Whether to use speaker diarization. Defaults to False.
            config (AgentConfig, optional): Optional override for agent config. Defaults to None.
            task_id (str, optional): If provided, progress is written to ``_task_progress[task_id]``.

        Returns:
            str: Processed session ID.
        """
        pipeline_config = config or self.default_config
        pipeline = SonarPipeline(config=pipeline_config)
        session_id = str(uuid.uuid4())

        if task_id:
            _task_progress[task_id] = {"status": "transcribing", "pct": 10}

        try:
            text = await run_in_threadpool(
                pipeline.transcribe_audio_to_text, audio_path, use_diarization
            )

            if task_id:
                _task_progress[task_id] = {"status": "indexing", "pct": 60}

            agent = await run_in_threadpool(pipeline.process_text, text, session_id)
            self.repository.add(session_id, agent, pipeline_config)

            if task_id:
                _task_progress[task_id] = {
                    "status": "ready",
                    "pct": 100,
                    "session_id": session_id,
                }

            return session_id

        except Exception:
            if task_id:
                _task_progress[task_id] = {
                    "status": "error",
                    "pct": 0,
                    "detail": traceback.format_exc(),
                }
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
        self.repository.add(session_id, agent, pipeline_config)
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
        record = self.repository.get(session_id)
        if not record:
            raise ValueError(f"Session '{session_id}' not found")
        
        response = await run_in_threadpool(record.agent.chat, message)
        return response

    def get_all_sessions(self) -> List[SessionInfo]:
        """List all managed sessions as DTOs.

        Returns:
            List[SessionInfo]: A list containing metadata for all active sessions.
        """
        records = self.repository.list_all()
        return [
            SessionInfo(
                session_id=r.session_id,
                created_at=r.created_at,
                last_active=r.last_active,
                config=r.config,
                message_count=len(r.agent._history),
            )
            for r in records
        ]

    def get_session_details(self, session_id: str) -> SessionInfo:
        """Get detailed metadata for a single session.

        Args:
            session_id (str): Unique identifier of the session.

        Raises:
            ValueError: If the session ID doesn't exist.

        Returns:
            SessionInfo: Detailed metadata DTO for the session.
        """
        record = self.repository.get(session_id)
        if not record:
            raise ValueError(f"Session '{session_id}' not found")
        
        return SessionInfo(
            session_id=record.session_id,
            created_at=record.created_at,
            last_active=record.last_active,
            config=record.config,
            message_count=len(record.agent._history),
        )

    def get_session_history(self, session_id: str) -> List[AgentMessage]:
        """Get the full message history for a session.

        Args:
            session_id (str): Unique identifier of the session.

        Raises:
            ValueError: If the session ID doesn't exist.

        Returns:
            List[AgentMessage]: The complete conversation history.
        """
        record = self.repository.get(session_id)
        if not record:
            raise ValueError(f"Session '{session_id}' not found")
        
        # Map internal LangChain messages back to DTOs
        from langchain_core.messages import HumanMessage
        history = []
        for msg in record.agent._history:
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

    def delete_session(self, session_id: str) -> bool:
        """Delete an active session from the repository.

        Args:
            session_id (str): Unique session identifier.

        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        return self.repository.delete(session_id)

    @staticmethod
    def get_task_progress(task_id: str) -> Optional[dict]:
        """Return the current progress dict for *task_id*, or ``None``."""
        return _task_progress.get(task_id)

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
