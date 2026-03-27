"""Agent Session Manager Service"""

import uuid
from typing import Dict, Optional
from fastapi.concurrency import run_in_threadpool

from Core.Schemas import AgentConfig, AgentResponse
from Agent_workflow.pipeline import SonarPipeline
from Agent_workflow.agent import SonarAgent


class AgentManager:
    """Manages active SonarAgent sessions."""

    def __init__(self, default_config: AgentConfig):
        self.default_config = default_config
        self.sessions: Dict[str, SonarAgent] = {}

    async def create_session_from_audio(
        self, 
        audio_path: str, 
        use_diarization: bool = False, 
        config: Optional[AgentConfig] = None
    ) -> str:
        """Process audio file and create a new agent session asynchronously.

        Args:
            audio_path (str): Path to the audio file on the server.
            use_diarization (bool, optional): Whether to use speaker diarization. Defaults to False.
            config (AgentConfig, optional): Optional override for agent config. Defaults to None.

        Returns:
            str: Processed session ID.
        """
        pipeline_config = config or self.default_config
        pipeline = SonarPipeline(config=pipeline_config)
        agent = await run_in_threadpool(
            pipeline.process_audio, audio_path, use_diarization
        )
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = agent
        return session_id

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
        agent = await run_in_threadpool(pipeline.process_text, text)
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = agent
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
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' not found")
        
        agent = self.sessions[session_id]
        response = await run_in_threadpool(agent.chat, message)
        return response

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
        """Delete an active session from memory.

        Args:
            session_id (str): Unique session identifier.

        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
