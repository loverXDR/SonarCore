"""Session repositories for the API"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from Agent_workflow.agent import SonarAgent
from Core.Schemas import AgentConfig


@dataclass
class SessionRecord:
    """Internal record for a session

    Attributes:
        session_id (str): Unique identifier for the session.
        agent (SonarAgent): The agent instance associated with the session.
        config (AgentConfig): The configuration used to initialize the agent.
        created_at (datetime): Timestamp when the session was created.
        last_active (datetime): Timestamp of the last activity in the session.
    """
    session_id: str
    agent: SonarAgent
    config: AgentConfig
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    def update_activity(self):
        """Update the last_active timestamp to the current time."""
        self.last_active = datetime.now()


class BaseSessionRepository(ABC):
    """Abstract base class for session repositories"""

    @abstractmethod
    def add(self, session_id: str, agent: SonarAgent, config: AgentConfig) -> None:
        """Add a new session to the repository

        Args:
            session_id (str): Unique identifier for the session.
            agent (SonarAgent): The agent instance to store.
            config (AgentConfig): The agent's initial configuration.
        """
        pass

    @abstractmethod
    def get(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a session by its ID

        Args:
            session_id (str): Unique identifier of the session.

        Returns:
            Optional[SessionRecord]: The found session record or None.
        """
        pass

    @abstractmethod
    def list_all(self) -> List[SessionRecord]:
        """List all available sessions

        Returns:
            List[SessionRecord]: A list containing all session records.
        """
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session from the repository

        Args:
            session_id (str): Unique identifier of the session to delete.

        Returns:
            bool: True if the session was found and deleted, False otherwise.
        """
        pass


class InMemorySessionRepository(BaseSessionRepository):
    """Memory-based session repository"""

    def __init__(self):
        self._sessions: Dict[str, SessionRecord] = {}

    def add(self, session_id: str, agent: SonarAgent, config: AgentConfig) -> None:
        """Add a new session memory-based repository

        Args:
            session_id (str): Unique identifier for the session.
            agent (SonarAgent): The agent instance.
            config (AgentConfig): Agent configuration.
        """
        self._sessions[session_id] = SessionRecord(
            session_id=session_id,
            agent=agent,
            config=config
        )

    def get(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a session from memory

        Args:
            session_id (str): session ID.

        Returns:
            Optional[SessionRecord]: session record or None.
        """
        record = self._sessions.get(session_id)
        if record:
            record.update_activity()
        return record

    def list_all(self) -> List[SessionRecord]:
        """List all active sessions in memory

        Returns:
            List[SessionRecord]: list of active records.
        """
        return list(self._sessions.values())

    def delete(self, session_id: str) -> bool:
        """Delete a session from dictionary

        Args:
            session_id (str): session ID.

        Returns:
            bool: True if deleted.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
