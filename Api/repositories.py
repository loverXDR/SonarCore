"""Session and task repositories for the API"""

import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import aiosqlite

from Core.Schemas import AgentConfig


@dataclass
class SessionRecord:
    """Persisted state of a session

    The live SonarAgent is not stored here: it is rebuilt from ``text``
    and ``config`` on demand (Qdrant embeddings are reused by text hash).

    Attributes:
        session_id (str): Unique identifier for the session.
        config (AgentConfig): The configuration used to initialize the agent.
        text (str): Source text (transcript or document) of the session.
        history (List[dict]): Chat history as serialized LangChain messages.
        created_at (datetime): Timestamp when the session was created.
        last_active (datetime): Timestamp of the last activity in the session.
    """
    session_id: str
    config: AgentConfig
    text: str
    history: List[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    def update_activity(self):
        """Update the last_active timestamp to the current time."""
        self.last_active = datetime.now()


class BaseSessionRepository(ABC):
    """Abstract base class for session repositories"""

    async def init(self) -> None:
        """Prepare the storage (create tables, etc.). No-op by default."""

    @abstractmethod
    async def add(self, session_id: str, config: AgentConfig, text: str) -> None:
        """Add a new session to the repository

        Args:
            session_id (str): Unique identifier for the session.
            config (AgentConfig): The agent's initial configuration.
            text (str): Source text the session was built from.
        """

    @abstractmethod
    async def get(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a session by its ID and mark it as active

        Args:
            session_id (str): Unique identifier of the session.

        Returns:
            Optional[SessionRecord]: The found session record or None.
        """

    @abstractmethod
    async def list_all(self) -> List[SessionRecord]:
        """List all available sessions

        Returns:
            List[SessionRecord]: A list containing all session records.
        """

    @abstractmethod
    async def save_history(self, session_id: str, history: List[dict]) -> None:
        """Replace the chat history of a session

        Args:
            session_id (str): Unique identifier of the session.
            history (List[dict]): Serialized LangChain messages.
        """

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a session from the repository

        Args:
            session_id (str): Unique identifier of the session to delete.

        Returns:
            bool: True if the session was found and deleted, False otherwise.
        """


class BaseTaskRepository(ABC):
    """Abstract base class for async task progress storage"""

    async def init(self) -> None:
        """Prepare the storage (create tables, etc.). No-op by default."""

    @abstractmethod
    async def set(self, task_id: str, progress: dict) -> None:
        """Store the current progress of a task

        Args:
            task_id (str): Unique identifier of the task.
            progress (dict): Progress payload (status, pct, ...).
        """

    @abstractmethod
    async def get(self, task_id: str) -> Optional[dict]:
        """Get the current progress of a task

        Args:
            task_id (str): Unique identifier of the task.

        Returns:
            Optional[dict]: Progress payload or None if unknown.
        """


class InMemorySessionRepository(BaseSessionRepository):
    """Memory-based session repository (lost on restart)"""

    def __init__(self):
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    async def add(self, session_id: str, config: AgentConfig, text: str) -> None:
        with self._lock:
            self._sessions[session_id] = SessionRecord(
                session_id=session_id, config=config, text=text
            )

    async def get(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            record = self._sessions.get(session_id)
            if record:
                record.update_activity()
            return record

    async def list_all(self) -> List[SessionRecord]:
        with self._lock:
            return list(self._sessions.values())

    async def save_history(self, session_id: str, history: List[dict]) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record:
                record.history = history
                record.update_activity()

    async def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


class InMemoryTaskRepository(BaseTaskRepository):
    """Memory-based task progress storage (lost on restart)"""

    def __init__(self):
        self._tasks: Dict[str, dict] = {}

    async def set(self, task_id: str, progress: dict) -> None:
        self._tasks[task_id] = progress

    async def get(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)


class _SQLiteMixin:
    """Shared connection helper for SQLite repositories"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _connect(self) -> aiosqlite.Connection:
        return aiosqlite.connect(self.db_path, timeout=30)


class SQLiteSessionRepository(_SQLiteMixin, BaseSessionRepository):
    """SQLite-backed session repository (survives restarts)"""

    async def init(self) -> None:
        async with self._connect() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    config      TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    history     TEXT NOT NULL DEFAULT '[]',
                    created_at  TEXT NOT NULL,
                    last_active TEXT NOT NULL
                )
                """
            )
            await db.commit()

    @staticmethod
    def _to_record(row: aiosqlite.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            config=AgentConfig.model_validate_json(row["config"]),
            text=row["text"],
            history=json.loads(row["history"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_active=datetime.fromisoformat(row["last_active"]),
        )

    async def add(self, session_id: str, config: AgentConfig, text: str) -> None:
        now = datetime.now().isoformat()
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO sessions (session_id, config, text, history, created_at, last_active) "
                "VALUES (?, ?, ?, '[]', ?, ?)",
                (session_id, config.model_dump_json(), text, now, now),
            )
            await db.commit()

    async def get(self, session_id: str) -> Optional[SessionRecord]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                "UPDATE sessions SET last_active = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )
            await db.commit()
            async with db.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return self._to_record(row) if row else None

    async def list_all(self) -> List[SessionRecord]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
        return [self._to_record(row) for row in rows]

    async def save_history(self, session_id: str, history: List[dict]) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE sessions SET history = ?, last_active = ? WHERE session_id = ?",
                (json.dumps(history, ensure_ascii=False), datetime.now().isoformat(), session_id),
            )
            await db.commit()

    async def delete(self, session_id: str) -> bool:
        async with self._connect() as db:
            cursor = await db.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            await db.commit()
            return cursor.rowcount > 0


class SQLiteTaskRepository(_SQLiteMixin, BaseTaskRepository):
    """SQLite-backed task progress storage (survives restarts)"""

    async def init(self) -> None:
        async with self._connect() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id    TEXT PRIMARY KEY,
                    progress   TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # Tasks that were running when the process died will never finish
            interrupted = json.dumps(
                {"status": "error", "pct": 0, "detail": "Interrupted by server restart"}
            )
            await db.execute(
                "UPDATE tasks SET progress = ?, updated_at = ? "
                "WHERE json_extract(progress, '$.status') NOT IN ('ready', 'error')",
                (interrupted, datetime.now().isoformat()),
            )
            await db.commit()

    async def set(self, task_id: str, progress: dict) -> None:
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO tasks (task_id, progress, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(task_id) DO UPDATE SET "
                "progress = excluded.progress, updated_at = excluded.updated_at",
                (task_id, json.dumps(progress, ensure_ascii=False), datetime.now().isoformat()),
            )
            await db.commit()

    async def get(self, task_id: str) -> Optional[dict]:
        async with self._connect() as db:
            async with db.execute(
                "SELECT progress FROM tasks WHERE task_id = ?", (task_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return json.loads(row[0]) if row else None
