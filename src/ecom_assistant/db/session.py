"""SQLite engine and request-scoped SQLAlchemy sessions."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ecom_assistant.db.base import Base


class Database:
    """Own the engine and create independent sessions for each request."""

    def __init__(self, database_url: str) -> None:
        url = make_url(database_url)
        if url.get_backend_name() != "sqlite":
            raise ValueError("Only SQLite database URLs are supported")

        is_memory_database = url.database in (None, "", ":memory:")
        if not is_memory_database and url.database:
            Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

        connect_args = {"check_same_thread": False, "timeout": 30}
        engine_options = {"connect_args": connect_args, "future": True}
        if is_memory_database:
            engine_options["poolclass"] = StaticPool
        self.engine: Engine = create_engine(database_url, **engine_options)
        event.listen(self.engine, "connect", self._enable_sqlite_constraints)
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    @staticmethod
    def _enable_sqlite_constraints(connection, _record) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    def initialize(self) -> None:
        """Create missing tables while preserving existing data."""
        from ecom_assistant.db import models  # noqa: F401

        Base.metadata.create_all(self.engine)
        faq_columns = {column["name"] for column in inspect(self.engine).get_columns("faq_documents")}
        if "document_type" not in faq_columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE faq_documents "
                        "ADD COLUMN document_type VARCHAR(32) NOT NULL DEFAULT 'faq'"
                    )
                )

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session and always close it after the unit of work."""
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    def close(self) -> None:
        """Release pooled SQLite connections."""
        self.engine.dispose()