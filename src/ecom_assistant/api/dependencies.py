"""Request-scoped dependencies shared by API routes."""

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from ecom_assistant.db.session import Database


def get_database(request: Request) -> Database:
    """Return the database owned by the current application instance."""
    return request.app.state.database


def get_db_session(request: Request) -> Iterator[Session]:
    """Yield and close one SQLAlchemy session for the current request."""
    database: Database = request.app.state.database
    with database.session() as session:
        yield session