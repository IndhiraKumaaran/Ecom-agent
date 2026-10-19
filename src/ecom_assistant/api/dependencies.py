"""Request-scoped dependencies shared by API routes."""

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from ecom_assistant.db.session import Database
from ecom_assistant.services.catalog import CatalogService
from ecom_assistant.services.cart import CartService
from ecom_assistant.workflow.conversation import ConversationWorkflow


def get_database(request: Request) -> Database:
    """Return the database owned by the current application instance."""
    return request.app.state.database


def get_catalog_service(request: Request) -> CatalogService:
    """Return the deterministic catalog service owned by the application."""
    return request.app.state.catalog_service


def get_cart_service(request: Request) -> CartService:
    """Return the simulated-cart service owned by this application instance."""
    return request.app.state.cart_service


def get_conversation_workflow(request: Request) -> ConversationWorkflow:
    """Return the conversation graph owned by the current application instance."""
    return request.app.state.conversation_workflow


def get_db_session(request: Request) -> Iterator[Session]:
    """Yield and close one SQLAlchemy session for the current request."""
    database: Database = request.app.state.database
    with database.session() as session:
        yield session