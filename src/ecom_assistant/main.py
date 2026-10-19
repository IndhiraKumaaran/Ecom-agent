"""FastAPI application factory and startup lifecycle."""

from contextlib import asynccontextmanager
from logging import getLogger
from typing import AsyncIterator

from fastapi import FastAPI

from ecom_assistant import __version__
from ecom_assistant.api.router import api_router
from ecom_assistant.api.v1.health import router as health_router
from ecom_assistant.config import Settings
from ecom_assistant.db.init_db import initialize_database
from ecom_assistant.db.session import Database
from ecom_assistant.logging_config import configure_logging
from ecom_assistant.rag.service import FAQService
from ecom_assistant.services.catalog import CatalogService
from ecom_assistant.services.cart import CartService
from ecom_assistant.workflow.graph import ShoppingWorkflow
from ecom_assistant.workflow.conversation import ConversationWorkflow

logger = getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an application instance with isolated settings and database."""
    application_settings = settings or Settings()
    database = Database(application_settings.database_url)
    faq_service = FAQService.from_settings(application_settings)
    catalog_service = CatalogService()
    cart_service = CartService(application_settings, catalog_service)
    shopping_workflow = ShoppingWorkflow(application_settings, database, catalog_service)
    conversation_workflow = ConversationWorkflow(
        application_settings,
        database,
        catalog_service,
        shopping_workflow,
        cart_service,
        faq_service,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        configure_logging(application_settings.log_level)
        logger.info("Initializing application in %s mode", application_settings.llm_mode)
        try:
            initialized_products = initialize_database(
                database, application_settings.catalog_seed_path
            )
            logger.info("Catalog initialization complete; processed %d products", initialized_products)
            with database.session() as session, session.begin():
                faq_ingestion = faq_service.ingest(session, application_settings.faq_source_path)
            logger.info(
                "FAQ index ready; indexed %d documents, removed %d stale documents",
                faq_ingestion.indexed_documents,
                faq_ingestion.deleted_documents,
            )
            yield
        finally:
            database.close()

    app = FastAPI(
        title=application_settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = application_settings
    app.state.database = database
    app.state.faq_service = faq_service
    app.state.catalog_service = catalog_service
    app.state.cart_service = cart_service
    app.state.shopping_workflow = shopping_workflow
    app.state.conversation_workflow = conversation_workflow
    app.include_router(api_router)
    app.include_router(health_router)
    return app


app = create_app()