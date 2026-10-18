"""Command-line entry point for refreshing the persistent FAQ index."""

import logging

from ecom_assistant.config import Settings
from ecom_assistant.db.init_db import initialize_database
from ecom_assistant.db.session import Database
from ecom_assistant.rag.service import FAQService


def main() -> None:
    """Initialize source tables and reconcile the configured Chroma collection."""
    settings = Settings()
    database = Database(settings.database_url)
    faq_service = FAQService.from_settings(settings)
    try:
        initialize_database(database, settings.catalog_seed_path)
        with database.session() as session, session.begin():
            report = faq_service.ingest(session, settings.faq_source_path)
        logging.basicConfig(level=settings.log_level)
        logging.getLogger(__name__).info(
            "FAQ index refresh complete: %d indexed, %d updated, %d stale removed",
            report.indexed_documents,
            report.updated_documents,
            report.deleted_documents,
        )
    finally:
        database.close()


if __name__ == "__main__":
    main()