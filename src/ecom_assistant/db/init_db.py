"""Database initialization and seed orchestration."""

from pathlib import Path

from ecom_assistant.db.session import Database
from ecom_assistant.services.catalog_import import import_catalog_csv


def initialize_database(database: Database, seed_path: Path) -> int:
    """Create tables and import seed products in one repeatable transaction."""
    database.initialize()
    with database.session() as session, session.begin():
        return import_catalog_csv(session, seed_path)