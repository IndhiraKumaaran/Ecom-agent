"""Phase 1 application, persistence, and seed behavior tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select, func

from ecom_assistant.config import Settings
from ecom_assistant.db.init_db import initialize_database
from ecom_assistant.db.models import Inventory, Product
from ecom_assistant.db.session import Database
from ecom_assistant.main import create_app

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.csv"


def make_settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        database_url=database_url,
        catalog_seed_path=CATALOG_PATH,
        faq_persist_directory=Path(database_url.removeprefix("sqlite:///" )).parent / "chroma",
    )


def test_application_starts_offline_and_health_endpoints_are_ready(tmp_path: Path) -> None:
    app = create_app(make_settings(f"sqlite:///{tmp_path / 'startup.db'}"))

    with TestClient(app) as client:
        versioned = client.get("/api/v1/health")
        convenience = client.get("/health")

    assert versioned.status_code == 200
    assert versioned.json() == {"status": "ok", "mode": "mock"}
    assert convenience.status_code == 200
    assert app.title == "E-Commerce Shopping Assistant"


def test_database_initialization_creates_required_tables(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'schema.db'}")
    try:
        database.initialize()
        table_names = set(inspect(database.engine).get_table_names())
    finally:
        database.close()

    assert {"products", "inventory", "carts", "cart_items", "faq_documents"} <= table_names


def test_catalog_seeding_is_repeatable_and_marks_sample_inventory(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'seed.db'}")
    try:
        first_count = initialize_database(database, CATALOG_PATH)
        second_count = initialize_database(database, CATALOG_PATH)

        with database.session() as session:
            product_count = session.scalar(select(func.count()).select_from(Product))
            inventory_count = session.scalar(select(func.count()).select_from(Inventory))
            unavailable = session.get(Inventory, "LAP-MRL-014")
    finally:
        database.close()

    assert first_count == second_count == 10
    assert product_count == inventory_count == 10
    assert unavailable is not None
    assert unavailable.stock_quantity == 0
    assert unavailable.data_source == "sample"


def test_openai_mode_requires_nonblank_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        Settings(_env_file=None, llm_mode="openai")

    with pytest.raises(ValueError, match="must not be blank"):
        Settings(_env_file=None, llm_mode="openai", openai_api_key="   ")


def test_non_sqlite_database_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="must use SQLite"):
        Settings(_env_file=None, database_url="postgresql://localhost/catalog")