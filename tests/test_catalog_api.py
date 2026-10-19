"""Deterministic catalog API tests using seeded SQLite records."""

from pathlib import Path
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from ecom_assistant.config import Settings
from ecom_assistant.db.models import Inventory, Product
from ecom_assistant.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def catalog_client(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'catalog.db'}",
        catalog_seed_path=PROJECT_ROOT / "data" / "catalog.csv",
        faq_source_path=PROJECT_ROOT / "data" / "faq_documents.json",
        faq_persist_directory=tmp_path / "chroma",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


def test_search_filters_price_ram_and_stock(catalog_client) -> None:
    client, _app = catalog_client

    response = client.get(
        "/api/v1/products",
        params={
            "max_price_inr": "60000",
            "min_ram_gb": 16,
            "in_stock": "true",
            "sort_by": "price_asc",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert {product["product_id"] for product in body["products"]} == {
        "LAP-AST-017",
        "LAP-SOL-015",
    }
    assert all(Decimal(product["price_inr"]) <= Decimal("60000") for product in body["products"])
    assert all(product["ram_gb"] >= 16 for product in body["products"])
    assert body["sample_data"] is True


def test_search_matches_text_and_brand(catalog_client) -> None:
    client, _app = catalog_client

    response = client.get(
        "/api/v1/products",
        params={"query": "long-battery", "brand": "Asterra"},
    )

    assert response.status_code == 200
    assert [product["product_id"] for product in response.json()["products"]] == ["LAP-AST-017"]


def test_search_returns_empty_result_for_no_matches(catalog_client) -> None:
    client, _app = catalog_client

    response = client.get("/api/v1/products", params={"query": "not-a-real-product"})

    assert response.status_code == 200
    assert response.json()["products"] == []
    assert response.json()["total"] == 0


@pytest.mark.parametrize(
    "params",
    [
        {"min_price_inr": "60000", "max_price_inr": "50000"},
        {"sort_by": "random"},
        {"min_ram_gb": 0},
        {"query": "   "},
    ],
)
def test_invalid_search_filters_return_422(catalog_client, params: dict[str, str]) -> None:
    client, _app = catalog_client

    response = client.get("/api/v1/products", params=params)

    assert response.status_code == 422


def test_product_detail_and_missing_product(catalog_client) -> None:
    client, _app = catalog_client

    found = client.get("/api/v1/products/LAP-AST-017")
    missing = client.get("/api/v1/products/UNKNOWN-001")

    assert found.status_code == 200
    assert found.json()["name"] == "AsterBook Air 14 Plus"
    assert found.json()["specifications_complete"] is True
    assert found.json()["price_inr"] == "58990.00"
    assert missing.status_code == 404


def test_comparison_uses_stored_products_and_preserves_request_order(catalog_client) -> None:
    client, _app = catalog_client

    response = client.post(
        "/api/v1/products/compare",
        json={"product_ids": ["LAP-SOL-015", "LAP-AST-017"]},
    )

    assert response.status_code == 200
    products = response.json()["products"]
    assert [product["product_id"] for product in products] == ["LAP-SOL-015", "LAP-AST-017"]
    assert products[0]["cpu"] == "Intel Core i5-1235U"
    assert products[1]["battery_hours"] == "11.0"


def test_comparison_rejects_duplicates_and_unknown_ids(catalog_client) -> None:
    client, _app = catalog_client

    duplicate = client.post(
        "/api/v1/products/compare",
        json={"product_ids": ["LAP-AST-017", "LAP-AST-017"]},
    )
    unknown = client.post(
        "/api/v1/products/compare",
        json={"product_ids": ["LAP-AST-017", "UNKNOWN-001"]},
    )

    assert duplicate.status_code == 422
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["product_ids"] == ["UNKNOWN-001"]


def test_incomplete_specification_is_reported(catalog_client) -> None:
    client, app = catalog_client
    with app.state.database.session() as session, session.begin():
        product = session.get(Product, "LAP-AST-017")
        product.cpu = ""

    response = client.get("/api/v1/products/LAP-AST-017")

    assert response.status_code == 200
    assert response.json()["cpu"] is None
    assert response.json()["specifications_complete"] is False
    assert "processor" in response.json()["missing_specifications"]


def test_inventory_reports_available_and_unavailable_sample_stock(catalog_client) -> None:
    client, _app = catalog_client

    available = client.get("/api/v1/inventory/LAP-AST-017")
    unavailable = client.get("/api/v1/inventory/LAP-MRL-014")
    missing = client.get("/api/v1/inventory/UNKNOWN-001")

    assert available.json()["stock_status"] == "in_stock"
    assert available.json()["stock_quantity"] == 9
    assert available.json()["data_source"] == "sample"
    assert unavailable.json()["stock_status"] == "out_of_stock"
    assert unavailable.json()["stock_quantity"] == 0
    assert missing.status_code == 404


def test_missing_inventory_is_unknown_not_treated_as_out_of_stock(catalog_client) -> None:
    client, app = catalog_client
    with app.state.database.session() as session, session.begin():
        inventory = session.get(Inventory, "LAP-AST-017")
        session.delete(inventory)

    inventory_response = client.get("/api/v1/inventory/LAP-AST-017")
    out_of_stock_response = client.get("/api/v1/products", params={"in_stock": "false"})

    assert inventory_response.status_code == 200
    assert inventory_response.json()["stock_status"] == "unknown"
    assert "LAP-AST-017" not in {
        product["product_id"] for product in out_of_stock_response.json()["products"]
    }


def test_budget_validation_uses_server_price_and_quantity(catalog_client) -> None:
    client, _app = catalog_client

    exact = client.post(
        "/api/v1/budget/validate",
        json={"product_id": "LAP-AST-017", "budget_inr": "117980.00", "quantity": 2},
    )
    over = client.post(
        "/api/v1/budget/validate",
        json={"product_id": "LAP-AST-017", "budget_inr": "117979.99", "quantity": 2},
    )
    invalid = client.post(
        "/api/v1/budget/validate",
        json={"product_id": "LAP-AST-017", "budget_inr": "0", "quantity": 1},
    )

    assert exact.status_code == 200
    assert exact.json()["total_price_inr"] == "117980.00"
    assert exact.json()["within_budget"] is True
    assert over.json()["within_budget"] is False
    assert over.json()["over_budget_inr"] == "0.01"
    assert invalid.status_code == 422


def test_budget_validation_handles_missing_product(catalog_client) -> None:
    client, _app = catalog_client

    response = client.post(
        "/api/v1/budget/validate",
        json={"product_id": "UNKNOWN-001", "budget_inr": "60000.00"},
    )

    assert response.status_code == 404