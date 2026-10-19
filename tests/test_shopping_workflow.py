"""Offline tests for shopping requirement extraction, tools, and graph integration."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ecom_assistant.config import Settings
from ecom_assistant.db.models import Product
from ecom_assistant.main import create_app
from ecom_assistant.schemas.shopping import ShoppingRecommendationRequest
from ecom_assistant.workflow.requirements import (
    RequirementExtractionError,
    RequirementExtractor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_REQUEST = "laptop under ₹60,000 with at least 16 GB RAM and good battery life"


@pytest.fixture
def shopping_client(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'shopping.db'}",
        catalog_seed_path=PROJECT_ROOT / "data" / "catalog.csv",
        faq_source_path=PROJECT_ROOT / "data" / "faq_documents.json",
        faq_persist_directory=tmp_path / "chroma",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app, settings


def test_mock_extractor_parses_explicit_example_without_api_key(shopping_client) -> None:
    _client, _app, settings = shopping_client
    extractor = RequirementExtractor(settings)

    requirements = extractor.extract(EXAMPLE_REQUEST)

    assert extractor.mode == "mock"
    assert settings.openai_api_key is None
    assert requirements.category == "laptop"
    assert str(requirements.max_budget_inr) == "60000"
    assert requirements.min_ram_gb == 16
    assert requirements.min_storage_gb is None
    assert requirements.in_stock_only is None
    assert "good battery life" in requirements.other_preferences


def test_mock_extractor_parses_spelled_out_budget_and_ram() -> None:
    extractor = RequirementExtractor(Settings(_env_file=None))

    requirements = extractor.extract(
        "Find me a laptop below sixty thousand with at least sixteen gigs of RAM"
    )

    assert requirements.category == "laptop"
    assert requirements.max_budget_inr == 60000
    assert requirements.min_ram_gb == 16


@pytest.mark.parametrize(
    "budget_phrase",
    [
        "under 60k",
        "budget of ₹60,000",
        "max budget 60 thousand",
        "max 60k",
        "budget is ₹60,000",
    ],
)
def test_mock_extractor_scales_explicit_budget_phrases(shopping_client, budget_phrase: str) -> None:
    _client, _app, settings = shopping_client

    requirements = RequirementExtractor(settings).extract(f"laptop {budget_phrase}")

    assert requirements.max_budget_inr == 60000


def test_mock_extractor_maps_notebooks_and_negated_stock_preferences(shopping_client) -> None:
    _client, _app, settings = shopping_client

    requirements = RequirementExtractor(settings).extract(
        "notebook under 60k, stock status does not matter"
    )

    assert requirements.category == "laptop"
    assert requirements.max_budget_inr == 60000
    assert requirements.in_stock_only is False


def test_mock_extractor_converts_terabyte_storage_to_catalog_gigabytes(shopping_client) -> None:
    _client, _app, settings = shopping_client

    requirements = RequirementExtractor(settings).extract("laptop with at least 1 TB SSD")

    assert requirements.min_storage_gb == 1000


def test_all_four_langchain_tools_query_real_catalog_services(shopping_client) -> None:
    _client, app, _settings = shopping_client
    tools = app.state.shopping_workflow.tools

    search = tools["search"].invoke(
        {
            "filters": {
                "category": "laptop",
                "max_price_inr": "60000",
                "min_ram_gb": 16,
                "min_battery_hours": "10",
                "in_stock": True,
            }
        }
    )
    product_ids = [product["product_id"] for product in search["products"]]
    comparison = tools["compare"].invoke({"product_ids": product_ids})
    inventory = tools["inventory"].invoke({"product_id": "LAP-AST-017"})
    budget = tools["budget"].invoke(
        {"product_id": "LAP-AST-017", "budget_inr": "60000", "quantity": 1}
    )

    assert product_ids == ["LAP-AST-017", "LAP-SOL-015"]
    assert [product["product_id"] for product in comparison["products"]] == product_ids
    assert inventory["stock_status"] == "in_stock"
    assert budget["total_price_inr"] == "58990.00"
    assert budget["within_budget"] is True
    assert set(tools) == {"search", "compare", "inventory", "budget"}


def test_recommendation_api_runs_actual_graph_and_never_relaxes_hard_constraints(
    shopping_client,
) -> None:
    client, app, _settings = shopping_client

    response = client.post(
        "/api/v1/shopping/recommend",
        json={"customer_request": EXAMPLE_REQUEST},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["execution_metadata"]["workflow_version"] == "shopping-recommendation-v1"
    assert body["execution_metadata"]["candidate_count"] == 10
    assert body["extraction_mode"] == "mock"
    assert [stage["stage"] for stage in body["trace"]] == [
        "requirement_extraction",
        "product_search",
        "comparison",
        "inventory_validation",
        "budget_validation",
        "recommendation_ranking",
    ]
    assert {item["product"]["product_id"] for item in body["recommendations"]} == {
        "LAP-AST-017",
        "LAP-SOL-015",
    }
    assert all(item["strict_match"] for item in body["recommendations"])
    assert all(item["product"]["price_inr"] <= "60000.00" for item in body["recommendations"])
    assert all(item["product"]["ram_gb"] >= 16 for item in body["recommendations"])
    alternatives = {item["product"]["product_id"]: item for item in body["alternatives"]}
    assert "price ₹62990.00 exceeds maximum budget ₹60000" in alternatives["LAP-AST-014"]["relaxed_constraints"]
    assert any("below the minimum 16 GB" in reason for reason in alternatives["LAP-SOL-014"]["relaxed_constraints"])
    assert len(body["comparison_results"]) == len(body["inventory_results"]) == 10
    assert len(body["budget_validation"]) == 10
    assert body["matching_products"]
    assert app.state.shopping_workflow.graph.get_graph().nodes["product_search"]


def test_ranking_is_deterministic_across_invocations(shopping_client) -> None:
    client, _app, _settings = shopping_client
    request = {"customer_request": EXAMPLE_REQUEST}

    first = client.post("/api/v1/shopping/recommend", json=request).json()
    second = client.post("/api/v1/shopping/recommend", json=request).json()

    first_rank = [(row["product"]["product_id"], row["score"]) for row in first["recommendations"]]
    second_rank = [(row["product"]["product_id"], row["score"]) for row in second["recommendations"]]
    assert first_rank == second_rank


def test_budget_override_is_hard_and_only_relaxed_products_are_alternatives(
    shopping_client,
) -> None:
    client, _app, _settings = shopping_client

    response = client.post(
        "/api/v1/shopping/recommend",
        json={
            "customer_request": "laptop with at least 16 GB RAM",
            "preferences": {"max_budget_inr": "58000", "min_ram_gb": 16},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert all(
        row["product"]["price_inr"] <= "58000.00"
        and row["product"]["ram_gb"] >= 16
        for row in body["recommendations"]
    )
    assert any(
        row["product"]["product_id"] == "LAP-AST-017"
        and any("exceeds maximum budget" in constraint for constraint in row["relaxed_constraints"])
        for row in body["alternatives"]
    )


@pytest.mark.parametrize(
    ("customer_request", "expected_ids"),
    [
        (
            "laptop with at least 1000 GB storage",
            {"LAP-ORL-016", "LAP-CIR-015", "LAP-KES-016"},
        ),
        ("laptop with battery life at least 12 hours", {"LAP-VLT-013"}),
    ],
)
def test_storage_and_battery_minima_are_hard_constraints(
    shopping_client,
    customer_request: str,
    expected_ids: set[str],
) -> None:
    client, _app, _settings = shopping_client

    response = client.post(
        "/api/v1/shopping/recommend",
        json={"customer_request": customer_request},
    )

    assert response.status_code == 200
    body = response.json()
    assert {item["product"]["product_id"] for item in body["recommendations"]} == expected_ids
    assert all(
        item["product"]["storage_gb"] >= body["extracted_requirements"]["min_storage_gb"]
        if body["extracted_requirements"]["min_storage_gb"] is not None
        else item["product"]["battery_hours"] >= body["extracted_requirements"]["min_battery_hours"]
        for item in body["recommendations"]
    )


def test_impossible_budget_returns_only_labeled_alternatives(shopping_client) -> None:
    client, _app, _settings = shopping_client

    response = client.post(
        "/api/v1/shopping/recommend",
        json={
            "customer_request": "laptop under ₹10,000 with at least 16 GB RAM",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_strict_matches"
    assert body["recommendations"] == []
    assert body["alternatives"]
    assert all(item["strict_match"] is False for item in body["alternatives"])
    assert all(
        "exceeds maximum budget" in " ".join(item["relaxed_constraints"])
        for item in body["alternatives"]
    )


def test_no_matching_category_and_ambiguous_request_recover_usefully(shopping_client) -> None:
    client, _app, _settings = shopping_client

    no_results = client.post(
        "/api/v1/shopping/recommend",
        json={"customer_request": "desktop under ₹60,000"},
    )
    ambiguous = client.post(
        "/api/v1/shopping/recommend",
        json={"customer_request": "Help me find a suitable computer"},
    )

    assert no_results.status_code == 200
    assert no_results.json()["status"] == "no_results"
    assert no_results.json()["recommendations"] == []
    assert "No catalog products" in no_results.json()["warnings"][-1]
    assert ambiguous.status_code == 200
    assert ambiguous.json()["status"] == "needs_clarification"
    assert "Please specify" in ambiguous.json()["warnings"][-1]


def test_in_stock_preference_excludes_out_of_stock_strict_recommendations(shopping_client) -> None:
    client, _app, _settings = shopping_client

    response = client.post(
        "/api/v1/shopping/recommend",
        json={"customer_request": "laptop in stock under ₹80,000"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "LAP-MRL-014" not in {item["product"]["product_id"] for item in body["recommendations"]}
    unavailable = next(item for item in body["alternatives"] if item["product"]["product_id"] == "LAP-MRL-014")
    assert "out_of_stock" in unavailable["relaxed_constraints"][0]
    assert any(item["product_id"] == "LAP-MRL-014" and item["stock_status"] == "out_of_stock" for item in body["inventory_results"])


def test_incomplete_specification_is_not_fabricated(shopping_client) -> None:
    client, app, _settings = shopping_client
    with app.state.database.session() as session, session.begin():
        product = session.get(Product, "LAP-AST-017")
        product.cpu = ""

    response = client.post(
        "/api/v1/shopping/recommend",
        json={"customer_request": "laptop under ₹60,000 with 16 GB RAM"},
    )

    assert response.status_code == 200
    product = next(item for item in response.json()["matching_products"] if item["product_id"] == "LAP-AST-017")
    assert product["cpu"] is None
    assert product["specifications_complete"] is False
    assert any("incomplete specifications" in warning for warning in response.json()["warnings"])


def test_invalid_structured_overrides_are_rejected(shopping_client) -> None:
    client, _app, _settings = shopping_client

    response = client.post(
        "/api/v1/shopping/recommend",
        json={"customer_request": EXAMPLE_REQUEST, "preferences": {"max_budget_inr": 0}},
    )

    assert response.status_code == 422


def test_live_model_timeout_returns_failure_without_mock_fallback(shopping_client) -> None:
    client, app, settings = shopping_client
    live_settings = Settings(
        _env_file=None,
        app_env="test",
        llm_mode="openai",
        openai_api_key="test-key-not-a-secret",
        database_url=settings.database_url,
        catalog_seed_path=settings.catalog_seed_path,
        faq_source_path=settings.faq_source_path,
        faq_persist_directory=settings.faq_persist_directory,
    )

    class TimeoutModel:
        def invoke(self, _messages):
            raise TimeoutError("simulated timeout")

    app.state.shopping_workflow.extractor = RequirementExtractor(live_settings, model=TimeoutModel())
    response = client.post(
        "/api/v1/shopping/recommend",
        json={"customer_request": EXAMPLE_REQUEST},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "failed"
    assert body["recommendations"] == []
    assert body["extraction_mode"] == "openai"
    assert body["errors"] == ["Requirement extraction failed; no mock fallback was used."]
    assert "test-key-not-a-secret" not in response.text


def test_live_extractor_wraps_llm_failures_without_fallback() -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="openai",
        openai_api_key="test-key-not-a-secret",
    )

    class FailingModel:
        def invoke(self, _messages):
            raise TimeoutError("timeout")

    with pytest.raises(RequirementExtractionError):
        RequirementExtractor(settings, model=FailingModel()).extract(EXAMPLE_REQUEST)


def test_live_mode_uses_configured_chat_model_without_network_call() -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="openai",
        openai_api_key="test-key-not-a-secret",
        faq_chat_model="gpt-4o-mini",
    )

    extractor = RequirementExtractor(settings)

    assert extractor.mode == "openai"
    assert extractor._structured_model is not None


def test_live_structured_extraction_accepts_request_supported_values() -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="openai",
        openai_api_key="test-key-not-a-secret",
    )

    class StructuredModel:
        def invoke(self, _messages):
            return {
                "category": "laptop",
                "max_budget_inr": "60000",
                "min_ram_gb": 16,
                "min_storage_gb": None,
                "preferred_cpu": None,
                "min_battery_hours": None,
                "in_stock_only": None,
                "other_preferences": ["good battery life"],
            }

    requirements = RequirementExtractor(settings, model=StructuredModel()).extract(EXAMPLE_REQUEST)

    assert requirements.category == "laptop"
    assert requirements.max_budget_inr == 60000
    assert requirements.min_ram_gb == 16
    assert requirements.other_preferences == ["good battery life"]


def test_live_extractor_rejects_hallucinated_hard_constraints() -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="openai",
        openai_api_key="test-key-not-a-secret",
    )

    class HallucinatingModel:
        def invoke(self, _messages):
            return {
                "category": "laptop",
                "max_budget_inr": "120000",
                "min_ram_gb": 16,
                "min_storage_gb": None,
                "preferred_cpu": None,
                "min_battery_hours": None,
                "in_stock_only": None,
                "other_preferences": [],
            }

    extractor = RequirementExtractor(settings, model=HallucinatingModel())
    with pytest.raises(RequirementExtractionError):
        extractor.extract("laptop under ₹60,000 with at least 16 GB RAM")


def test_live_extractor_rejects_dropped_explicit_hard_constraints() -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="openai",
        openai_api_key="test-key-not-a-secret",
    )

    class OmittingModel:
        def invoke(self, _messages):
            return {
                "category": "laptop",
                "max_budget_inr": None,
                "min_ram_gb": None,
                "min_storage_gb": None,
                "preferred_cpu": None,
                "min_battery_hours": None,
                "in_stock_only": None,
                "other_preferences": [],
            }

    extractor = RequirementExtractor(settings, model=OmittingModel())
    with pytest.raises(RequirementExtractionError):
        extractor.extract("laptop under ₹60,000 with at least 16 GB RAM")


def test_shopping_request_and_mock_requirement_schemas_validate() -> None:
    with pytest.raises(ValueError):
        ShoppingRecommendationRequest(customer_request="   ")