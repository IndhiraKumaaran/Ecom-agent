"""Simulated cart proposal and explicit-confirmation API tests."""

from datetime import UTC, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from ecom_assistant.config import Settings
from ecom_assistant.db.models import Cart, CartItem, CartProposal, Inventory, Product
from ecom_assistant.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cart_client(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'cart.db'}",
        catalog_seed_path=PROJECT_ROOT / "data" / "catalog.csv",
        faq_source_path=PROJECT_ROOT / "data" / "faq_documents.json",
        faq_persist_directory=tmp_path / "chroma",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


def create_cart(client: TestClient) -> dict:
    response = client.post("/api/v1/cart")
    assert response.status_code == 201
    return response.json()


def propose_add(client: TestClient, cart_id: str, product_id: str = "LAP-AST-017", quantity: int = 1) -> dict:
    response = client.post(
        f"/api/v1/cart/{cart_id}/proposals",
        json={"operation": "add", "product_id": product_id, "quantity": quantity},
    )
    assert response.status_code == 201, response.text
    return response.json()


def confirm(client: TestClient, cart_id: str, proposal_id: str, confirm_value: bool = True):
    return client.post(
        f"/api/v1/cart/{cart_id}/confirmations/{proposal_id}",
        json={"confirm": confirm_value},
    )


def test_cart_initialization_and_empty_totals(cart_client) -> None:
    client, _app = cart_client

    cart = create_cart(client)
    retrieved = client.get(f"/api/v1/cart/{cart['cart_id']}")

    assert cart["simulated"] is True
    assert cart["status"] == "open"
    assert cart["items"] == []
    assert cart["total_price_inr"] == "0.00"
    assert retrieved.status_code == 200
    assert retrieved.json() == cart


@pytest.mark.parametrize("quantity", [0, -1, 51, True])
def test_invalid_quantities_are_rejected(cart_client, quantity: int | bool) -> None:
    client, _app = cart_client
    cart = create_cart(client)

    response = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "add", "product_id": "LAP-AST-017", "quantity": quantity},
    )

    assert response.status_code == 422


def test_unknown_product_and_client_supplied_price_are_rejected(cart_client) -> None:
    client, _app = cart_client
    cart = create_cart(client)

    unknown = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "add", "product_id": "UNKNOWN-001", "quantity": 1},
    )
    supplied_price = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={
            "operation": "add",
            "product_id": "LAP-AST-017",
            "quantity": 1,
            "unit_price_inr": "1.00",
        },
    )

    assert unknown.status_code == 404
    assert supplied_price.status_code == 422


def test_proposal_does_not_mutate_cart_and_uses_server_price(cart_client) -> None:
    client, _app = cart_client
    cart = create_cart(client)

    proposal = propose_add(client, cart["cart_id"], quantity=2)
    unchanged = client.get(f"/api/v1/cart/{cart['cart_id']}").json()

    assert proposal["status"] == "pending"
    assert proposal["unit_price_inr"] == "58990.00"
    assert proposal["projected_total_inr"] == "117980.00"
    assert unchanged["items"] == []
    assert unchanged["total_price_inr"] == "0.00"


def test_false_or_missing_confirmation_never_mutates_cart(cart_client) -> None:
    client, _app = cart_client
    cart = create_cart(client)
    proposal = propose_add(client, cart["cart_id"])

    false_confirmation = confirm(client, cart["cart_id"], proposal["proposal_id"], False)
    missing_confirmation = client.post(
        f"/api/v1/cart/{cart['cart_id']}/confirmations/{proposal['proposal_id']}",
        json={},
    )
    non_boolean_confirmation = client.post(
        f"/api/v1/cart/{cart['cart_id']}/confirmations/{proposal['proposal_id']}",
        json={"confirm": 1},
    )

    assert false_confirmation.status_code == 422
    assert missing_confirmation.status_code == 422
    assert non_boolean_confirmation.status_code == 422
    assert client.get(f"/api/v1/cart/{cart['cart_id']}").json()["items"] == []


def test_confirmation_applies_exact_proposal_and_duplicate_is_idempotent(cart_client) -> None:
    client, app = cart_client
    cart = create_cart(client)
    proposal = propose_add(client, cart["cart_id"], quantity=2)

    first = confirm(client, cart["cart_id"], proposal["proposal_id"])
    second = confirm(client, cart["cart_id"], proposal["proposal_id"])
    retrieved = client.get(f"/api/v1/cart/{cart['cart_id']}").json()

    assert first.status_code == 200
    assert first.json()["status"] == "confirmed"
    assert first.json()["cart"]["total_price_inr"] == "117980.00"
    assert "no purchase" in first.json()["message"]
    assert second.status_code == 200
    assert second.json()["status"] == "already_confirmed"
    assert retrieved["total_price_inr"] == "117980.00"
    assert len(retrieved["items"]) == 1
    with app.state.database.session() as session:
        assert session.query(CartItem).filter_by(cart_id=cart["cart_id"]).one().quantity == 2


def test_simultaneous_confirmation_replay_applies_only_once(cart_client) -> None:
    client, app = cart_client
    cart = create_cart(client)
    proposal = propose_add(client, cart["cart_id"], quantity=1)
    barrier = Barrier(2)

    def confirm_in_independent_session() -> str:
        barrier.wait()
        with app.state.database.session() as session:
            result = app.state.cart_service.confirm(
                session,
                cart["cart_id"],
                proposal["proposal_id"],
            )
        return result.status

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: confirm_in_independent_session(), range(2)))

    contents = client.get(f"/api/v1/cart/{cart['cart_id']}").json()
    assert sorted(results) == ["already_confirmed", "confirmed"]
    assert len(contents["items"]) == 1
    assert contents["items"][0]["quantity"] == 1
    assert contents["total_price_inr"] == "58990.00"


def test_expired_proposal_is_rejected_without_mutation(cart_client) -> None:
    client, app = cart_client
    cart = create_cart(client)
    proposal = propose_add(client, cart["cart_id"])
    with app.state.database.session() as session, session.begin():
        row = session.get(CartProposal, proposal["proposal_id"])
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)

    response = confirm(client, cart["cart_id"], proposal["proposal_id"])

    assert response.status_code == 410
    assert response.json()["status"] == "expired"
    assert client.get(f"/api/v1/cart/{cart['cart_id']}").json()["items"] == []


def test_changed_product_price_marks_proposal_stale(cart_client) -> None:
    client, app = cart_client
    cart = create_cart(client)
    proposal = propose_add(client, cart["cart_id"])
    with app.state.database.session() as session, session.begin():
        product = session.get(Product, "LAP-AST-017")
        product.price_inr = product.price_inr + 1

    response = confirm(client, cart["cart_id"], proposal["proposal_id"])

    assert response.status_code == 409
    assert response.json()["status"] == "stale"
    assert client.get(f"/api/v1/cart/{cart['cart_id']}").json()["items"] == []


def test_changed_inventory_marks_proposal_stale(cart_client) -> None:
    client, app = cart_client
    cart = create_cart(client)
    proposal = propose_add(client, cart["cart_id"])
    with app.state.database.session() as session, session.begin():
        inventory = session.get(Inventory, "LAP-AST-017")
        inventory.stock_quantity -= 1

    response = confirm(client, cart["cart_id"], proposal["proposal_id"])

    assert response.status_code == 409
    assert response.json()["status"] == "stale"
    assert client.get(f"/api/v1/cart/{cart['cart_id']}").json()["items"] == []


def test_insufficient_and_zero_stock_are_rejected(cart_client) -> None:
    client, _app = cart_client
    cart = create_cart(client)

    insufficient = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "add", "product_id": "LAP-AST-017", "quantity": 10},
    )
    unavailable = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "add", "product_id": "LAP-MRL-014", "quantity": 1},
    )

    assert insufficient.status_code == 409
    assert unavailable.status_code == 409
    assert client.get(f"/api/v1/cart/{cart['cart_id']}").json()["items"] == []


def test_unknown_inventory_is_not_treated_as_available(cart_client) -> None:
    client, app = cart_client
    cart = create_cart(client)
    with app.state.database.session() as session, session.begin():
        session.delete(session.get(Inventory, "LAP-AST-017"))

    proposal = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "add", "product_id": "LAP-AST-017", "quantity": 1},
    )

    assert proposal.status_code == 409
    assert "Inventory is unknown" in proposal.json()["detail"]
    assert client.get(f"/api/v1/cart/{cart['cart_id']}").json()["items"] == []


def test_confirmed_simulated_carts_cannot_exceed_shared_sample_stock(cart_client) -> None:
    client, app = cart_client
    first_cart = create_cart(client)
    second_cart = create_cart(client)
    with app.state.database.session() as session, session.begin():
        inventory = session.get(Inventory, "LAP-AST-017")
        inventory.stock_quantity = 3

    first_proposal = propose_add(client, first_cart["cart_id"], quantity=2)
    second_proposal = propose_add(client, second_cart["cart_id"], quantity=2)
    first_result = confirm(client, first_cart["cart_id"], first_proposal["proposal_id"])
    second_result = confirm(client, second_cart["cart_id"], second_proposal["proposal_id"])

    assert first_result.status_code == 200
    assert first_result.json()["status"] == "confirmed"
    assert second_result.status_code == 409
    assert second_result.json()["status"] == "rejected"
    assert first_result.json()["cart"]["items"][0]["quantity"] == 2
    assert second_result.json()["cart"]["items"] == []


def test_update_and_remove_are_each_explicitly_confirmed(cart_client) -> None:
    client, _app = cart_client
    cart = create_cart(client)
    add_proposal = propose_add(client, cart["cart_id"], quantity=2)
    assert confirm(client, cart["cart_id"], add_proposal["proposal_id"]).json()["status"] == "confirmed"

    update_proposal = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "update", "product_id": "LAP-AST-017", "quantity": 3},
    ).json()
    assert client.get(f"/api/v1/cart/{cart['cart_id']}").json()["items"][0]["quantity"] == 2
    assert confirm(client, cart["cart_id"], update_proposal["proposal_id"]).json()["status"] == "confirmed"
    assert client.get(f"/api/v1/cart/{cart['cart_id']}").json()["total_price_inr"] == "176970.00"

    remove_proposal = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "remove", "product_id": "LAP-AST-017"},
    ).json()
    assert remove_proposal["projected_total_inr"] == "0.00"
    remove_result = confirm(client, cart["cart_id"], remove_proposal["proposal_id"])
    assert remove_result.json()["status"] == "confirmed"
    assert remove_result.json()["cart"]["items"] == []
    final = client.get(f"/api/v1/cart/{cart['cart_id']}").json()
    assert final["items"] == []
    assert final["total_price_inr"] == "0.00"


def test_cart_revision_change_invalidates_other_pending_proposals(cart_client) -> None:
    client, _app = cart_client
    cart = create_cart(client)
    first = propose_add(client, cart["cart_id"], "LAP-AST-017", 1)
    second = propose_add(client, cart["cart_id"], "LAP-SOL-015", 1)

    assert confirm(client, cart["cart_id"], first["proposal_id"]).json()["status"] == "confirmed"
    stale = confirm(client, cart["cart_id"], second["proposal_id"])

    assert stale.status_code == 409
    assert stale.json()["status"] == "stale"
    contents = client.get(f"/api/v1/cart/{cart['cart_id']}").json()
    assert [item["product_id"] for item in contents["items"]] == ["LAP-AST-017"]


def test_missing_cart_and_invalid_cart_or_proposal_ids(cart_client) -> None:
    client, _app = cart_client

    missing = client.get("/api/v1/cart/00000000-0000-0000-0000-000000000000")
    invalid_cart = client.get("/api/v1/cart/not-a-uuid")
    invalid_proposal = client.post(
        "/api/v1/cart/00000000-0000-0000-0000-000000000000/confirmations/not-a-uuid",
        json={"confirm": True},
    )

    assert missing.status_code == 404
    assert invalid_cart.status_code == 422
    assert invalid_proposal.status_code == 422


def test_mutation_operations_require_their_exact_quantities(cart_client) -> None:
    client, _app = cart_client
    cart = create_cart(client)

    update_without_quantity = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "update", "product_id": "LAP-AST-017"},
    )
    remove_with_quantity = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "remove", "product_id": "LAP-AST-017", "quantity": 1},
    )
    unsupported_product = client.post(
        f"/api/v1/cart/{cart['cart_id']}/proposals",
        json={"operation": "purchase", "product_id": "LAP-AST-017", "quantity": 1},
    )

    assert update_without_quantity.status_code == 422
    assert remove_with_quantity.status_code == 422
    assert unsupported_product.status_code == 422


def test_catalog_and_recommendation_apis_remain_available(cart_client) -> None:
    client, _app = cart_client

    products = client.get("/api/v1/products", params={"max_price_inr": 60000, "min_ram_gb": 16})
    recommendation = client.post(
        "/api/v1/shopping/recommend",
        json={"customer_request": "laptop under ₹60,000 with at least 16 GB RAM"},
    )

    assert products.status_code == 200
    assert products.json()["total"] == 2
    assert recommendation.status_code == 200
    assert recommendation.json()["status"] == "success"