"""Run a disposable end-to-end mock-mode shopping assistant demonstration."""

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from ecom_assistant.config import Settings
from ecom_assistant.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Exercise the main shopping and simulated-cart flow with temporary storage."""
    with TemporaryDirectory(prefix="ecom-assistant-demo-") as directory:
        root = Path(directory)
        settings = Settings(
            _env_file=None,
            app_env="test",
            llm_mode="mock",
            database_url=f"sqlite:///{root / 'demo.db'}",
            catalog_seed_path=PROJECT_ROOT / "data" / "catalog.csv",
            faq_source_path=PROJECT_ROOT / "data" / "faq_documents.json",
            faq_persist_directory=root / "chroma",
        )
        with TestClient(create_app(settings)) as client:
            search = client.post(
                "/api/v1/shopping/chat",
                json={"message": "Find laptops under ₹60,000 with at least 16 GB RAM"},
            ).json()
            session_id = search["session_id"]
            print("1. Search:", [(item["product_id"], item["price_inr"]) for item in search["products"]])

            recommendation = client.post(
                "/api/v1/shopping/recommend",
                json={
                    "customer_request": "Recommend a laptop under ₹60,000 with at least 16 GB RAM and good battery life"
                },
            ).json()
            print("2. Recommendation:", [item["product"]["product_id"] for item in recommendation["recommendations"]])

            comparison = client.post(
                "/api/v1/shopping/chat",
                json={"message": "Compare those two", "session_id": session_id},
            ).json()
            print("3. Comparison:", [item["product_id"] for item in comparison["comparison"]])

            faq = client.post(
                "/api/v1/shopping/chat",
                json={"message": "What warranty is listed for LAP-AST-014?", "session_id": session_id},
            ).json()
            print("4. FAQ:", faq["status"], [source["document_id"] for source in faq["sources"]])

            unavailable = client.post(
                "/api/v1/shopping/chat",
                json={"message": "Find tablets under ₹10,000", "session_id": session_id},
            ).json()
            print("5. Unavailable category:", unavailable["status"], unavailable["recovery_suggestions"])

            cart = client.post("/api/v1/cart").json()
            pending = client.post(
                "/api/v1/shopping/chat",
                json={
                    "message": "Add 1 unit of LAP-AST-017 to my cart",
                    "session_id": session_id,
                    "cart_id": cart["cart_id"],
                },
            ).json()
            before_confirmation = client.get(f"/api/v1/cart/{cart['cart_id']}").json()
            print(
                "6-7. Pending proposal / unchanged cart:",
                pending["status"],
                before_confirmation["items"],
                before_confirmation["total_price_inr"],
            )

            confirmation = client.post(
                "/api/v1/shopping/chat",
                json={"message": "Confirm this proposal", "session_id": session_id},
            ).json()
            final_cart = client.get(f"/api/v1/cart/{cart['cart_id']}").json()
            print(
                "8. Explicit confirmation:",
                confirmation["status"],
                final_cart["items"][0]["quantity"],
                final_cart["total_price_inr"],
            )


if __name__ == "__main__":
    main()
