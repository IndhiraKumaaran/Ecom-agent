"""Offline FAQ retrieval, metadata, and safe re-ingestion tests."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from ecom_assistant.config import Settings
from ecom_assistant.db.models import FAQDocument
from ecom_assistant.rag.service import FAQServiceError
from ecom_assistant.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.csv"
FAQ_SOURCE_PATH = PROJECT_ROOT / "data" / "faq_documents.json"


@pytest.fixture
def faq_client(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'faq-test.db'}",
        catalog_seed_path=CATALOG_PATH,
        faq_source_path=FAQ_SOURCE_PATH,
        faq_persist_directory=tmp_path / "chroma",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


def test_faq_answers_from_catalog_sources(faq_client) -> None:
    client, _app = faq_client

    response = client.post(
        "/api/v1/faq/ask",
        json={"question": "What processor is in this laptop?", "product_id": "LAP-AST-014"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "grounded"
    assert "AMD Ryzen 5 7530U" in body["answer"]
    assert any(source["source"] == "catalog.csv" for source in body["sources"])
    assert any(snippet["product_id"] == "LAP-AST-014" for snippet in body["snippets"])


def test_product_filter_keeps_warranty_evidence_product_specific(faq_client) -> None:
    client, _app = faq_client

    aster = client.post(
        "/api/v1/faq/ask",
        json={"question": "What warranty is listed for this laptop?", "product_id": "LAP-AST-014"},
    ).json()
    nivara = client.post(
        "/api/v1/faq/ask",
        json={"question": "What warranty is listed for this laptop?", "product_id": "LAP-NVR-015"},
    ).json()

    assert aster["status"] == nivara["status"] == "grounded"
    assert "12 months" in aster["answer"]
    assert "24 months" in nivara["answer"]
    assert all(source["product_id"] == "LAP-AST-014" for source in aster["sources"])
    assert all(source["product_id"] == "LAP-NVR-015" for source in nivara["sources"])


@pytest.mark.parametrize(
    "question",
    [
        "How do I change the wallpaper on my computer?",
        "What is the return policy for this product?",
    ],
)
def test_irrelevant_or_unsupplied_policy_questions_decline(faq_client, question: str) -> None:
    client, _app = faq_client

    response = client.post("/api/v1/faq/ask", json={"question": question})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer"] == "The catalog does not contain enough information to answer that question."
    assert body["sources"] == []
    assert body["snippets"] == []


def test_known_spec_does_not_support_an_unlisted_feature(faq_client) -> None:
    client, _app = faq_client

    response = client.post(
        "/api/v1/faq/ask",
        json={
            "question": "Can the RAM be upgraded?",
            "product_id": "LAP-AST-014",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"


def test_safe_reingestion_updates_changed_documents_and_removes_stale_sources(
    faq_client, tmp_path: Path
) -> None:
    _client, app = faq_client
    source_copy = tmp_path / "faq_documents.json"
    records = json.loads(FAQ_SOURCE_PATH.read_text(encoding="utf-8"))
    source_copy.write_text(json.dumps(records), encoding="utf-8")

    with app.state.database.session() as session, session.begin():
        report = app.state.faq_service.ingest(session, source_copy)
    assert report.indexed_documents == 0
    assert report.deleted_documents == 0

    changed_records = [record for record in records if record["faq_id"] != "catalog-currency-sample"]
    changed_records[0]["answer"] = (
        "Fictional sample FAQ only: the AsterBook Air 14 is listed with 15 months "
        "of limited hardware coverage. This is invented demonstration content."
    )
    source_copy.write_text(json.dumps(changed_records), encoding="utf-8")
    with app.state.database.session() as session, session.begin():
        updated_report = app.state.faq_service.ingest(session, source_copy)

    assert updated_report.updated_documents == 1
    assert updated_report.deleted_documents == 1
    assert app.state.faq_service.collection.count() == 8 + len(changed_records)
    with app.state.database.session() as session:
        seeded_rows = session.scalars(
            select(FAQDocument).where(FAQDocument.document_id.like("seed-faq:%"))
        ).all()
    assert len(seeded_rows) == len(changed_records)


def test_retrieval_error_returns_safe_service_unavailable(faq_client, monkeypatch) -> None:
    client, app = faq_client

    def fail_retrieval(*_args, **_kwargs):
        raise FAQServiceError("simulated Chroma outage")

    monkeypatch.setattr(app.state.faq_service, "_retrieve", fail_retrieval)
    response = client.post("/api/v1/faq/ask", json={"question": "What currency are prices in?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "FAQ retrieval is temporarily unavailable."}


def test_unknown_product_filter_returns_not_found(faq_client) -> None:
    client, _app = faq_client

    response = client.post(
        "/api/v1/faq/ask",
        json={"question": "What is the battery estimate?", "product_id": "NOT-A-PRODUCT"},
    )

    assert response.status_code == 404