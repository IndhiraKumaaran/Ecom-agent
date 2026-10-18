"""Validated loading and catalog-to-document conversion for FAQ ingestion."""

import json
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from ecom_assistant.db.models import FAQDocument, Product
from ecom_assistant.rag.embeddings import content_terms

FAQ_ROW_PREFIX = "seed-faq:"
MANAGED_BY = "ecom-assistant-faq"


class SourceFAQ(BaseModel):
    """One supplied, catalog-grounded fictional FAQ source record."""

    model_config = ConfigDict(extra="forbid")

    faq_id: str = Field(min_length=1, max_length=48, pattern=r"^[a-zA-Z0-9_-]+$")
    product_id: str | None = Field(default=None, max_length=32)
    document_type: Literal["warranty", "product_faq", "general_faq"]
    question: str = Field(min_length=5, max_length=500)
    answer: str = Field(min_length=5, max_length=3000)
    source: str = Field(default="fictional_faq_documents.json", min_length=1, max_length=120)

    @field_validator("question", "answer", "source")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("FAQ source text must not be blank")
        return value.strip()


def load_source_faqs(source_path: Path) -> list[SourceFAQ]:
    """Load and validate the supplied JSON FAQ source file."""
    with Path(source_path).open(encoding="utf-8") as source_file:
        raw_records = json.load(source_file)
    if not isinstance(raw_records, list):
        raise ValueError("FAQ source JSON must contain a list of documents")
    records = [SourceFAQ.model_validate(record) for record in raw_records]
    ids = [record.faq_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("FAQ source contains duplicate faq_id values")
    return records


def sync_faq_rows(session: Session, records: list[SourceFAQ]) -> list[FAQDocument]:
    """Upsert source FAQ rows and delete previously seeded rows no longer supplied."""
    products = set(session.scalars(select(Product.product_id)).all())
    for record in records:
        if record.product_id is not None and record.product_id not in products:
            raise ValueError(f"FAQ source references unknown product_id: {record.product_id}")

    desired_ids = {f"{FAQ_ROW_PREFIX}{record.faq_id}" for record in records}
    existing_rows = session.scalars(
        select(FAQDocument).where(FAQDocument.document_id.like(f"{FAQ_ROW_PREFIX}%"))
    ).all()
    rows_by_id = {row.document_id: row for row in existing_rows}

    for row_id, existing in rows_by_id.items():
        if row_id not in desired_ids:
            session.delete(existing)

    rows: list[FAQDocument] = []
    for record in records:
        row_id = f"{FAQ_ROW_PREFIX}{record.faq_id}"
        row = rows_by_id.get(row_id)
        if row is None:
            row = FAQDocument(document_id=row_id)
            session.add(row)
        row.product_id = record.product_id
        row.document_type = record.document_type
        row.question = record.question
        row.answer = record.answer
        row.source = record.source
        rows.append(row)

    session.flush()
    return rows


def catalog_documents(products: list[Product]) -> list[Document]:
    """Create one factual product-specification document per catalog product."""
    documents: list[Document] = []
    for product in products:
        content = "\n".join(
            (
                f"Product: {product.name} (ID: {product.product_id})",
                f"Brand: {product.brand}",
                f"Category: {product.category}",
                f"Price: INR {product.price_inr}",
                f"Processor: {product.cpu}",
                f"Memory: {product.ram_gb} GB RAM",
                f"Storage: {product.storage_gb} GB {product.storage_type}",
                f"Graphics: {product.gpu}",
                f"Display: {product.display_inches}-inch {product.display_resolution}",
                f"Estimated battery: {product.battery_hours} hours",
                f"Weight: {product.weight_kg} kg",
                f"Operating system: {product.operating_system}",
                f"Use case: {product.use_case}",
                f"Description: {product.description}",
                f"Search tags: {product.tags}",
            )
        )
        documents.append(
            Document(
                id=f"catalog:{product.product_id}:specs",
                page_content=content,
                metadata={
                    "document_id": f"catalog:{product.product_id}:specs",
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "document_type": "product_specification",
                    "source": "catalog.csv",
                    "managed_by": MANAGED_BY,
                },
            )
        )
    return documents


def faq_documents(records: list[FAQDocument], products: dict[str, Product]) -> list[Document]:
    """Create source-preserving FAQ and warranty documents from validated rows."""
    documents: list[Document] = []
    for record in records:
        product = products.get(record.product_id) if record.product_id else None
        product_prefix = (
            f"Product: {product.name} (ID: {product.product_id})\n" if product else ""
        )
        content = f"{product_prefix}Question: {record.question}\nAnswer: {record.answer}"
        documents.append(
            Document(
                id=f"faq:{record.document_id.removeprefix(FAQ_ROW_PREFIX)}",
                page_content=content,
                metadata={
                    "document_id": record.document_id,
                    "product_id": record.product_id or "__general__",
                    "product_name": product.name if product else "",
                    "document_type": record.document_type,
                    "source": record.source,
                    "managed_by": MANAGED_BY,
                },
            )
        )
    return documents


def document_hash(document: Document) -> str:
    """Return a stable fingerprint for document text and its source metadata."""
    from hashlib import sha256

    canonical = json.dumps(
        {"content": document.page_content, "metadata": document.metadata},
        sort_keys=True,
        ensure_ascii=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def evidence_terms(question: str, metadata: dict[str, str]) -> set[str]:
    """Exclude product identity from a question before checking factual overlap."""
    terms = content_terms(question)
    terms -= content_terms(metadata.get("product_id", ""))
    terms -= content_terms(metadata.get("product_name", ""))
    return terms