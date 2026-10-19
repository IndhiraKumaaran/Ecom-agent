"""Typed, read-only LangChain tools backed by the deterministic catalog service."""

import re
from decimal import Decimal
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ecom_assistant.db.session import Database
from ecom_assistant.schemas.catalog import ProductSearchFilters
from ecom_assistant.services.catalog import (
    BudgetValidationData,
    CatalogService,
    MissingProductsError,
    ProductNotFoundError,
)

PRODUCT_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,32}$")


class SearchProductsInput(BaseModel):
    """Validated filters for the catalog search tool."""

    model_config = ConfigDict(extra="forbid")

    filters: ProductSearchFilters


class CheckInventoryInput(BaseModel):
    """Validated catalog product identifier for inventory lookup."""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, max_length=32)

    @field_validator("product_id")
    @classmethod
    def validate_product_id(cls, value: str) -> str:
        value = value.strip()
        if not PRODUCT_ID_RE.fullmatch(value):
            raise ValueError("product_id has an invalid format")
        return value


class CompareProductsInput(BaseModel):
    """Validated IDs for internal comparison of a bounded search result set."""

    model_config = ConfigDict(extra="forbid")

    product_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("product_ids")
    @classmethod
    def validate_product_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not PRODUCT_ID_RE.fullmatch(value) for value in normalized):
            raise ValueError("product_ids contains an invalid ID format")
        if len(normalized) != len(set(normalized)):
            raise ValueError("product_ids must be unique")
        return normalized


class ValidateBudgetInput(CheckInventoryInput):
    """Validated product, customer budget, and quantity for budget calculation."""

    budget_inr: Decimal = Field(gt=0, max_digits=10, decimal_places=2)
    quantity: int = Field(default=1, ge=1, le=50)


def create_shopping_tools(
    database: Database,
    catalog_service: CatalogService,
) -> dict[str, StructuredTool]:
    """Create tools whose only authority is read access to the catalog services."""

    def search_products(filters: ProductSearchFilters) -> dict[str, Any]:
        with database.session() as session:
            products, total = catalog_service.search_products(session, filters)
            return {
                "products": [
                    catalog_service.to_response(product).model_dump(mode="json")
                    for product in products
                ],
                "total": total,
                "sample_data": True,
            }

    def compare_products(product_ids: list[str]) -> dict[str, Any]:
        try:
            with database.session() as session:
                products = catalog_service.compare_products(session, product_ids)
                return {
                    "products": [
                        catalog_service.to_response(product).model_dump(mode="json")
                        for product in products
                    ],
                    "sample_data": True,
                }
        except MissingProductsError as exc:
            return {"error": "missing_products", "product_ids": exc.product_ids}

    def check_inventory(product_id: str) -> dict[str, Any]:
        try:
            with database.session() as session:
                return catalog_service.get_inventory(session, product_id).__dict__
        except ProductNotFoundError:
            return {"error": "product_not_found", "product_id": product_id}

    def validate_budget(
        product_id: str,
        budget_inr: Decimal,
        quantity: int = 1,
    ) -> dict[str, Any]:
        try:
            with database.session() as session:
                result: BudgetValidationData = catalog_service.validate_budget(
                    session,
                    product_id,
                    budget_inr,
                    quantity,
                )
                return {
                    key: str(value) if isinstance(value, Decimal) else value
                    for key, value in result.__dict__.items()
                }
        except ProductNotFoundError:
            return {"error": "product_not_found", "product_id": product_id}

    return {
        "search": StructuredTool.from_function(
            func=search_products,
            name="SearchProductsTool",
            description="Search real SQLite catalog products using validated filters. Read-only.",
            args_schema=SearchProductsInput,
        ),
        "compare": StructuredTool.from_function(
            func=compare_products,
            name="CompareProductsTool",
            description="Compare one or more actual catalog product IDs. Read-only.",
            args_schema=CompareProductsInput,
        ),
        "inventory": StructuredTool.from_function(
            func=check_inventory,
            name="CheckInventoryTool",
            description="Read sample stock and delivery values for an actual product ID. Read-only.",
            args_schema=CheckInventoryInput,
        ),
        "budget": StructuredTool.from_function(
            func=validate_budget,
            name="ValidateBudgetTool",
            description="Compute actual catalog price times quantity against a budget. Read-only.",
            args_schema=ValidateBudgetInput,
        ),
    }