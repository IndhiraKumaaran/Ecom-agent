"""Request and response contracts for deterministic catalog operations."""

import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ecom_assistant.schemas.product import ProductResponse

PRODUCT_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,32}$")


class ProductSearchFilters(BaseModel):
    """Validated filters accepted by the product listing endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(default=None, max_length=120)
    brand: str | None = Field(default=None, max_length=80)
    category: str | None = Field(default=None, max_length=60)
    min_price_inr: Decimal | None = Field(default=None, gt=0, max_digits=10, decimal_places=2)
    max_price_inr: Decimal | None = Field(default=None, gt=0, max_digits=10, decimal_places=2)
    min_ram_gb: int | None = Field(default=None, ge=1, le=256)
    min_storage_gb: int | None = Field(default=None, ge=1, le=8192)
    min_battery_hours: Decimal | None = Field(default=None, gt=0, max_digits=4, decimal_places=1)
    in_stock: bool | None = None
    sort_by: Literal["relevance", "price_asc", "price_desc", "battery_desc", "name"] = "relevance"
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @field_validator("query", "brand", "category")
    @classmethod
    def text_filters_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("text filters must not be blank")
        return value

    @model_validator(mode="after")
    def price_range_must_be_ordered(self) -> "ProductSearchFilters":
        if (
            self.min_price_inr is not None
            and self.max_price_inr is not None
            and self.min_price_inr > self.max_price_inr
        ):
            raise ValueError("min_price_inr must not exceed max_price_inr")
        return self


class ProductSearchResponse(BaseModel):
    """Paginated product results with an explicit sample-data label."""

    products: list[ProductResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    sample_data: bool = True


class ProductComparisonRequest(BaseModel):
    """Two to eight unique catalog product IDs to compare."""

    model_config = ConfigDict(extra="forbid")

    product_ids: list[str] = Field(min_length=2, max_length=8)

    @field_validator("product_ids")
    @classmethod
    def product_ids_must_be_valid_and_unique(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not PRODUCT_ID_PATTERN.fullmatch(value) for value in normalized):
            raise ValueError("product_ids must contain valid catalog ID formats")
        if len(normalized) != len(set(normalized)):
            raise ValueError("product_ids must be unique")
        return normalized


class ProductComparisonResponse(BaseModel):
    """Comparison rows in the same order as the request."""

    products: list[ProductResponse]
    sample_data: bool = True


class InventoryResponse(BaseModel):
    """Sample inventory and delivery details for one catalog product."""

    product_id: str
    stock_status: Literal["in_stock", "out_of_stock", "unknown"]
    stock_quantity: int | None = Field(default=None, ge=0)
    estimated_delivery_days: int | None = Field(default=None, ge=0)
    data_source: Literal["sample", "unknown"]
    sample_data: bool = True


class BudgetValidationRequest(BaseModel):
    """Product and quantity to validate against a customer-provided budget."""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, max_length=32)
    budget_inr: Decimal = Field(gt=0, max_digits=10, decimal_places=2)
    quantity: int = Field(default=1, ge=1, le=50)

    @field_validator("product_id")
    @classmethod
    def product_id_must_be_valid(cls, value: str) -> str:
        value = value.strip()
        if not PRODUCT_ID_PATTERN.fullmatch(value):
            raise ValueError("product_id must use letters, numbers, or hyphens")
        return value


class BudgetValidationResponse(BaseModel):
    """Server-calculated product total and budget outcome; no discounts applied."""

    product_id: str
    unit_price_inr: Decimal
    quantity: int = Field(ge=1)
    total_price_inr: Decimal
    budget_inr: Decimal
    within_budget: bool
    over_budget_inr: Decimal = Field(ge=0)
    sample_data: bool = True