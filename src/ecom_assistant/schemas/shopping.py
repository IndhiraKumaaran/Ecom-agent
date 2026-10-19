"""Validated contracts for shopping requirement extraction and recommendations."""

from decimal import Decimal
from typing import Literal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ecom_assistant.schemas.catalog import BudgetValidationResponse
from ecom_assistant.schemas.product import ProductResponse
from ecom_assistant.schemas.catalog import InventoryResponse


class ShoppingRequirements(BaseModel):
    """Only customer-stated shopping requirements; omitted constraints stay unset."""

    model_config = ConfigDict(extra="forbid")

    category: str | None = Field(default=None, max_length=60)
    max_budget_inr: Decimal | None = Field(default=None, gt=0, max_digits=10, decimal_places=2)
    min_ram_gb: int | None = Field(default=None, ge=1, le=256)
    min_storage_gb: int | None = Field(default=None, ge=1, le=8192)
    preferred_cpu: str | None = Field(default=None, max_length=120)
    min_battery_hours: Decimal | None = Field(default=None, gt=0, max_digits=4, decimal_places=1)
    in_stock_only: bool | None = None
    other_preferences: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("category", "preferred_cpu")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("requirement values must not be blank")
        return value

    @field_validator("other_preferences")
    @classmethod
    def normalize_preferences(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("other_preferences must not contain blank values")
        return list(dict.fromkeys(normalized))


class ShoppingPreferenceOverrides(ShoppingRequirements):
    """Optional, explicitly structured customer preferences that override extraction."""


class ShoppingRecommendationRequest(BaseModel):
    """Natural-language request plus optional validated preference overrides."""

    model_config = ConfigDict(extra="forbid")

    customer_request: str = Field(min_length=5, max_length=2000)
    preferences: ShoppingPreferenceOverrides | None = None

    @field_validator("customer_request")
    @classmethod
    def request_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("customer_request must not be blank")
        return value


class WorkflowStageTrace(BaseModel):
    """Non-secret workflow diagnostic for one executed graph stage."""

    stage: str
    status: Literal["success", "warning", "failed", "skipped"]
    detail: str


class RankedRecommendation(BaseModel):
    """A strict match ranked using deterministic, inspectable factors."""

    product: ProductResponse
    rank: int = Field(ge=1)
    score: int = Field(ge=0, le=100)
    strict_match: Literal[True] = True
    ranking_factors: dict[str, int]
    reasons: list[str]
    trade_offs: list[str]


class RelaxedAlternative(BaseModel):
    """A real catalog product that misses at least one explicit hard constraint."""

    product: ProductResponse
    strict_match: Literal[False] = False
    ranking_factors: dict[str, int]
    relaxed_constraints: list[str]
    explanation: str


class ShoppingRecommendationResponse(BaseModel):
    """Recommendation result and execution trace; no cart mutation is performed."""

    status: Literal[
        "success",
        "needs_clarification",
        "no_results",
        "no_strict_matches",
        "failed",
        "partial_failure",
    ]
    customer_request: str
    extraction_mode: Literal["mock", "openai"] | None
    extracted_requirements: ShoppingRequirements | None
    candidate_product_ids: list[str]
    matching_products: list[ProductResponse]
    comparison_results: list[ProductResponse]
    inventory_results: list[InventoryResponse]
    budget_validation: list[BudgetValidationResponse]
    recommendations: list[RankedRecommendation]
    alternatives: list[RelaxedAlternative]
    warnings: list[str]
    errors: list[str]
    trace: list[WorkflowStageTrace]
    execution_metadata: dict[str, Any]