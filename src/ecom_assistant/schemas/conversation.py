"""Validated conversation API contracts with operational-only workflow traces."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ecom_assistant.schemas.cart import CartProposalResponse, CartResponse
from ecom_assistant.schemas.catalog import BudgetValidationResponse, InventoryResponse
from ecom_assistant.schemas.faq import FAQAnswerResponse, FAQSourceReference
from ecom_assistant.schemas.product import ProductResponse
from ecom_assistant.schemas.shopping import ShoppingPreferenceOverrides, ShoppingRecommendationResponse

ConversationIntent = Literal[
    "product_search",
    "recommendation",
    "comparison",
    "inventory",
    "budget_validation",
    "faq",
    "cart_view",
    "cart_proposal",
    "cart_confirmation",
    "clarification",
    "unknown",
]


class ConversationRequest(BaseModel):
    """One customer message plus optional persisted context/resource references."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=2, max_length=2000)
    session_id: str | None = Field(default=None, min_length=36, max_length=36)
    cart_id: str | None = Field(default=None, min_length=36, max_length=36)
    proposal_id: str | None = Field(default=None, min_length=36, max_length=36)
    preferences: ShoppingPreferenceOverrides | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value

    @field_validator("session_id", "cart_id", "proposal_id")
    @classmethod
    def optional_ids_must_be_uuids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        import re

        if not re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            value,
        ):
            raise ValueError("IDs must be UUIDs")
        return value


class ConversationTraceEvent(BaseModel):
    """An operational event; never contains prompts or private reasoning."""

    event: str
    status: Literal["success", "warning", "failed", "skipped"]
    detail: str = Field(max_length=240)


class ConversationResponse(BaseModel):
    """Structured response for one assistant turn and its delegated capability."""

    session_id: str
    intent: ConversationIntent
    status: Literal[
        "success",
        "clarification_required",
        "no_results",
        "insufficient_evidence",
        "proposal_pending",
        "proposal_confirmed",
        "proposal_expired",
        "proposal_stale",
        "proposal_rejected",
        "error",
    ]
    answer: str
    products: list[ProductResponse] = Field(default_factory=list)
    comparison: list[ProductResponse] = Field(default_factory=list)
    recommendation: ShoppingRecommendationResponse | None = None
    inventory: list[InventoryResponse] = Field(default_factory=list)
    budget: BudgetValidationResponse | None = None
    faq: FAQAnswerResponse | None = None
    sources: list[FAQSourceReference] = Field(default_factory=list)
    cart: CartResponse | None = None
    proposal: CartProposalResponse | None = None
    clarification_questions: list[str] = Field(default_factory=list)
    recovery_suggestions: list[str] = Field(default_factory=list)
    trace: list[ConversationTraceEvent] = Field(default_factory=list)
    execution_metadata: dict[str, Any] = Field(default_factory=dict)