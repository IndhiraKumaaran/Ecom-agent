"""Validated request and response schemas for simulated cart operations."""

import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
PRODUCT_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,32}$")


class CartItemResponse(BaseModel):
    """One persisted cart item and server-calculated line total."""

    product_id: str
    name: str
    quantity: int = Field(ge=1, le=50)
    unit_price_inr: Decimal
    line_total_inr: Decimal
    sample_data: bool = True


class CartResponse(BaseModel):
    """Simulated cart contents and a server-calculated total."""

    cart_id: str
    status: Literal["open", "confirmed"]
    items: list[CartItemResponse]
    total_price_inr: Decimal
    simulated: Literal[True] = True
    sample_data: Literal[True] = True


class CartProposalRequest(BaseModel):
    """Propose exactly one add, update, or remove mutation."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["add", "update", "remove"]
    product_id: str = Field(min_length=1, max_length=32)
    quantity: int | None = Field(default=None, strict=True, ge=1, le=50)

    @field_validator("product_id")
    @classmethod
    def validate_product_id(cls, value: str) -> str:
        value = value.strip()
        if not PRODUCT_ID_PATTERN.fullmatch(value):
            raise ValueError("product_id has an invalid format")
        return value

    @model_validator(mode="after")
    def quantity_must_match_operation(self) -> "CartProposalRequest":
        if self.operation in {"add", "update"} and self.quantity is None:
            raise ValueError("quantity is required for add and update operations")
        if self.operation == "remove" and self.quantity is not None:
            raise ValueError("quantity must be omitted for remove operations")
        return self


class CartProposalResponse(BaseModel):
    """Persisted proposal details exactly as the client must explicitly confirm."""

    proposal_id: str
    cart_id: str
    operation: Literal["add", "update", "remove"]
    product_id: str
    quantity: int | None
    resulting_quantity: int = Field(ge=0, le=50)
    unit_price_inr: Decimal
    projected_total_inr: Decimal
    status: Literal["pending", "confirmed", "expired", "stale", "rejected"]
    expires_at: str
    simulated: Literal[True] = True


class CartConfirmRequest(BaseModel):
    """Explicit affirmative customer confirmation of the path's proposal ID."""

    model_config = ConfigDict(extra="forbid")

    confirm: StrictBool

    @field_validator("confirm")
    @classmethod
    def confirmation_must_be_affirmative(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("confirm must be the boolean true")
        return value


class CartConfirmationResponse(BaseModel):
    """Outcome of applying an exact pending proposal, or why it was not applied."""

    status: Literal[
        "confirmed",
        "already_confirmed",
        "expired",
        "stale",
        "rejected",
    ]
    proposal: CartProposalResponse
    cart: CartResponse | None
    message: str
    simulated: Literal[True] = True