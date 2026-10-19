"""Validated product response schema for catalog endpoints."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class ProductResponse(BaseModel):
    """Public product details and completeness information."""

    model_config = ConfigDict(from_attributes=True)

    product_id: str
    name: str
    brand: str
    price_inr: Decimal
    category: str
    description: str
    cpu: str | None
    ram_gb: int | None
    storage_gb: int | None
    storage_type: str | None
    battery_hours: Decimal | None
    gpu: str | None
    display_inches: Decimal | None
    display_resolution: str | None
    weight_kg: Decimal | None
    operating_system: str | None
    use_case: str | None
    tags: str | None
    specifications_complete: bool
    missing_specifications: list[str] = Field(default_factory=list)
    sample_data: bool = True