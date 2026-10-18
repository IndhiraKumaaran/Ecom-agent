"""Validated product response schema for later catalog endpoints."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class ProductResponse(BaseModel):
    """Public, non-secret product catalog fields."""

    model_config = ConfigDict(from_attributes=True)

    product_id: str
    name: str
    brand: str
    price_inr: Decimal
    cpu: str
    ram_gb: int
    storage_gb: int
    battery_hours: Decimal
    category: str
    description: str