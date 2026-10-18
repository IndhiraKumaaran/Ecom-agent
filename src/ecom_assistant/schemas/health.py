"""Health API response contract."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Readiness details for the API and its configured inference mode."""

    status: Literal["ok"]
    mode: Literal["mock", "openai"]