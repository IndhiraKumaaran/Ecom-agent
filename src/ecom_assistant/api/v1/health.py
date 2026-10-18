"""Version 1 health and readiness endpoint."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from ecom_assistant.api.dependencies import get_db_session
from ecom_assistant.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request, session: Session = Depends(get_db_session)) -> HealthResponse:
    """Check database connectivity and report the configured LLM mode."""
    session.execute(text("SELECT 1"))
    return HealthResponse(status="ok", mode=request.app.state.settings.llm_mode)