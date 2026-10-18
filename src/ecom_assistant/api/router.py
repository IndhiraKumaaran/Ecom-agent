"""Top-level API router assembly."""

from fastapi import APIRouter

from ecom_assistant.api.v1.faq import router as faq_router
from ecom_assistant.api.v1.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/api/v1")
api_router.include_router(faq_router, prefix="/api/v1")