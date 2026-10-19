"""Shopping recommendation workflow endpoint."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from ecom_assistant.schemas.shopping import (
    ShoppingRecommendationRequest,
    ShoppingRecommendationResponse,
)

router = APIRouter(prefix="/shopping", tags=["shopping"])


@router.post(
    "/recommend",
    response_model=ShoppingRecommendationResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ShoppingRecommendationResponse}},
)
def recommend(
    payload: ShoppingRecommendationRequest,
    request: Request,
) -> ShoppingRecommendationResponse | JSONResponse:
    """Run the compiled catalog workflow without changing a cart."""
    result: ShoppingRecommendationResponse = request.app.state.shopping_workflow.invoke(payload)
    if result.status in {"failed", "partial_failure"}:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=result.model_dump(mode="json"),
        )
    return result