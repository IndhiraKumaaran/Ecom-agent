"""Unified conversational shopping-assistant endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from ecom_assistant.api.dependencies import get_conversation_workflow
from ecom_assistant.schemas.conversation import ConversationRequest, ConversationResponse
from ecom_assistant.workflow.conversation import ConversationWorkflow

router = APIRouter(prefix="/shopping", tags=["shopping-chat"])


@router.post("/chat", response_model=ConversationResponse)
def shopping_chat(
    payload: ConversationRequest,
    workflow: Annotated[ConversationWorkflow, Depends(get_conversation_workflow)],
) -> ConversationResponse | JSONResponse:
    """Route one customer message to an existing shopping capability."""
    try:
        result = workflow.invoke(payload)
        if result.status == "error":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=result.model_dump(mode="json"),
            )
        return result
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shopping services are temporarily unavailable.",
        ) from exc