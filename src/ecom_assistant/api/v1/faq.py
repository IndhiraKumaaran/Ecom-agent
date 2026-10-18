"""Product FAQ question endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ecom_assistant.api.dependencies import get_db_session
from ecom_assistant.rag.service import FAQServiceError, UnknownProductError
from ecom_assistant.schemas.faq import FAQAnswerResponse, FAQAskRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/faq", tags=["faq"])


@router.post("/ask", response_model=FAQAnswerResponse)
def ask_faq(
    payload: FAQAskRequest,
    request: Request,
    session: Session = Depends(get_db_session),
) -> FAQAnswerResponse:
    """Answer from retrieved catalog evidence, without changing cart state."""
    try:
        return request.app.state.faq_service.ask(session, payload)
    except UnknownProductError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested product_id is not present in the catalog.",
        ) from exc
    except FAQServiceError as exc:
        logger.warning("FAQ service unavailable: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FAQ retrieval is temporarily unavailable.",
        ) from exc