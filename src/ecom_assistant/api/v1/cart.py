"""Safe simulated-cart API with persisted explicit mutation proposals."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ecom_assistant.api.dependencies import get_cart_service, get_db_session
from ecom_assistant.schemas.cart import (
    CartConfirmRequest,
    CartConfirmationResponse,
    CartProposalRequest,
    CartProposalResponse,
    CartResponse,
    UUID_PATTERN,
)
from ecom_assistant.services.cart import (
    CartMutationRejectedError,
    CartNotFoundError,
    CartProposalNotFoundError,
    CartService,
)
from ecom_assistant.services.catalog import ProductNotFoundError

router = APIRouter(prefix="/cart", tags=["simulated-cart"])


@router.post("", response_model=CartResponse, status_code=status.HTTP_201_CREATED)
def create_cart(
    service: Annotated[CartService, Depends(get_cart_service)],
    session: Annotated[Session, Depends(get_db_session)],
) -> CartResponse:
    """Create an empty simulated cart; no order or reservation is created."""
    return service.create_cart(session)


@router.get("/{cart_id}", response_model=CartResponse)
def get_cart(
    cart_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    service: Annotated[CartService, Depends(get_cart_service)],
    session: Annotated[Session, Depends(get_db_session)],
) -> CartResponse:
    """Retrieve simulated cart contents and totals."""
    try:
        return service.get_cart(session, cart_id)
    except CartNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Simulated cart not found") from exc


@router.post(
    "/{cart_id}/proposals",
    response_model=CartProposalResponse,
    status_code=status.HTTP_201_CREATED,
)
def propose_cart_mutation(
    cart_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    payload: CartProposalRequest,
    service: Annotated[CartService, Depends(get_cart_service)],
    session: Annotated[Session, Depends(get_db_session)],
) -> CartProposalResponse:
    """Validate and persist a pending proposal without changing cart items."""
    try:
        return service.propose(session, cart_id, payload)
    except CartNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Simulated cart not found") from exc
    except ProductNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Product not found in catalog") from exc
    except CartMutationRejectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/{cart_id}/confirmations/{proposal_id}",
    response_model=CartConfirmationResponse,
    responses={
        status.HTTP_409_CONFLICT: {"model": CartConfirmationResponse},
        status.HTTP_410_GONE: {"model": CartConfirmationResponse},
    },
)
def confirm_cart_proposal(
    cart_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    proposal_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    payload: CartConfirmRequest,
    service: Annotated[CartService, Depends(get_cart_service)],
    session: Annotated[Session, Depends(get_db_session)],
) -> CartConfirmationResponse | JSONResponse:
    """Explicitly confirm the exact proposal ID in this request path."""
    try:
        result = service.confirm(session, cart_id, proposal_id)
    except CartNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Simulated cart not found") from exc
    except CartProposalNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Cart proposal not found") from exc

    if result.status == "expired":
        return JSONResponse(status_code=status.HTTP_410_GONE, content=result.model_dump(mode="json"))
    if result.status in {"stale", "rejected"}:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=result.model_dump(mode="json"))
    return result