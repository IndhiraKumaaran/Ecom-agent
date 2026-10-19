"""Catalog search, comparison, inventory, and budget endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from sqlalchemy.orm import Session

from ecom_assistant.api.dependencies import get_catalog_service, get_db_session
from ecom_assistant.schemas.catalog import (
    BudgetValidationRequest,
    BudgetValidationResponse,
    InventoryResponse,
    ProductComparisonRequest,
    ProductComparisonResponse,
    ProductSearchFilters,
    ProductSearchResponse,
)
from ecom_assistant.schemas.product import ProductResponse
from ecom_assistant.services.catalog import (
    CatalogService,
    MissingProductsError,
    ProductNotFoundError,
)

router = APIRouter(tags=["products", "inventory", "budget"])
PRODUCT_ID_PATTERN = r"^[A-Za-z0-9-]{1,32}$"


@router.get("/products", response_model=ProductSearchResponse)
def list_products(
    filters: Annotated[ProductSearchFilters, Query()],
    service: Annotated[CatalogService, Depends(get_catalog_service)],
    session: Annotated[Session, Depends(get_db_session)],
) -> ProductSearchResponse:
    """Search and filter stored product records; an empty match returns an empty list."""
    products, total = service.search_products(session, filters)
    return ProductSearchResponse(
        products=[service.to_response(product) for product in products],
        total=total,
        limit=filters.limit,
        offset=filters.offset,
    )


@router.get("/products/{product_id}", response_model=ProductResponse)
def get_product(
    product_id: Annotated[str, Path(min_length=1, max_length=32, pattern=PRODUCT_ID_PATTERN)],
    service: Annotated[CatalogService, Depends(get_catalog_service)],
    session: Annotated[Session, Depends(get_db_session)],
) -> ProductResponse:
    """Return one stored product and identify incomplete specifications."""
    try:
        product = service.get_product(session, product_id)
    except ProductNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found") from exc
    return service.to_response(product)


@router.post("/products/compare", response_model=ProductComparisonResponse)
def compare_products(
    payload: ProductComparisonRequest,
    service: Annotated[CatalogService, Depends(get_catalog_service)],
    session: Annotated[Session, Depends(get_db_session)],
) -> ProductComparisonResponse:
    """Compare two or more actual catalog IDs in the caller's requested order."""
    try:
        products = service.compare_products(session, payload.product_ids)
    except MissingProductsError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "One or more products were not found", "product_ids": exc.product_ids},
        ) from exc
    return ProductComparisonResponse(
        products=[service.to_response(product) for product in products]
    )


@router.get("/inventory/{product_id}", response_model=InventoryResponse)
def get_inventory(
    product_id: Annotated[str, Path(min_length=1, max_length=32, pattern=PRODUCT_ID_PATTERN)],
    service: Annotated[CatalogService, Depends(get_catalog_service)],
    session: Annotated[Session, Depends(get_db_session)],
) -> InventoryResponse:
    """Return explicitly labeled sample stock and estimated delivery data."""
    try:
        inventory = service.get_inventory(session, product_id)
    except ProductNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found") from exc
    return InventoryResponse(**inventory.__dict__)


@router.post("/budget/validate", response_model=BudgetValidationResponse)
def validate_budget(
    payload: BudgetValidationRequest,
    service: Annotated[CatalogService, Depends(get_catalog_service)],
    session: Annotated[Session, Depends(get_db_session)],
) -> BudgetValidationResponse:
    """Calculate the catalog price times quantity and compare it to the budget."""
    try:
        result = service.validate_budget(
            session,
            payload.product_id,
            payload.budget_inr,
            payload.quantity,
        )
    except ProductNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found") from exc
    return BudgetValidationResponse(**result.__dict__)