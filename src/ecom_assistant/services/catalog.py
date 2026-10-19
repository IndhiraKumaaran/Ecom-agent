"""Deterministic catalog search, comparison, inventory, and budget operations."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ecom_assistant.db.models import Inventory, Product
from ecom_assistant.schemas.catalog import ProductSearchFilters
from ecom_assistant.schemas.product import ProductResponse

SPECIFICATION_FIELDS = {
    "cpu": "processor",
    "ram_gb": "RAM",
    "storage_gb": "storage capacity",
    "storage_type": "storage type",
    "battery_hours": "battery estimate",
    "gpu": "graphics",
    "display_inches": "display size",
    "display_resolution": "display resolution",
    "weight_kg": "weight",
    "operating_system": "operating system",
    "use_case": "use case",
}


@dataclass(frozen=True)
class InventoryResponseData:
    product_id: str
    stock_status: Literal["in_stock", "out_of_stock", "unknown"]
    stock_quantity: int | None
    estimated_delivery_days: int | None
    data_source: Literal["sample", "unknown"]


@dataclass(frozen=True)
class BudgetValidationData:
    product_id: str
    unit_price_inr: Decimal
    quantity: int
    total_price_inr: Decimal
    budget_inr: Decimal
    within_budget: bool
    over_budget_inr: Decimal


class ProductNotFoundError(LookupError):
    """Raised when a requested product ID is absent from the catalog."""


class MissingProductsError(LookupError):
    """Raised when one or more IDs in a comparison are not in the catalog."""

    def __init__(self, product_ids: list[str]) -> None:
        self.product_ids = product_ids
        super().__init__("One or more requested products were not found")


class CatalogService:
    """Query the existing SQLite catalog and compute deterministic totals."""

    def search_products(
        self,
        session: Session,
        filters: ProductSearchFilters,
    ) -> tuple[list[Product], int]:
        """Filter and paginate products using parameterized SQLAlchemy expressions."""
        statement = select(Product)
        count_statement = select(func.count()).select_from(Product)
        conditions = []

        if filters.query:
            escaped_query = (
                filters.query.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            pattern = f"%{escaped_query}%"
            conditions.append(
                or_(
                    Product.name.ilike(pattern, escape="\\"),
                    Product.brand.ilike(pattern, escape="\\"),
                    Product.cpu.ilike(pattern, escape="\\"),
                    Product.gpu.ilike(pattern, escape="\\"),
                    Product.description.ilike(pattern, escape="\\"),
                    Product.tags.ilike(pattern, escape="\\"),
                    Product.use_case.ilike(pattern, escape="\\"),
                )
            )
        if filters.brand:
            conditions.append(func.lower(Product.brand) == filters.brand.lower())
        if filters.category:
            conditions.append(func.lower(Product.category) == filters.category.lower())
        if filters.min_price_inr is not None:
            conditions.append(Product.price_inr >= filters.min_price_inr)
        if filters.max_price_inr is not None:
            conditions.append(Product.price_inr <= filters.max_price_inr)
        if filters.min_ram_gb is not None:
            conditions.append(Product.ram_gb >= filters.min_ram_gb)
        if filters.min_storage_gb is not None:
            conditions.append(Product.storage_gb >= filters.min_storage_gb)
        if filters.min_battery_hours is not None:
            conditions.append(Product.battery_hours >= filters.min_battery_hours)
        if filters.in_stock is True:
            conditions.append(Product.inventory.has(Inventory.stock_quantity > 0))
        elif filters.in_stock is False:
            conditions.append(Product.inventory.has(Inventory.stock_quantity == 0))

        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        if filters.sort_by == "price_asc":
            statement = statement.order_by(Product.price_inr, Product.product_id)
        elif filters.sort_by == "price_desc":
            statement = statement.order_by(Product.price_inr.desc(), Product.product_id)
        elif filters.sort_by == "battery_desc":
            statement = statement.order_by(Product.battery_hours.desc(), Product.product_id)
        elif filters.sort_by == "name":
            statement = statement.order_by(func.lower(Product.name), Product.product_id)
        elif filters.query:
            exact_name = func.lower(Product.name) == filters.query.lower()
            name_prefix = func.lower(Product.name).like(
                f"{filters.query.lower().replace('%', '\\%').replace('_', '\\_')}%",
                escape="\\",
            )
            statement = statement.order_by(exact_name.desc(), name_prefix.desc(), Product.product_id)
        else:
            statement = statement.order_by(Product.product_id)

        total = session.scalar(count_statement) or 0
        products = session.scalars(
            statement.offset(filters.offset).limit(filters.limit)
        ).all()
        return list(products), total

    def get_product(self, session: Session, product_id: str) -> Product:
        """Fetch one product or raise a domain-specific not-found error."""
        product = session.get(Product, product_id)
        if product is None:
            raise ProductNotFoundError(product_id)
        return product

    def compare_products(self, session: Session, product_ids: list[str]) -> list[Product]:
        """Return stored products in caller order and report all missing IDs."""
        products = session.scalars(
            select(Product).where(Product.product_id.in_(product_ids))
        ).all()
        by_id = {product.product_id: product for product in products}
        missing_ids = [product_id for product_id in product_ids if product_id not in by_id]
        if missing_ids:
            raise MissingProductsError(missing_ids)
        return [by_id[product_id] for product_id in product_ids]

    def get_inventory(self, session: Session, product_id: str) -> InventoryResponseData:
        """Return sample inventory, distinguishing missing inventory from zero stock."""
        product = self.get_product(session, product_id)
        inventory = product.inventory
        if inventory is None:
            return InventoryResponseData(
                product_id=product.product_id,
                stock_status="unknown",
                stock_quantity=None,
                estimated_delivery_days=None,
                data_source="unknown",
            )
        return InventoryResponseData(
            product_id=product.product_id,
            stock_status="in_stock" if inventory.stock_quantity > 0 else "out_of_stock",
            stock_quantity=inventory.stock_quantity,
            estimated_delivery_days=inventory.delivery_days,
            data_source="sample",
        )

    def validate_budget(
        self,
        session: Session,
        product_id: str,
        budget_inr: Decimal,
        quantity: int,
    ) -> BudgetValidationData:
        """Compute the full price from catalog price and quantity without discounts."""
        product = self.get_product(session, product_id)
        total = (product.price_inr * quantity).quantize(Decimal("0.01"))
        return BudgetValidationData(
            product_id=product.product_id,
            unit_price_inr=product.price_inr,
            quantity=quantity,
            total_price_inr=total,
            budget_inr=budget_inr,
            within_budget=total <= budget_inr,
            over_budget_inr=max(total - budget_inr, Decimal("0.00")),
        )

    @staticmethod
    def to_response(product: Product) -> ProductResponse:
        """Convert stored data while marking empty optional specifications."""
        missing = []
        for field, label in SPECIFICATION_FIELDS.items():
            value = getattr(product, field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(label)

        response_values = {
            field: getattr(product, field)
            for field in (
                "product_id",
                "name",
                "brand",
                "price_inr",
                "category",
                "description",
                *SPECIFICATION_FIELDS.keys(),
                "tags",
            )
        }
        for field, value in response_values.items():
            if isinstance(value, str) and not value.strip() and field not in {"product_id", "name", "brand", "category", "description"}:
                response_values[field] = None

        return ProductResponse(
            **response_values,
            specifications_complete=not missing,
            missing_specifications=missing,
        )
