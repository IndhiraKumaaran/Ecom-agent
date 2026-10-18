"""Validated and idempotent CSV import for the initial product catalog."""

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy.orm import Session

from ecom_assistant.db.models import Inventory, Product

REQUIRED_COLUMNS = {
    "product_id",
    "name",
    "brand",
    "price_inr",
    "cpu",
    "ram_gb",
    "storage_gb",
    "storage_type",
    "battery_hours",
    "stock_quantity",
    "category",
    "description",
    "gpu",
    "display_inches",
    "display_resolution",
    "weight_kg",
    "operating_system",
    "use_case",
    "delivery_days",
    "tags",
}


def _decimal(row: dict[str, str], column: str) -> Decimal:
    try:
        return Decimal(row[column])
    except (InvalidOperation, KeyError) as exc:
        raise ValueError(f"Invalid decimal value for {column!r}") from exc


def _integer(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid integer value for {column!r}") from exc


def import_catalog_csv(session: Session, csv_path: Path) -> int:
    """Insert or update catalog rows and corresponding sample inventory.

    The caller owns the transaction, allowing startup seeding to be atomic.
    Re-importing a product updates that product rather than duplicating it.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog seed file does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(f"Catalog CSV is missing columns: {', '.join(sorted(missing_columns))}")

        seen_ids: set[str] = set()
        count = 0
        for row in reader:
            product_id = row["product_id"].strip()
            if not product_id:
                raise ValueError("Catalog product_id must not be blank")
            if product_id in seen_ids:
                raise ValueError(f"Duplicate product_id in catalog CSV: {product_id}")
            seen_ids.add(product_id)

            product = session.get(Product, product_id)
            if product is None:
                product = Product(product_id=product_id)
                session.add(product)

            product.name = row["name"].strip()
            product.brand = row["brand"].strip()
            product.price_inr = _decimal(row, "price_inr")
            product.cpu = row["cpu"].strip()
            product.ram_gb = _integer(row, "ram_gb")
            product.storage_gb = _integer(row, "storage_gb")
            product.storage_type = row["storage_type"].strip()
            product.battery_hours = _decimal(row, "battery_hours")
            product.category = row["category"].strip()
            product.description = row["description"].strip()
            product.gpu = row["gpu"].strip()
            product.display_inches = _decimal(row, "display_inches")
            product.display_resolution = row["display_resolution"].strip()
            product.weight_kg = _decimal(row, "weight_kg")
            product.operating_system = row["operating_system"].strip()
            product.use_case = row["use_case"].strip()
            product.tags = row["tags"].strip()

            inventory = session.get(Inventory, product_id)
            if inventory is None:
                inventory = Inventory(product_id=product_id)
                session.add(inventory)
            inventory.stock_quantity = _integer(row, "stock_quantity")
            inventory.delivery_days = _integer(row, "delivery_days")
            inventory.data_source = "sample"
            count += 1

        session.flush()
        return count