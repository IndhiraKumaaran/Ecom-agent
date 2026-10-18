"""Relational models for catalog, inventory, simulated carts, and FAQ content."""

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ecom_assistant.db.base import Base


class Product(Base):
    """A catalog product with searchable laptop specifications."""

    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("price_inr > 0", name="ck_products_price_positive"),
        CheckConstraint("ram_gb > 0", name="ck_products_ram_positive"),
        CheckConstraint("storage_gb > 0", name="ck_products_storage_positive"),
        CheckConstraint("battery_hours > 0", name="ck_products_battery_positive"),
        CheckConstraint("display_inches > 0", name="ck_products_display_positive"),
        CheckConstraint("weight_kg > 0", name="ck_products_weight_positive"),
    )

    product_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    brand: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    price_inr: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    cpu: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    ram_gb: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_gb: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_type: Mapped[str] = mapped_column(String(24), nullable=False)
    battery_hours: Mapped[Decimal] = mapped_column(Numeric(4, 1), nullable=False)
    category: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    gpu: Mapped[str] = mapped_column(String(120), nullable=False)
    display_inches: Mapped[Decimal] = mapped_column(Numeric(4, 1), nullable=False)
    display_resolution: Mapped[str] = mapped_column(String(40), nullable=False)
    weight_kg: Mapped[Decimal] = mapped_column(Numeric(4, 2), nullable=False)
    operating_system: Mapped[str] = mapped_column(String(80), nullable=False)
    use_case: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    tags: Mapped[str] = mapped_column(Text, nullable=False, default="")

    inventory: Mapped["Inventory | None"] = relationship(
        back_populates="product", cascade="all, delete-orphan", uselist=False
    )
    faq_documents: Mapped[list["FAQDocument"]] = relationship(back_populates="product")


class Inventory(Base):
    """Clearly labeled sample stock and delivery estimate for a product."""

    __tablename__ = "inventory"
    __table_args__ = (
        CheckConstraint("stock_quantity >= 0", name="ck_inventory_stock_nonnegative"),
        CheckConstraint("delivery_days >= 0", name="ck_inventory_delivery_nonnegative"),
    )

    product_id: Mapped[str] = mapped_column(
        ForeignKey("products.product_id", ondelete="CASCADE"), primary_key=True
    )
    stock_quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    delivery_days: Mapped[int] = mapped_column(Integer, nullable=False)
    data_source: Mapped[str] = mapped_column(String(24), nullable=False, default="sample")
    product: Mapped[Product] = relationship(back_populates="inventory")


class Cart(Base):
    """A simulated cart; confirmation does not represent an order or payment."""

    __tablename__ = "carts"
    __table_args__ = (
        CheckConstraint("status IN ('open', 'confirmed')", name="ck_carts_status"),
    )

    cart_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    items: Mapped[list["CartItem"]] = relationship(
        back_populates="cart", cascade="all, delete-orphan"
    )


class CartItem(Base):
    """A product quantity and price snapshot in a simulated cart."""

    __tablename__ = "cart_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_cart_items_quantity_positive"),
        CheckConstraint("unit_price_inr > 0", name="ck_cart_items_price_positive"),
        UniqueConstraint("cart_id", "product_id", name="uq_cart_items_cart_product"),
    )

    cart_item_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cart_id: Mapped[str] = mapped_column(
        ForeignKey("carts.cart_id", ondelete="CASCADE"), nullable=False, index=True
    )
    product_id: Mapped[str] = mapped_column(
        ForeignKey("products.product_id", ondelete="RESTRICT"), nullable=False
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price_inr: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    cart: Mapped[Cart] = relationship(back_populates="items")
    product: Mapped[Product] = relationship()


class FAQDocument(Base):
    """Catalog-grounded FAQ content ready for later vector indexing."""

    __tablename__ = "faq_documents"
    __table_args__ = (
        UniqueConstraint("product_id", "question", name="uq_faq_product_question"),
    )

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str | None] = mapped_column(
        ForeignKey("products.product_id", ondelete="SET NULL"), nullable=True, index=True
    )
    document_type: Mapped[str] = mapped_column(String(32), nullable=False, default="faq")
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(80), nullable=False, default="catalog")
    product: Mapped[Product | None] = relationship(back_populates="faq_documents")