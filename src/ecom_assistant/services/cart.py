"""Transactional simulated-cart operations and explicit proposal confirmation."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ecom_assistant.config import Settings
from ecom_assistant.db.models import Cart, CartItem, CartProposal, Product
from ecom_assistant.schemas.cart import (
    CartConfirmationResponse,
    CartItemResponse,
    CartProposalRequest,
    CartProposalResponse,
    CartResponse,
)
from ecom_assistant.services.catalog import CatalogService, ProductNotFoundError


class CartNotFoundError(LookupError):
    """The requested simulated cart does not exist."""


class CartProposalNotFoundError(LookupError):
    """The requested proposal does not exist for this cart."""


class CartMutationRejectedError(ValueError):
    """A proposed operation is invalid for current cart or stock state."""


class CartService:
    """Use SQLite transactions for proposals and exact, idempotent confirmations."""

    def __init__(
        self,
        settings: Settings,
        catalog_service: CatalogService,
    ) -> None:
        self.proposal_ttl_seconds = settings.cart_proposal_ttl_seconds
        self.catalog_service = catalog_service

    def create_cart(self, session: Session) -> CartResponse:
        """Create an empty simulated cart."""
        with session.begin():
            cart = Cart(cart_id=str(uuid4()), status="open", revision=0)
            session.add(cart)
            session.flush()
            return self._cart_response(session, cart)

    def get_cart(self, session: Session, cart_id: str) -> CartResponse:
        """Return a current cart snapshot or raise not found."""
        cart = session.get(Cart, cart_id)
        if cart is None:
            raise CartNotFoundError(cart_id)
        return self._cart_response(session, cart)

    def propose(
        self,
        session: Session,
        cart_id: str,
        request: CartProposalRequest,
    ) -> CartProposalResponse:
        """Validate and persist an exact proposal without changing cart contents."""
        with session.begin():
            cart = self._require_cart(session, cart_id)
            product = self.catalog_service.get_product(session, request.product_id)
            item = session.scalar(
                select(CartItem).where(
                    CartItem.cart_id == cart_id,
                    CartItem.product_id == product.product_id,
                )
            )

            if request.operation == "add":
                resulting_quantity = (item.quantity if item else 0) + int(request.quantity)
            elif request.operation == "update":
                if item is None:
                    raise CartMutationRejectedError("Cannot update a product not already in the cart")
                resulting_quantity = int(request.quantity)
            else:
                if item is None:
                    raise CartMutationRejectedError("Cannot remove a product not already in the cart")
                resulting_quantity = 0

            if resulting_quantity > 50:
                raise CartMutationRejectedError("Cart item quantity cannot exceed 50")

            inventory = self.catalog_service.get_inventory(session, product.product_id)
            if request.operation != "remove":
                if inventory.stock_status == "unknown" or inventory.stock_quantity is None:
                    raise CartMutationRejectedError("Inventory is unknown; request cannot be proposed")
                if resulting_quantity > inventory.stock_quantity:
                    raise CartMutationRejectedError("Requested quantity exceeds current sample stock")

            projected_total = self._projected_total(
                session,
                cart_id,
                product,
                request.operation,
                resulting_quantity,
            )
            context_fingerprint = self._context_fingerprint(
                session,
                cart,
                product,
                inventory,
            )
            now = datetime.now(UTC)
            proposal = CartProposal(
                proposal_id=str(uuid4()),
                cart_id=cart_id,
                product_id=product.product_id,
                operation=request.operation,
                quantity=request.quantity or 0,
                resulting_quantity=resulting_quantity,
                unit_price_inr=product.price_inr,
                projected_total_inr=projected_total,
                cart_revision=cart.revision,
                context_fingerprint=context_fingerprint,
                inventory_quantity=inventory.stock_quantity,
                inventory_delivery_days=inventory.estimated_delivery_days,
                status="pending",
                created_at=now,
                expires_at=now + timedelta(seconds=self.proposal_ttl_seconds),
            )
            session.add(proposal)
            session.flush()
            return self._proposal_response(proposal)

    def confirm(
        self,
        session: Session,
        cart_id: str,
        proposal_id: str,
    ) -> CartConfirmationResponse:
        """Atomically consume one pending proposal and apply its exact mutation once."""
        now = datetime.now(UTC)
        with session.begin():
            claim = session.execute(
                update(CartProposal)
                .where(
                    CartProposal.proposal_id == proposal_id,
                    CartProposal.cart_id == cart_id,
                    CartProposal.status == "pending",
                    CartProposal.expires_at > now,
                )
                .values(status="applying")
            )
            proposal = session.scalar(
                select(CartProposal).where(
                    CartProposal.proposal_id == proposal_id,
                    CartProposal.cart_id == cart_id,
                )
            )
            if proposal is None:
                raise CartProposalNotFoundError(proposal_id)

            cart = session.get(Cart, cart_id)
            if cart is None:
                raise CartNotFoundError(cart_id)

            if claim.rowcount != 1:
                if proposal.status == "pending" and self._is_expired(proposal.expires_at, now):
                    proposal.status = "expired"
                    return self._confirmation_response(
                        "expired", proposal, cart, "The proposal expired; the cart was not changed."
                    )
                if proposal.status == "confirmed":
                    return self._confirmation_response(
                        "already_confirmed",
                        proposal,
                        cart,
                        "This proposal was already applied; no additional mutation was made.",
                    )
                status = proposal.status if proposal.status in {"expired", "stale", "rejected"} else "rejected"
                return self._confirmation_response(
                    status,
                    proposal,
                    cart,
                    f"The proposal is {status}; the cart was not changed.",
                )

            product = self.catalog_service.get_product(session, proposal.product_id)
            inventory = self.catalog_service.get_inventory(session, product.product_id)
            fingerprint = self._context_fingerprint(session, cart, product, inventory)
            if (
                cart.revision != proposal.cart_revision
                or product.price_inr != proposal.unit_price_inr
                or inventory.stock_quantity != proposal.inventory_quantity
                or inventory.estimated_delivery_days != proposal.inventory_delivery_days
                or fingerprint != proposal.context_fingerprint
            ):
                proposal.status = "stale"
                return self._confirmation_response(
                    "stale",
                    proposal,
                    cart,
                    "Cart, catalog price, or inventory changed; request a fresh proposal.",
                )

            if proposal.operation != "remove":
                if inventory.stock_status != "in_stock" or inventory.stock_quantity is None:
                    proposal.status = "rejected"
                    return self._confirmation_response(
                        "rejected",
                        proposal,
                        cart,
                        "Current sample inventory is unavailable; the cart was not changed.",
                    )
                if proposal.resulting_quantity > inventory.stock_quantity:
                    proposal.status = "rejected"
                    return self._confirmation_response(
                        "rejected",
                        proposal,
                        cart,
                        "Current sample stock is insufficient; the cart was not changed.",
                    )
                reserved_by_other_carts = session.scalar(
                    select(func.coalesce(func.sum(CartItem.quantity), 0)).where(
                        CartItem.product_id == product.product_id,
                        CartItem.cart_id != cart_id,
                    )
                ) or 0
                available_for_cart = inventory.stock_quantity - reserved_by_other_carts
                if proposal.resulting_quantity > available_for_cart:
                    proposal.status = "rejected"
                    return self._confirmation_response(
                        "rejected",
                        proposal,
                        cart,
                        "Other simulated carts account for the remaining sample stock; the cart was not changed.",
                    )

            item = session.scalar(
                select(CartItem).where(
                    CartItem.cart_id == cart_id,
                    CartItem.product_id == product.product_id,
                )
            )
            if proposal.operation == "remove":
                if item is None:
                    proposal.status = "stale"
                    return self._confirmation_response(
                        "stale", proposal, cart, "The cart item changed; request a fresh proposal."
                    )
                session.delete(item)
            elif proposal.operation == "add":
                if item is None:
                    session.add(
                        CartItem(
                            cart_id=cart_id,
                            product_id=product.product_id,
                            quantity=proposal.resulting_quantity,
                            unit_price_inr=proposal.unit_price_inr,
                        )
                    )
                else:
                    item.quantity = proposal.resulting_quantity
                    item.unit_price_inr = proposal.unit_price_inr
            else:
                if item is None:
                    proposal.status = "stale"
                    return self._confirmation_response(
                        "stale", proposal, cart, "The cart item changed; request a fresh proposal."
                    )
                item.quantity = proposal.resulting_quantity
                item.unit_price_inr = proposal.unit_price_inr

            cart.revision += 1
            proposal.status = "confirmed"
            proposal.confirmed_at = now
            session.flush()
            return self._confirmation_response(
                "confirmed",
                proposal,
                cart,
                "The simulated cart was updated; no purchase or order was made.",
            )

    def _require_cart(self, session: Session, cart_id: str) -> Cart:
        cart = session.get(Cart, cart_id)
        if cart is None:
            raise CartNotFoundError(cart_id)
        if cart.status != "open":
            raise CartMutationRejectedError("This simulated cart is not open for changes")
        return cart

    def _cart_response(self, session: Session, cart: Cart) -> CartResponse:
        items = session.scalars(
            select(CartItem)
            .where(CartItem.cart_id == cart.cart_id)
            .order_by(CartItem.product_id)
        ).all()
        response_items = [
            CartItemResponse(
                product_id=item.product_id,
                name=item.product.name,
                quantity=item.quantity,
                unit_price_inr=item.unit_price_inr,
                line_total_inr=(item.unit_price_inr * item.quantity).quantize(Decimal("0.01")),
            )
            for item in items
        ]
        total = sum((item.line_total_inr for item in response_items), Decimal("0.00"))
        return CartResponse(
            cart_id=cart.cart_id,
            status=cart.status,
            items=response_items,
            total_price_inr=total,
        )

    def _proposal_response(self, proposal: CartProposal) -> CartProposalResponse:
        return CartProposalResponse(
            proposal_id=proposal.proposal_id,
            cart_id=proposal.cart_id,
            operation=proposal.operation,
            product_id=proposal.product_id,
            quantity=proposal.quantity if proposal.operation != "remove" else None,
            resulting_quantity=proposal.resulting_quantity,
            unit_price_inr=proposal.unit_price_inr,
            projected_total_inr=proposal.projected_total_inr,
            status=proposal.status,
            expires_at=self._utc_datetime(proposal.expires_at).isoformat(),
        )

    def _confirmation_response(
        self,
        status: str,
        proposal: CartProposal,
        cart: Cart,
        message: str,
    ) -> CartConfirmationResponse:
        return CartConfirmationResponse(
            status=status,
            proposal=self._proposal_response(proposal),
            cart=self._cart_response_from_loaded(cart),
            message=message,
        )

    def _cart_response_from_loaded(self, cart: Cart) -> CartResponse:
        """Build a cart snapshot while the confirmation transaction is open."""
        items = list(cart.items)
        response_items = [
            CartItemResponse(
                product_id=item.product_id,
                name=item.product.name,
                quantity=item.quantity,
                unit_price_inr=item.unit_price_inr,
                line_total_inr=(item.unit_price_inr * item.quantity).quantize(Decimal("0.01")),
            )
            for item in sorted(items, key=lambda value: value.product_id)
        ]
        total = sum((item.line_total_inr for item in response_items), Decimal("0.00"))
        return CartResponse(
            cart_id=cart.cart_id,
            status=cart.status,
            items=response_items,
            total_price_inr=total,
        )

    def _projected_total(
        self,
        session: Session,
        cart_id: str,
        product: Product,
        operation: str,
        resulting_quantity: int,
    ) -> Decimal:
        items = session.scalars(select(CartItem).where(CartItem.cart_id == cart_id)).all()
        total = Decimal("0.00")
        target_seen = False
        for item in items:
            if item.product_id == product.product_id:
                target_seen = True
                if operation != "remove":
                    total += product.price_inr * resulting_quantity
            else:
                total += item.unit_price_inr * item.quantity
        if operation == "add" and not target_seen:
            total += product.price_inr * resulting_quantity
        return total.quantize(Decimal("0.01"))

    def _context_fingerprint(
        self,
        session: Session,
        cart: Cart,
        target_product: Product,
        target_inventory,
    ) -> str:
        items = session.scalars(
            select(CartItem)
            .where(CartItem.cart_id == cart.cart_id)
            .order_by(CartItem.product_id)
        ).all()
        item_context = []
        for item in items:
            catalog_product = self.catalog_service.get_product(session, item.product_id)
            item_context.append(
                {
                    "product_id": item.product_id,
                    "quantity": item.quantity,
                    "cart_unit_price": str(item.unit_price_inr),
                    "catalog_unit_price": str(catalog_product.price_inr),
                }
            )
        payload = {
            "cart_revision": cart.revision,
            "items": item_context,
            "target_product_id": target_product.product_id,
            "target_price": str(target_product.price_inr),
            "target_inventory_quantity": target_inventory.stock_quantity,
            "target_inventory_delivery_days": target_inventory.estimated_delivery_days,
            "target_inventory_source": target_inventory.data_source,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _utc_datetime(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @classmethod
    def _is_expired(cls, expires_at: datetime, now: datetime) -> bool:
        return cls._utc_datetime(expires_at) <= cls._utc_datetime(now)