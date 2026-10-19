"""Deterministic LangGraph routing across existing shopping capabilities."""

import json
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ecom_assistant.config import Settings
from ecom_assistant.db.models import Cart, CartProposal, ConversationContext, Product
from ecom_assistant.db.session import Database
from ecom_assistant.rag.service import FAQServiceError
from ecom_assistant.schemas.cart import CartProposalRequest
from ecom_assistant.schemas.catalog import BudgetValidationResponse, InventoryResponse, ProductSearchFilters
from ecom_assistant.schemas.conversation import (
    ConversationIntent,
    ConversationRequest,
    ConversationResponse,
    ConversationTraceEvent,
)
from ecom_assistant.schemas.faq import FAQAskRequest
from ecom_assistant.schemas.product import ProductResponse
from ecom_assistant.schemas.shopping import (
    ShoppingPreferenceOverrides,
    ShoppingRecommendationRequest,
    ShoppingRequirements,
)
from ecom_assistant.services.cart import (
    CartMutationRejectedError,
    CartNotFoundError,
    CartProposalNotFoundError,
    CartService,
)
from ecom_assistant.services.catalog import (
    CatalogService,
    MissingProductsError,
    ProductNotFoundError,
)
from ecom_assistant.workflow.graph import ShoppingWorkflow, merge_preferences
from ecom_assistant.workflow.requirements import RequirementExtractionError

PRODUCT_ID_RE = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
QUANTITY_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
QUANTITY_RE = re.compile(
    r"\b(\d+|one|two|three|four|five)\s+(?:units?|items?|pieces?|of\s+(?:that|it|this))\b",
    re.IGNORECASE,
)
NEGATED_CONFIRMATION_RE = re.compile(
    r"\b(?:do not|don't|never|not yet|should not|shouldn't)\b.{0,40}\b(?:confirm|approve)\b",
    re.IGNORECASE,
)


class ConversationState(TypedDict, total=False):
    """One request's bounded typed state; durable context is stored separately."""

    message: str
    session_id: str
    cart_id: str | None
    proposal_id: str | None
    preferences: dict[str, Any] | None
    context: dict[str, Any]
    requirements: dict[str, Any] | None
    context_expired: bool
    intent: ConversationIntent
    status: str
    answer: str
    products: list[dict[str, Any]]
    resolved_product_ids: list[str]
    comparison: list[dict[str, Any]]
    recommendation: dict[str, Any] | None
    inventory: list[dict[str, Any]]
    budget: dict[str, Any] | None
    faq: dict[str, Any] | None
    cart: dict[str, Any] | None
    proposal: dict[str, Any] | None
    clarification_questions: list[str]
    recovery_suggestions: list[str]
    trace: list[dict[str, str]]
    errors: list[str]
    execution_metadata: dict[str, Any]


def _event(
    state: ConversationState,
    event: str,
    status: Literal["success", "warning", "failed", "skipped"],
    detail: str,
) -> list[dict[str, str]]:
    safe_detail = detail[:240]
    return [
        *state.get("trace", []),
        ConversationTraceEvent(event=event, status=status, detail=safe_detail).model_dump(),
    ]


class ConversationWorkflow:
    """Compile a deterministic intent graph over the existing service layer."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        catalog_service: CatalogService,
        shopping_workflow: ShoppingWorkflow,
        cart_service: CartService,
        faq_service,
    ) -> None:
        self.settings = settings
        self.database = database
        self.catalog_service = catalog_service
        self.shopping_workflow = shopping_workflow
        self.cart_service = cart_service
        self.faq_service = faq_service
        self.graph = self._compile_graph()

    def invoke(self, request: ConversationRequest) -> ConversationResponse:
        """Load expiring structured context, invoke the graph, then persist references."""
        session_id = request.session_id or str(uuid4())
        context, context_expired = self._load_context(session_id)
        initial: ConversationState = {
            "message": request.message,
            "session_id": session_id,
            "cart_id": request.cart_id or context.get("active_cart_id"),
            "proposal_id": request.proposal_id,
            "preferences": request.preferences.model_dump(exclude_unset=True)
            if request.preferences
            else None,
            "context": context,
            "requirements": context.get("last_requirements"),
            "context_expired": context_expired,
            "intent": "unknown",
            "status": "error",
            "answer": "",
            "products": [],
            "resolved_product_ids": [],
            "comparison": [],
            "recommendation": None,
            "inventory": [],
            "budget": None,
            "faq": None,
            "cart": None,
            "proposal": None,
            "clarification_questions": [],
            "recovery_suggestions": [],
            "trace": [],
            "errors": [],
            "execution_metadata": {"router": "deterministic", "graph": "conversation-v1"},
        }
        result: ConversationState = self.graph.invoke(initial)
        self._persist_context(result)
        return ConversationResponse(
            session_id=session_id,
            intent=result.get("intent", "unknown"),
            status=result.get("status", "error"),
            answer=result.get("answer", "I could not complete that request."),
            products=[ProductResponse.model_validate(item) for item in result.get("products", [])],
            comparison=[ProductResponse.model_validate(item) for item in result.get("comparison", [])],
            recommendation=result.get("recommendation"),
            inventory=[InventoryResponse.model_validate(item) for item in result.get("inventory", [])],
            budget=BudgetValidationResponse.model_validate(result["budget"])
            if result.get("budget")
            else None,
            faq=result.get("faq"),
            sources=result.get("faq", {}).get("sources", []) if result.get("faq") else [],
            cart=result.get("cart"),
            proposal=result.get("proposal"),
            clarification_questions=result.get("clarification_questions", []),
            recovery_suggestions=result.get("recovery_suggestions", []),
            trace=[ConversationTraceEvent.model_validate(item) for item in result.get("trace", [])],
            execution_metadata=result.get("execution_metadata", {}),
        )

    def _compile_graph(self):
        builder = StateGraph(ConversationState)
        builder.add_node("intent_detection", self._detect_intent)
        handler_nodes = {}
        for intent, handler in (
            ("product_search", self._product_search),
            ("recommendation", self._recommendation),
            ("comparison", self._comparison),
            ("inventory", self._inventory),
            ("budget_validation", self._budget_validation),
            ("faq", self._faq),
            ("cart_view", self._cart_view),
            ("cart_proposal", self._cart_proposal),
            ("cart_confirmation", self._cart_confirmation),
            ("clarification", self._clarification),
            ("unknown", self._unknown),
        ):
            node_name = f"handle_{intent}"
            handler_nodes[intent] = node_name
            builder.add_node(node_name, handler)
        builder.add_edge(START, "intent_detection")
        builder.add_conditional_edges(
            "intent_detection",
            lambda state: state["intent"],
            handler_nodes,
        )
        for node_name in handler_nodes.values():
            builder.add_edge(node_name, END)
        return builder.compile()

    def _detect_intent(self, state: ConversationState) -> ConversationState:
        if state.get("context_expired"):
            if re.search(r"\b(it|that one|those|them|both|this proposal|that proposal)\b", state["message"].casefold()):
                intent: ConversationIntent = "clarification"
                detail = "Conversation context expired."
            else:
                intent = self.classify(state["message"], state.get("proposal_id"))
                detail = f"Selected {intent}; expired context was ignored."
        else:
            intent = self.classify(state["message"], state.get("proposal_id"))
            detail = f"Selected {intent} using deterministic intent rules."
        return {
            "intent": intent,
            "execution_metadata": {
                **state.get("execution_metadata", {}),
                "intent_classifier": "deterministic",
            },
            "trace": _event(state, "intent_detected", "success", detail),
        }

    @staticmethod
    def classify(message: str, proposal_id: str | None = None) -> ConversationIntent:
        """Map clear language to a supported action; agreement alone is never confirmation."""
        text = message.casefold()
        if NEGATED_CONFIRMATION_RE.search(text):
            return "clarification"
        has_confirmation = bool(
            re.search(r"\b(confirm|approve|explicitly confirm)\b", text)
        )
        has_proposal_reference = bool(proposal_id or UUID_RE.search(message))
        if has_confirmation:
            return "cart_confirmation"
        if re.fullmatch(r"\s*(okay|ok|yes|sure|sounds good|go ahead)[.!\s]*", text):
            return "clarification"

        cart_mentioned = bool(re.search(r"\b(cart|basket)\b", text))
        if cart_mentioned and re.search(
            r"\b(?:do not|don't|never|should not|shouldn't)\s+(?:add|put|remove|delete|update|change)\b",
            text,
        ):
            return "clarification"
        if cart_mentioned and re.search(r"\b(view|show|see|what(?:'s| is) in|contents?)\b", text):
            return "cart_view"
        if cart_mentioned and re.search(r"\b(add|put|remove|delete|update|change|set)\b", text):
            return "cart_proposal"
        if has_proposal_reference:
            return "clarification"

        if re.search(r"\b(?:show|find|get)\s+(?:me\s+)?(?:cheaper|lower-priced|less expensive)\s+options\b", text):
            return "product_search"
        if re.search(r"\b(compare|comparison|difference|which is cheaper|which costs less)\b", text):
            return "comparison"
        if re.search(r"\b(warranty|return policy|refund|business policy|guaranteed delivery)\b", text):
            return "faq"
        if re.search(r"\b(inventory|stock|in stock|available|availability|delivery estimate|when will)\b", text):
            return "inventory"
        if re.search(r"\b(budget|under budget|within budget|cost|price|total cost)\b", text):
            return "budget_validation"
        if (
            re.search(r"\b(?:under|below|less than|up to)\b", text)
            and PRODUCT_ID_RE.search(message)
        ) or re.search(r"\b(?:is|will|can)\s+(?:it|this|that)\s+(?:under|below)\b", text):
            return "budget_validation"
        if re.search(r"\b(recommend|recommendation|suggest|best for|best laptop)\b", text):
            return "recommendation"
        if re.search(r"\b(search|find|show|list|browse|discover)\b", text):
            return "product_search"
        if re.search(r"\b(what is|what's|does it have|how much ram|battery life|processor|specification)\b", text):
            return "faq"
        if re.search(r"\b(laptop|notebook|desktop|tablet)s?\b", text):
            return "product_search"
        if re.search(r"\b(it|that one|those|them|both|the first|the second)\b", text):
            return "clarification"
        if re.search(r"\b(other one|other product|meant the other)\b", text):
            return "clarification"
        return "unknown"

    def _product_search(self, state: ConversationState) -> ConversationState:
        try:
            requirements = self._extract_with_context(state)
        except RequirementExtractionError:
            return self._failure(state, "Product filters could not be extracted safely.")
        prior_ids = state.get("context", {}).get("last_product_ids", [])
        explicit_ids = self._resolve_product_ids(state)
        if requirements.category is None and not prior_ids and not explicit_ids:
            return self._clarify(
                state,
                "Which product category should I search?",
                event="catalog_searched",
            )
        filters = ProductSearchFilters(
            category=requirements.category,
            max_price_inr=requirements.max_budget_inr,
            min_ram_gb=requirements.min_ram_gb,
            min_storage_gb=requirements.min_storage_gb,
            min_battery_hours=requirements.min_battery_hours,
            in_stock=requirements.in_stock_only,
            limit=100,
            sort_by="price_asc",
        )
        try:
            with self.database.session() as session:
                if re.search(
                    r"\b(?:show|find|get)\s+(?:me\s+)?(?:cheaper|lower-priced|less expensive)\s+options\b",
                    state["message"],
                    re.IGNORECASE,
                ):
                    if not prior_ids:
                        return self._clarify(
                            state,
                            "Which previously shown product should I find cheaper options than?",
                            event="catalog_searched",
                        )
                    previous_products = self.catalog_service.compare_products(session, prior_ids)
                    cheapest_cutoff = max(product.price_inr for product in previous_products) - Decimal("0.01")
                    max_price = min(
                        filters.max_price_inr or cheapest_cutoff,
                        cheapest_cutoff,
                    )
                    filters = filters.model_copy(update={"max_price_inr": max_price})
                products, _total = self.catalog_service.search_products(session, filters)
                product_ids = [product.product_id for product in products]
                if explicit_ids:
                    products_by_id = {product.product_id: product for product in products}
                    products = [products_by_id[product_id] for product_id in explicit_ids if product_id in products_by_id]
                    product_ids = [product.product_id for product in products]
                if requirements.category is None and prior_ids:
                    allowed = set(prior_ids)
                    products = [product for product in products if product.product_id in allowed]
                    product_ids = [product.product_id for product in products]
                result_products = [
                    self.catalog_service.to_response(product).model_dump(mode="json")
                    for product in products
                ]
        except SQLAlchemyError:
            return self._failure(state, "Product search is temporarily unavailable.")

        if not product_ids:
            return {
                "intent": "product_search",
                "status": "no_results",
                "requirements": requirements.model_dump(mode="json"),
                "answer": "No catalog products matched those filters.",
                "products": [],
                "recovery_suggestions": ["Try a higher budget, lower minimum specification, or another category."],
                "trace": _event(state, "catalog_searched", "warning", "Search returned no catalog products."),
            }
        return {
            "intent": "product_search",
            "status": "success",
            "requirements": requirements.model_dump(mode="json"),
            "answer": f"Found {len(result_products)} matching fictional catalog products.",
            "products": result_products,
            "resolved_product_ids": product_ids,
            "resolved_product_ids": product_ids,
            "trace": _event(state, "catalog_searched", "success", f"Returned {len(product_ids)} catalog products."),
        }

    def _recommendation(self, state: ConversationState) -> ConversationState:
        preferences = state.get("preferences")
        try:
            requirements = self._extract_with_context(state)
            if preferences is not None:
                requirements = merge_preferences(
                    requirements,
                    ShoppingRecommendationRequest(
                        customer_request=state["message"],
                        preferences=preferences,
                    ).preferences,
                )
            if not requirements.category:
                return self._clarify(
                    state,
                    "What product category should I recommend?",
                    event="recommendation_requested",
                )
            request = ShoppingRecommendationRequest(customer_request=state["message"])
            result = self.shopping_workflow.invoke(
                request,
                pre_extracted_requirements=requirements,
            )
        except RequirementExtractionError:
            return self._failure(
                state,
                "Requirement extraction is unavailable; no mock fallback was used.",
            )
        except ValidationError:
            return self._clarify(
                state,
                "Could you restate the product category and any required budget or specifications?",
                event="recommendation_requested",
            )
        recommendations = result.model_dump(mode="json")
        if result.status in {"failed", "partial_failure"}:
            return self._failure(state, "Recommendation services could not complete the request.")
        status = "success" if result.recommendations else (
            "no_results" if result.status == "no_results" else "clarification_required"
        )
        answer = (
            f"Found {len(result.recommendations)} strict matches."
            if result.recommendations
            else "No strict matches were found; any alternatives list the constraints they miss."
        )
        return {
            "intent": "recommendation",
            "status": status,
            "requirements": result.extracted_requirements.model_dump(mode="json")
            if result.extracted_requirements
            else None,
            "answer": answer,
            "recommendation": recommendations,
            "products": [item.product.model_dump(mode="json") for item in result.recommendations],
            "trace": _event(state, "recommendations_ranked", "success" if result.recommendations else "warning", answer),
        }

    def _comparison(self, state: ConversationState) -> ConversationState:
        product_ids = self._resolve_product_ids(state)
        if len(product_ids) != 2:
            return self._clarify(
                state,
                "Which two catalog products should I compare? Name them or refer to exactly two previously shown products.",
                event="comparison_requested",
            )
        try:
            with self.database.session() as session:
                products = self.catalog_service.compare_products(session, product_ids)
                serialized = [
                    self.catalog_service.to_response(product).model_dump(mode="json")
                    for product in products
                ]
        except (ProductNotFoundError, MissingProductsError):
            return self._clarify(state, "One product was not found; please name two catalog products.", "comparison_requested")
        except SQLAlchemyError:
            return self._failure(state, "Product comparison is temporarily unavailable.")

        answer = "Here are the stored catalog specifications in your requested order."
        if re.search(r"\b(cheaper|costs less|lowest price)\b", state["message"].casefold()):
            cheapest = min(serialized, key=lambda product: Decimal(str(product["price_inr"])))
            answer = f"{cheapest['name']} has the lowest listed sample price at ₹{cheapest['price_inr']}."
        return {
            "intent": "comparison",
            "status": "success",
            "answer": answer,
            "products": serialized,
            "comparison": serialized,
            "resolved_product_ids": product_ids,
            "trace": _event(state, "products_compared", "success", f"Compared {len(serialized)} actual catalog products."),
        }

    def _inventory(self, state: ConversationState) -> ConversationState:
        product_ids = self._resolve_product_ids(state)
        if not product_ids:
            return self._clarify(state, "Which catalog product should I check for stock or delivery?", "inventory_checked")
        try:
            results = []
            with self.database.session() as session:
                for product_id in product_ids:
                    results.append(self.catalog_service.get_inventory(session, product_id).__dict__)
        except ProductNotFoundError:
            return self._clarify(state, "I could not resolve that product ID; please name a catalog product.", "inventory_checked")
        except SQLAlchemyError:
            return self._failure(state, "Inventory lookup is temporarily unavailable.")
        summaries = [f"{item['product_id']}: {item['stock_status']}" for item in results]
        return {
            "intent": "inventory",
            "status": "success",
            "answer": "Sample inventory status: " + "; ".join(summaries) + ".",
            "inventory": results,
            "resolved_product_ids": product_ids,
            "trace": _event(state, "inventory_checked", "success", f"Checked {len(results)} catalog inventory records."),
        }

    def _budget_validation(self, state: ConversationState) -> ConversationState:
        product_ids = self._resolve_product_ids(state)
        if len(product_ids) != 1:
            return self._clarify(state, "Which one catalog product should I validate against your budget?", "budget_validated")
        try:
            requirements = self._extract_with_context(state)
        except RequirementExtractionError:
            return self._failure(state, "The budget could not be extracted safely.")
        if requirements.max_budget_inr is None:
            return self._clarify(state, "What maximum budget in INR should I use?", "budget_validated")
        try:
            with self.database.session() as session:
                result = self.catalog_service.validate_budget(
                    session,
                    product_ids[0],
                    requirements.max_budget_inr,
                    1,
                )
        except ProductNotFoundError:
            return self._clarify(state, "That product was not found in the catalog.", "budget_validated")
        except SQLAlchemyError:
            return self._failure(state, "Budget validation is temporarily unavailable.")
        value = {
            **result.__dict__,
            "sample_data": True,
        }
        return {
            "intent": "budget_validation",
            "status": "success",
            "requirements": requirements.model_dump(mode="json"),
            "answer": (
                f"Catalog price is ₹{result.total_price_inr}; "
                + ("it is within" if result.within_budget else "it exceeds")
                + f" your ₹{result.budget_inr} budget."
            ),
            "budget": value,
            "resolved_product_ids": product_ids,
            "trace": _event(state, "budget_validated", "success", "Budget calculated from server-side catalog price."),
        }

    def _faq(self, state: ConversationState) -> ConversationState:
        if re.search(r"\b(it|that one|this product)\b", state["message"].casefold()):
            resolved = self._resolve_product_ids(state)
            if len(resolved) != 1:
                return self._clarify(
                    state,
                    "Which product are you asking about? I have more than one possible match.",
                    "faq_clarification",
                )
        product_ids = self._resolve_product_ids(state)
        product_id = product_ids[0] if len(product_ids) == 1 else None
        try:
            from ecom_assistant.schemas.faq import FAQAskRequest

            with self.database.session() as session:
                result = self.faq_service.ask(
                    session,
                    FAQAskRequest(question=state["message"], product_id=product_id),
                )
        except (FAQServiceError, SQLAlchemyError):
            return self._failure(state, "FAQ retrieval is temporarily unavailable.")
        status = "success" if result.status == "grounded" else "insufficient_evidence"
        return {
            "intent": "faq",
            "status": status,
            "answer": result.answer,
            "faq": result.model_dump(mode="json"),
            "resolved_product_ids": [product_id] if product_id else [],
            "trace": _event(
                state,
                "faq_retrieved",
                "success" if result.status == "grounded" else "warning",
                f"FAQ retrieval returned {len(result.sources)} source references.",
            ),
        }

    def _cart_view(self, state: ConversationState) -> ConversationState:
        cart_id = state.get("cart_id")
        if not cart_id:
            return self._clarify(state, "Provide a cart ID or create a simulated cart first.", "cart_viewed")
        try:
            with self.database.session() as session:
                result = self.cart_service.get_cart(session, cart_id)
        except CartNotFoundError:
            return self._clarify(state, "That simulated cart was not found. Create a cart or check its ID.", "cart_viewed")
        except SQLAlchemyError:
            return self._failure(state, "The simulated cart could not be retrieved.")
        return {
            "intent": "cart_view",
            "status": "success",
            "answer": f"This simulated cart has {len(result.items)} items and totals ₹{result.total_price_inr}.",
            "cart": result.model_dump(mode="json"),
            "resolved_product_ids": [item.product_id for item in result.items],
            "trace": _event(state, "cart_viewed", "success", "Read simulated cart contents."),
        }

    def _cart_proposal(self, state: ConversationState) -> ConversationState:
        cart_id = state.get("cart_id")
        if not cart_id:
            return self._clarify(state, "Provide a cart ID before proposing a cart change.", "cart_proposal")
        product_ids = self._resolve_product_ids(state)
        if len(product_ids) != 1:
            return self._clarify(state, "Which one catalog product should I add, update, or remove?", "cart_proposal")
        operation = self._cart_operation(state["message"])
        quantity_match = QUANTITY_RE.search(state["message"])
        quantity = None
        if quantity_match:
            quantity_token = quantity_match.group(1).casefold()
            quantity = (
                QUANTITY_WORDS[quantity_token]
                if quantity_token in QUANTITY_WORDS
                else int(quantity_token)
            )
        if operation in {"add", "update"} and quantity is None:
            return self._clarify(state, "What quantity should I include in the pending proposal?", "cart_proposal")
        try:
            request = CartProposalRequest(
                operation=operation,
                product_id=product_ids[0],
                quantity=quantity,
            )
            with self.database.session() as session:
                proposal = self.cart_service.propose(session, cart_id, request)
        except ProductNotFoundError:
            return self._clarify(state, "That product ID is not in the catalog.", "cart_proposal")
        except (CartMutationRejectedError, ValidationError) as exc:
            return {
                "intent": "cart_proposal",
                "status": "proposal_rejected",
                "answer": str(exc),
                "trace": _event(state, "proposal_created", "failed", "Cart proposal validation rejected the operation."),
            }
        except CartNotFoundError:
            return self._clarify(state, "That simulated cart was not found.", "cart_proposal")
        except SQLAlchemyError:
            return self._failure(state, "The cart proposal could not be created.")
        return {
            "intent": "cart_proposal",
            "status": "proposal_pending",
            "answer": (
                f"A simulated {proposal.operation} proposal is pending for {proposal.quantity or proposal.resulting_quantity} "
                f"unit(s) of {proposal.product_id}. Confirm proposal {proposal.proposal_id} explicitly; "
                "the cart has not changed."
            ),
            "proposal": proposal.model_dump(mode="json"),
            "resolved_product_ids": [proposal.product_id],
            "trace": _event(state, "proposal_created", "success", "Pending proposal created; cart was not mutated."),
        }

    def _cart_confirmation(self, state: ConversationState) -> ConversationState:
        message = state["message"].casefold()
        if NEGATED_CONFIRMATION_RE.search(message):
            return self._clarify(state, "Understood; the cart was not changed.", "confirmation_required")
        if not re.search(r"\b(confirm|approve)\b", message):
            return self._clarify(
                state,
                "Please explicitly say ‘confirm’ and identify the pending proposal. Generic agreement is not confirmation.",
                "confirmation_required",
            )
        explicit_id = state.get("proposal_id") or UUID_RE.search(state["message"])
        proposal_id = explicit_id.group(0) if hasattr(explicit_id, "group") else explicit_id
        if not proposal_id and re.search(r"\b(this|that|the) proposal\b", message):
            contextual_proposal_id = state.get("context", {}).get("last_proposal_id")
            cart_id = state.get("cart_id")
            if contextual_proposal_id and cart_id:
                with self.database.session() as session:
                    pending_ids = session.scalars(
                        select(CartProposal.proposal_id).where(
                            CartProposal.cart_id == cart_id,
                            CartProposal.status == "pending",
                        )
                    ).all()
                if pending_ids == [contextual_proposal_id]:
                    proposal_id = contextual_proposal_id
        cart_id = state.get("cart_id")
        if not proposal_id or not cart_id:
            return self._clarify(
                state,
                "Provide the cart ID and exact pending proposal ID to confirm.",
                "confirmation_required",
            )
        try:
            with self.database.session() as session:
                result = self.cart_service.confirm(session, cart_id, proposal_id)
        except (CartNotFoundError, CartProposalNotFoundError):
            return self._clarify(state, "The cart or pending proposal was not found.", "confirmation_required")
        except SQLAlchemyError:
            return self._failure(state, "The proposal could not be confirmed.")
        status_by_cart = {
            "confirmed": "proposal_confirmed",
            "already_confirmed": "proposal_confirmed",
            "expired": "proposal_expired",
            "stale": "proposal_stale",
            "rejected": "proposal_rejected",
        }
        return {
            "intent": "cart_confirmation",
            "status": status_by_cart[result.status],
            "answer": result.message,
            "cart": result.cart.model_dump(mode="json") if result.cart else None,
            "proposal": result.proposal.model_dump(mode="json"),
            "resolved_product_ids": [result.proposal.product_id],
            "trace": _event(state, "proposal_confirmed", "success" if result.status in {"confirmed", "already_confirmed"} else "warning", result.status),
        }

    def _clarification(self, state: ConversationState) -> ConversationState:
        expired_message = "This conversation context expired; please restate the product or cart details."
        pending_proposal = state.get("context", {}).get("last_proposal_id")
        if pending_proposal and re.fullmatch(
            r"\s*(okay|ok|yes|sure|sounds good|go ahead)[.!\s]*",
            state["message"].casefold(),
        ):
            return self._clarify(
                state,
                f"I have not changed the cart. To proceed, explicitly confirm proposal {pending_proposal}.",
                "confirmation_required",
            )
        return self._clarify(
            state,
            expired_message if state.get("context_expired") else "Could you clarify what you would like me to do?",
            "clarification_requested",
        )

    def _unknown(self, state: ConversationState) -> ConversationState:
        return {
            "intent": "unknown",
            "status": "clarification_required",
            "answer": "I could not determine the shopping action. You can ask me to search, recommend, compare, check stock, validate a budget, ask a FAQ, or view/propose a cart change.",
            "clarification_questions": ["What would you like me to do?"],
            "trace": _event(state, "unknown_intent", "warning", "No supported deterministic intent rule matched."),
        }

    @staticmethod
    def _clarify(state: ConversationState, question: str, event: str) -> ConversationState:
        return {
            "intent": "clarification",
            "status": "clarification_required",
            "answer": question,
            "clarification_questions": [question],
            "trace": _event(state, event, "warning", question),
        }

    @staticmethod
    def _failure(state: ConversationState, message: str) -> ConversationState:
        return {
            "intent": state.get("intent", "unknown"),
            "status": "error",
            "answer": message,
            "trace": _event(state, "capability_failed", "failed", message),
        }

    def _extract_with_context(self, state: ConversationState) -> ShoppingRequirements:
        extracted = self.shopping_workflow.extractor.extract(state["message"])
        prior_json = state.get("context", {}).get("last_requirements")
        if prior_json:
            prior = ShoppingRequirements.model_validate(prior_json)
            values = prior.model_dump()
            for field in (
                "category",
                "max_budget_inr",
                "min_ram_gb",
                "min_storage_gb",
                "preferred_cpu",
                "min_battery_hours",
                "in_stock_only",
            ):
                new_value = getattr(extracted, field)
                if new_value is not None:
                    values[field] = new_value
            if extracted.other_preferences:
                values["other_preferences"] = list(
                    dict.fromkeys([*prior.other_preferences, *extracted.other_preferences])
                )
            extracted = ShoppingRequirements.model_validate(values)
        preference_overrides = state.get("preferences")
        if preference_overrides:
            extracted = merge_preferences(
                extracted,
                ShoppingPreferenceOverrides.model_validate(preference_overrides),
            )
        return extracted

    def _resolve_product_ids(self, state: ConversationState) -> list[str]:
        message = state["message"]
        ids = list(dict.fromkeys(PRODUCT_ID_RE.findall(message)))
        if ids:
            return ids
        with self.database.session() as session:
            products = session.scalars(select(Product).order_by(Product.product_id)).all()
            matches = [
                product.product_id
                for product in products
                if re.search(rf"(?<!\w){re.escape(product.name)}(?!\w)", message, re.IGNORECASE)
            ]
        if matches:
            longest = max(
                len(next(product.name for product in products if product.product_id == product_id))
                for product_id in matches
            )
            return [
                product_id
                for product_id in matches
                if len(next(product.name for product in products if product.product_id == product_id)) == longest
            ]

        prior_ids = state.get("context", {}).get("last_product_ids", [])
        normalized = message.casefold()
        if state.get("intent") == "cart_proposal" and len(prior_ids) == 1:
            return prior_ids
        if re.search(r"\b(those two|both|compare those|compare them|which is cheaper)\b", normalized):
            return prior_ids if len(prior_ids) == 2 else []
        if re.search(r"\b(it|that one|this product|the first|the second)\b", normalized):
            if len(prior_ids) == 1:
                return prior_ids
            if len(prior_ids) >= 2 and re.search(r"\b(first|second)\b", normalized):
                return [prior_ids[0 if "first" in normalized else 1]]
        if re.search(r"\b(?:that|this)\s+(?:laptop|notebook|desktop|tablet|product)\b", normalized):
            return prior_ids if len(prior_ids) == 1 else []
        if "cheapest" in normalized and prior_ids:
            return prior_ids[:1] if len(prior_ids) == 1 else []
        return []

    @staticmethod
    def _cart_operation(message: str) -> Literal["add", "update", "remove"]:
        normalized = message.casefold()
        if re.search(r"\b(remove|delete|take out)\b", normalized):
            return "remove"
        if re.search(r"\b(update|change|set)\b", normalized):
            return "update"
        return "add"

    def _load_context(self, session_id: str) -> tuple[dict[str, Any], bool]:
        with self.database.session() as session:
            record = session.get(ConversationContext, session_id)
            if record is None:
                return {}, False
            if self._is_expired(record.expires_at):
                return {}, True
            return {
                "active_cart_id": record.active_cart_id,
                "last_product_ids": json.loads(record.last_product_ids_json),
                "last_requirements": json.loads(record.last_requirements_json),
                "last_proposal_id": record.last_proposal_id,
                "last_intent": record.last_intent,
            }, False

    def _persist_context(self, state: ConversationState) -> None:
        context = state.get("context", {})
        products = state.get("products", [])
        recommendation = state.get("recommendation")
        if recommendation:
            products = [
                item["product"]
                for item in recommendation.get("recommendations", [])
            ] or products
        comparison = state.get("comparison", [])
        if comparison:
            products = comparison
        product_ids = state.get("resolved_product_ids") or [
            item["product_id"] for item in products if item.get("product_id")
        ]
        if not product_ids:
            product_ids = context.get("last_product_ids", [])
        requirements = state.get("requirements") or context.get("last_requirements", {})
        if recommendation and recommendation.get("extracted_requirements"):
            requirements = recommendation["extracted_requirements"]
        cart = state.get("cart") or {}
        active_cart_id = cart.get("cart_id") or state.get("cart_id") or context.get("active_cart_id")
        proposal = state.get("proposal") or {}
        last_proposal_id = context.get("last_proposal_id")
        if proposal.get("status") == "pending":
            last_proposal_id = proposal.get("proposal_id")
        elif state.get("intent") == "cart_confirmation":
            last_proposal_id = None

        now = datetime.now(UTC)
        with self.database.session() as session, session.begin():
            if active_cart_id is not None and session.get(Cart, active_cart_id) is None:
                active_cart_id = None
            record = session.get(ConversationContext, state["session_id"])
            if record is None:
                record = ConversationContext(
                    session_id=state["session_id"],
                    last_product_ids_json="[]",
                    last_requirements_json="{}",
                )
                session.add(record)
            record.active_cart_id = active_cart_id
            record.last_product_ids_json = json.dumps(product_ids)
            record.last_requirements_json = json.dumps(requirements)
            record.last_proposal_id = last_proposal_id
            record.last_intent = state.get("intent")
            record.updated_at = now
            record.expires_at = now + timedelta(seconds=self.settings.conversation_context_ttl_seconds)

    @staticmethod
    def _is_expired(value: datetime) -> bool:
        expires_at = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return expires_at <= datetime.now(UTC)