"""Typed LangGraph orchestration for deterministic catalog recommendations."""

from decimal import Decimal
from typing import Any, Literal, TypedDict

from langchain_core.tools import ToolException
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from ecom_assistant.config import Settings
from ecom_assistant.db.session import Database
from ecom_assistant.schemas.catalog import (
    BudgetValidationResponse,
    InventoryResponse,
    ProductSearchFilters,
)
from ecom_assistant.schemas.product import ProductResponse
from ecom_assistant.schemas.shopping import (
    RankedRecommendation,
    RelaxedAlternative,
    ShoppingPreferenceOverrides,
    ShoppingRecommendationRequest,
    ShoppingRecommendationResponse,
    ShoppingRequirements,
    WorkflowStageTrace,
)
from ecom_assistant.services.catalog import CatalogService
from ecom_assistant.workflow.requirements import (
    RequirementExtractionError,
    RequirementExtractor,
)
from ecom_assistant.tools.catalog import create_shopping_tools

class ShoppingState(TypedDict, total=False):
    """Typed graph state shared between each shopping stage."""

    request: str
    structured_preferences: dict[str, Any] | None
    pre_extracted_requirements: dict[str, Any] | None
    requirements: dict[str, Any] | None
    extraction_mode: Literal["mock", "openai"] | None
    candidate_product_ids: list[str]
    search_results: list[dict[str, Any]]
    comparison_results: list[dict[str, Any]]
    inventory_results: list[dict[str, Any]]
    budget_validation: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    alternatives: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    trace: list[dict[str, str]]
    execution_metadata: dict[str, Any]
    status: str


def _trace(
    state: ShoppingState,
    stage: str,
    status: Literal["success", "warning", "failed", "skipped"],
    detail: str,
) -> list[dict[str, str]]:
    return [*state.get("trace", []), WorkflowStageTrace(stage=stage, status=status, detail=detail).model_dump()]


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def merge_preferences(
    extracted: ShoppingRequirements,
    overrides: ShoppingPreferenceOverrides | None,
) -> ShoppingRequirements:
    """Apply only explicitly supplied structured overrides."""
    if overrides is None:
        return extracted
    values = extracted.model_dump()
    for field in overrides.model_fields_set:
        values[field] = getattr(overrides, field)
    return ShoppingRequirements.model_validate(values)


class ShoppingWorkflow:
    """Construct and invoke the explicit multi-step shopping recommendation graph."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        catalog_service: CatalogService,
        extractor: RequirementExtractor | None = None,
    ) -> None:
        self.extractor = extractor or RequirementExtractor(settings)
        self.tools = create_shopping_tools(database, catalog_service)
        self.graph = self._compile_graph()

    def invoke(
        self,
        request: ShoppingRecommendationRequest,
        pre_extracted_requirements: ShoppingRequirements | None = None,
    ) -> ShoppingRecommendationResponse:
        """Run the compiled graph and validate the public response boundary."""
        initial_state: ShoppingState = {
            "request": request.customer_request,
            "structured_preferences": (
                request.preferences.model_dump(exclude_unset=True)
                if request.preferences is not None
                else None
            ),
            "pre_extracted_requirements": (
                pre_extracted_requirements.model_dump(mode="json")
                if pre_extracted_requirements is not None
                else None
            ),
            "requirements": None,
            "extraction_mode": None,
            "candidate_product_ids": [],
            "search_results": [],
            "comparison_results": [],
            "inventory_results": [],
            "budget_validation": [],
            "recommendations": [],
            "alternatives": [],
            "warnings": [],
            "errors": [],
            "trace": [],
            "execution_metadata": {
                "workflow_version": "shopping-recommendation-v1",
                "extractor_mode": self.extractor.mode,
                "candidate_count": 0,
            },
            "status": "failed",
        }
        result = self.graph.invoke(initial_state)
        return ShoppingRecommendationResponse(
            status=result["status"],
            customer_request=result["request"],
            extraction_mode=result.get("extraction_mode"),
            extracted_requirements=(
                ShoppingRequirements.model_validate(result["requirements"])
                if result.get("requirements") is not None
                else None
            ),
            candidate_product_ids=result.get("candidate_product_ids", []),
            matching_products=[
                ProductResponse.model_validate(item["product"])
                for item in result.get("recommendations", [])
            ],
            comparison_results=[
                ProductResponse.model_validate(product)
                for product in result.get("comparison_results", [])
            ],
            inventory_results=[
                InventoryResponse.model_validate(item)
                for item in result.get("inventory_results", [])
            ],
            budget_validation=[
                BudgetValidationResponse.model_validate(item)
                for item in result.get("budget_validation", [])
            ],
            recommendations=[
                RankedRecommendation.model_validate(item)
                for item in result.get("recommendations", [])
            ],
            alternatives=[
                RelaxedAlternative.model_validate(item)
                for item in result.get("alternatives", [])
            ],
            warnings=result.get("warnings", []),
            errors=result.get("errors", []),
            trace=[WorkflowStageTrace.model_validate(item) for item in result.get("trace", [])],
            execution_metadata=result.get("execution_metadata", {}),
        )

    def _compile_graph(self):
        builder = StateGraph(ShoppingState)
        builder.add_node("extract_requirements", self._extract_requirements)
        builder.add_node("product_search", self._product_search)
        builder.add_node("comparison", self._comparison)
        builder.add_node("inventory_validation", self._inventory_validation)
        builder.add_node("budget_check", self._budget_validation)
        builder.add_node("recommendation_ranking", self._recommendation_ranking)
        builder.add_node("recovery", self._recovery)
        builder.add_edge(START, "extract_requirements")
        builder.add_conditional_edges(
            "extract_requirements",
            self._after_extraction,
            {"search": "product_search", "recover": "recovery"},
        )
        builder.add_conditional_edges(
            "product_search",
            self._after_search,
            {"compare": "comparison", "recover": "recovery"},
        )
        builder.add_conditional_edges(
            "comparison",
            self._after_stage,
            {"inventory": "inventory_validation", "recover": "recovery"},
        )
        builder.add_conditional_edges(
            "inventory_validation",
            self._after_stage,
            {"budget": "budget_check", "recover": "recovery"},
        )
        builder.add_conditional_edges(
            "budget_check",
            self._after_stage,
            {"rank": "recommendation_ranking", "recover": "recovery"},
        )
        builder.add_conditional_edges(
            "recommendation_ranking",
            self._after_ranking,
            {"done": END, "recover": "recovery"},
        )
        builder.add_edge("recovery", END)
        return builder.compile()

    def _extract_requirements(self, state: ShoppingState) -> ShoppingState:
        try:
            extracted = (
                ShoppingRequirements.model_validate(state["pre_extracted_requirements"])
                if state.get("pre_extracted_requirements") is not None
                else self.extractor.extract(state["request"])
            )
            overrides = (
                ShoppingPreferenceOverrides.model_validate(state["structured_preferences"])
                if state.get("structured_preferences") is not None
                else None
            )
            requirements = merge_preferences(extracted, overrides)
        except RequirementExtractionError:
            message = "Requirement extraction failed; no mock fallback was used."
            return {
                "extraction_mode": self.extractor.mode,
                "errors": [message],
                "status": "failed",
                "trace": _trace(state, "requirement_extraction", "failed", message),
            }
        except ValidationError:
            message = "Requirements or structured overrides were invalid."
            return {
                "errors": [message],
                "status": "needs_clarification",
                "trace": _trace(state, "requirement_extraction", "failed", message),
            }

        warnings = list(state.get("warnings", []))
        if self.extractor.mode == "mock":
            warnings.append("Requirements were parsed by the deterministic mock extractor; no LLM was used.")
        if requirements.max_budget_inr is None:
            warnings.append("No maximum budget was stated; no budget ceiling was applied.")
        if requirements.min_ram_gb is None:
            warnings.append("No minimum RAM was stated; no RAM minimum was applied.")
        return {
            "requirements": requirements.model_dump(mode="json"),
            "extraction_mode": self.extractor.mode,
            "execution_metadata": {
                **state.get("execution_metadata", {}),
                "extractor_mode": self.extractor.mode,
            },
            "warnings": warnings,
            "status": "running",
            "trace": _trace(state, "requirement_extraction", "success", f"Extracted requirements in {self.extractor.mode} mode."),
        }

    @staticmethod
    def _after_extraction(state: ShoppingState) -> str:
        requirements = state.get("requirements")
        if state.get("errors") or not requirements or not requirements.get("category"):
            return "recover"
        return "search"

    def _product_search(self, state: ShoppingState) -> ShoppingState:
        requirements = ShoppingRequirements.model_validate(state["requirements"])
        filters = ProductSearchFilters(category=requirements.category, limit=100, sort_by="price_asc")
        try:
            result = self.tools["search"].invoke({"filters": filters.model_dump(exclude_none=True)})
        except (ToolException, SQLAlchemyError, ValidationError, ValueError):
            message = "Catalog search failed."
            return {
                "errors": [message],
                "status": "failed",
                "trace": _trace(state, "product_search", "failed", message),
            }
        products = result.get("products", [])
        return {
            "search_results": products,
            "candidate_product_ids": [product["product_id"] for product in products],
            "execution_metadata": {
                **state.get("execution_metadata", {}),
                "candidate_count": len(products),
            },
            "trace": _trace(
                state,
                "product_search",
                "success" if products else "warning",
                f"Catalog search returned {len(products)} products.",
            ),
        }

    @staticmethod
    def _after_search(state: ShoppingState) -> str:
        return "recover" if state.get("errors") or not state.get("candidate_product_ids") else "compare"

    def _comparison(self, state: ShoppingState) -> ShoppingState:
        try:
            result = self.tools["compare"].invoke(
                {"product_ids": state.get("candidate_product_ids", [])}
            )
        except (ToolException, SQLAlchemyError, ValidationError, ValueError):
            message = "Product comparison failed."
            return {
                "errors": [message],
                "status": "failed",
                "trace": _trace(state, "comparison", "failed", message),
            }
        if result.get("error"):
            message = "One or more catalog products could not be compared."
            return {
                "errors": [message],
                "status": "failed",
                "trace": _trace(state, "comparison", "failed", message),
            }
        return {
            "comparison_results": result["products"],
            "trace": _trace(state, "comparison", "success", f"Compared {len(result['products'])} catalog products."),
        }

    def _inventory_validation(self, state: ShoppingState) -> ShoppingState:
        inventory_results = []
        for product_id in state.get("candidate_product_ids", []):
            try:
                result = self.tools["inventory"].invoke({"product_id": product_id})
            except (ToolException, SQLAlchemyError, ValidationError, ValueError):
                result = {"error": "inventory_lookup_failed", "product_id": product_id}
            if result.get("error"):
                message = "Inventory validation failed for one or more products."
                return {
                    "inventory_results": inventory_results,
                    "errors": [message],
                    "status": "partial_failure",
                    "trace": _trace(state, "inventory_validation", "failed", message),
                }
            inventory_results.append(result)
        return {
            "inventory_results": inventory_results,
            "trace": _trace(state, "inventory_validation", "success", f"Checked inventory for {len(inventory_results)} products."),
        }

    def _budget_validation(self, state: ShoppingState) -> ShoppingState:
        requirements = ShoppingRequirements.model_validate(state["requirements"])
        if requirements.max_budget_inr is None:
            return {
                "budget_validation": [],
                "warnings": [*state.get("warnings", []), "Budget validation was skipped because no budget was stated."],
                "trace": _trace(state, "budget_validation", "skipped", "No explicit customer budget was provided."),
            }

        results = []
        for product_id in state.get("candidate_product_ids", []):
            try:
                result = self.tools["budget"].invoke(
                    {
                        "product_id": product_id,
                        "budget_inr": str(requirements.max_budget_inr),
                        "quantity": 1,
                    }
                )
            except (ToolException, SQLAlchemyError, ValidationError, ValueError):
                result = {"error": "budget_validation_failed", "product_id": product_id}
            if result.get("error"):
                message = "Budget validation failed for one or more products."
                return {
                    "budget_validation": results,
                    "errors": [message],
                    "status": "partial_failure",
                    "trace": _trace(state, "budget_validation", "failed", message),
                }
            results.append(result)
        return {
            "budget_validation": results,
            "trace": _trace(state, "budget_validation", "success", f"Validated budget for {len(results)} products."),
        }

    def _recommendation_ranking(self, state: ShoppingState) -> ShoppingState:
        requirements = ShoppingRequirements.model_validate(state["requirements"])
        inventory_by_id = {item["product_id"]: item for item in state["inventory_results"]}
        budget_by_id = {item["product_id"]: item for item in state["budget_validation"]}
        strict: list[dict[str, Any]] = []
        alternatives: list[dict[str, Any]] = []

        for product in state["comparison_results"]:
            product_id = product["product_id"]
            inventory = inventory_by_id[product_id]
            budget_result = budget_by_id.get(product_id)
            violations = self._hard_constraint_violations(
                product,
                inventory,
                budget_result,
                requirements,
            )
            factors, reasons, trade_offs = self._score_product(
                product,
                inventory,
                budget_result,
                requirements,
            )
            if violations:
                alternatives.append(
                    RelaxedAlternative(
                        product=ProductResponse.model_validate(product),
                        ranking_factors=factors,
                        relaxed_constraints=violations,
                        explanation="Not a strict match: " + "; ".join(violations),
                    ).model_dump(mode="json")
                )
            else:
                strict.append(
                    {
                        "product": product,
                        "ranking_factors": factors,
                        "reasons": reasons,
                        "trade_offs": trade_offs,
                    }
                )

        strict.sort(
            key=lambda item: (
                -sum(item["ranking_factors"].values()),
                Decimal(str(item["product"]["price_inr"])),
                item["product"]["product_id"],
            )
        )
        alternatives.sort(
            key=lambda item: (
                -sum(item["ranking_factors"].values()),
                Decimal(str(item["product"]["price_inr"])),
                item["product"]["product_id"],
            )
        )

        recommendations = [
            RankedRecommendation(
                product=ProductResponse.model_validate(item["product"]),
                rank=index,
                score=sum(item["ranking_factors"].values()),
                ranking_factors=item["ranking_factors"],
                reasons=item["reasons"],
                trade_offs=item["trade_offs"],
            ).model_dump(mode="json")
            for index, item in enumerate(strict, start=1)
        ]
        warnings = list(state.get("warnings", []))
        for item in state["comparison_results"]:
            if not item.get("specifications_complete", True):
                warnings.append(
                    f"{item['product_id']} has incomplete specifications: "
                    + ", ".join(item.get("missing_specifications", []))
                )
        stage_status = "success" if recommendations else "warning"
        detail = f"Ranked {len(recommendations)} strict matches and {len(alternatives)} alternatives."
        return {
            "recommendations": recommendations,
            "alternatives": alternatives,
            "warnings": list(dict.fromkeys(warnings)),
            "status": "success" if recommendations else "no_strict_matches",
            "trace": _trace(state, "recommendation_ranking", stage_status, detail),
        }

    @staticmethod
    def _hard_constraint_violations(
        product: dict[str, Any],
        inventory: dict[str, Any],
        budget: dict[str, Any] | None,
        requirements: ShoppingRequirements,
    ) -> list[str]:
        violations: list[str] = []
        if requirements.max_budget_inr is not None and budget is not None and not budget["within_budget"]:
            violations.append(
                f"price ₹{budget['total_price_inr']} exceeds maximum budget ₹{budget['budget_inr']}"
            )
        if requirements.min_ram_gb is not None:
            ram = product.get("ram_gb")
            if ram is None or ram < requirements.min_ram_gb:
                violations.append(
                    f"RAM {ram if ram is not None else 'unknown'} GB is below the "
                    f"minimum {requirements.min_ram_gb} GB"
                )
        if requirements.min_storage_gb is not None:
            storage = product.get("storage_gb")
            if storage is None or storage < requirements.min_storage_gb:
                violations.append(
                    f"storage {storage if storage is not None else 'unknown'} GB is below the "
                    f"minimum {requirements.min_storage_gb} GB"
                )
        if requirements.min_battery_hours is not None:
            battery = _decimal(product.get("battery_hours"))
            if battery is None or battery < requirements.min_battery_hours:
                violations.append(
                    "battery estimate is below the stated minimum "
                    f"{requirements.min_battery_hours} hours"
                )
        if requirements.in_stock_only is True and inventory["stock_status"] != "in_stock":
            violations.append(f"sample inventory status is {inventory['stock_status']}")
        return violations

    @staticmethod
    def _score_product(
        product: dict[str, Any],
        inventory: dict[str, Any],
        budget: dict[str, Any] | None,
        requirements: ShoppingRequirements,
    ) -> tuple[dict[str, int], list[str], list[str]]:
        price = Decimal(str(product["price_inr"]))
        ram = product.get("ram_gb")
        storage = product.get("storage_gb")
        battery = _decimal(product.get("battery_hours"))
        in_stock = inventory["stock_status"] == "in_stock"

        availability_score = 20 if in_stock else 0
        budget_score = 0
        if budget is not None and budget["within_budget"]:
            budget_score = 20
        elif requirements.max_budget_inr is None:
            budget_score = 10
        ram_score = min(15, int(ram or 0) // 2)
        storage_score = min(10, int(storage or 0) // 128)
        battery_score = min(15, int(battery or 0))

        catalog_text = " ".join(
            str(product.get(field) or "")
            for field in ("cpu", "description", "tags", "use_case", "gpu")
        ).casefold()
        preference_score = 0
        matched_preferences = []
        if requirements.preferred_cpu:
            if requirements.preferred_cpu.casefold() in str(product.get("cpu") or "").casefold():
                preference_score += 10
                matched_preferences.append(requirements.preferred_cpu)
        for preference in requirements.other_preferences:
            if "battery" in preference.casefold():
                if battery is not None and battery >= Decimal("10"):
                    preference_score += 5
                    matched_preferences.append(preference)
            elif preference.casefold() in catalog_text:
                preference_score += 5
                matched_preferences.append(preference)
        preference_score = min(20, preference_score)

        factors = {
            "availability": availability_score,
            "budget_fit": budget_score,
            "ram_capacity": ram_score,
            "storage_capacity": storage_score,
            "battery_estimate": battery_score,
            "preference_match": preference_score,
        }
        reasons = [
            f"Catalog price is ₹{price} (sample data).",
            f"Catalog RAM is {ram} GB." if ram is not None else "Catalog RAM is not listed.",
            f"Catalog storage is {storage} GB." if storage is not None else "Catalog storage is not listed.",
            (
                f"Catalog battery estimate is {battery} hours."
                if battery is not None
                else "Catalog battery estimate is not listed."
            ),
            (
                f"Sample inventory shows {inventory['stock_quantity']} units and an estimated "
                f"{inventory['estimated_delivery_days']}-day delivery."
                if inventory["stock_status"] != "unknown"
                else "Sample inventory is unavailable for this product."
            ),
        ]
        if matched_preferences:
            reasons.append("Matches stated preferences present in catalog fields: " + ", ".join(matched_preferences) + ".")
        trade_offs = []
        if requirements.max_budget_inr is not None:
            remaining = requirements.max_budget_inr - price
            trade_offs.append(f"₹{remaining} of the stated budget remains before any fees.")
        if not product.get("specifications_complete", True):
            trade_offs.append(
                "Specifications are incomplete: " + ", ".join(product.get("missing_specifications", []))
            )
        return factors, reasons, trade_offs

    @staticmethod
    def _after_stage(state: ShoppingState) -> str:
        if state.get("errors") or state.get("status") == "partial_failure":
            return "recover"
        stage = state.get("trace", [])[-1].get("stage") if state.get("trace") else ""
        if stage == "comparison":
            return "inventory"
        if stage == "inventory_validation":
            return "budget"
        return "rank"

    @staticmethod
    def _after_ranking(state: ShoppingState) -> str:
        return "done" if state.get("recommendations") else "recover"

    @staticmethod
    def _recovery(state: ShoppingState) -> ShoppingState:
        if state.get("errors"):
            status = state.get("status", "failed")
            detail = "The shopping workflow could not complete; no recommendation was fabricated."
        elif not state.get("requirements") or not state["requirements"].get("category"):
            status = "needs_clarification"
            detail = "Please specify what product category you are shopping for."
        elif not state.get("candidate_product_ids"):
            status = "no_results"
            detail = "No catalog products matched the stated category. Try another category or search term."
        else:
            status = "no_strict_matches"
            detail = "No available products satisfy all explicit hard requirements. Alternatives show which constraints they miss."
        return {
            "status": status,
            "warnings": [*state.get("warnings", []), detail],
            "trace": _trace(state, "recovery", "warning" if not state.get("errors") else "failed", detail),
        }
