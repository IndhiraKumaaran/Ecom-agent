"""Validated request and response contracts for product FAQ retrieval."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FAQAskRequest(BaseModel):
    """Customer question and optional catalog product scope."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=5, max_length=1000)
    product_id: str | None = Field(default=None, min_length=1, max_length=32)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question", "product_id")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("question must not be blank")
        return value


class FAQSourceReference(BaseModel):
    """Source identity and Chroma relevance data for a grounded answer."""

    document_id: str
    product_id: str | None
    document_type: str
    source: str
    distance: float = Field(ge=0)


class FAQSnippet(FAQSourceReference):
    """A retrieved evidence excerpt and its source metadata."""

    content: str


class FAQAnswerResponse(BaseModel):
    """Grounded answer or a clear insufficient-evidence response."""

    status: Literal["grounded", "insufficient_evidence"]
    answer: str
    sources: list[FAQSourceReference]
    snippets: list[FAQSnippet]
    warnings: list[str]