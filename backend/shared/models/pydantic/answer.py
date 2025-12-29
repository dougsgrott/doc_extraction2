"""
Answer Pydantic schemas for validation and API responses.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
import uuid

from ..enums import AnswerStatus, SourceType
from .common import TimestampMixin


class AnswerBase(BaseModel):
    """Base answer schema."""
    question_text: str = Field(..., min_length=1)
    answer_text: Optional[str] = None
    category: Optional[str] = Field(None, max_length=255)
    section: Optional[str] = Field(None, max_length=255)
    sequence: Optional[int] = Field(None, ge=0)
    tags: Optional[List[str]] = None


class AnswerCreate(AnswerBase):
    """Schema for creating a new answer."""
    rfp_id: Optional[uuid.UUID] = None
    source: SourceType = SourceType.MANUAL
    status: AnswerStatus = AnswerStatus.PENDING


class AnswerUpdate(BaseModel):
    """Schema for updating an answer."""
    answer_text: Optional[str] = None
    status: Optional[AnswerStatus] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class AnswerResponse(AnswerBase, TimestampMixin):
    """Schema for answer response."""
    id: uuid.UUID
    rfp_id: Optional[uuid.UUID]
    status: AnswerStatus
    source: SourceType
    is_answered: bool = Field(default=False, description="Whether answer is provided")

    @field_validator('is_answered', mode='before')
    @classmethod
    def compute_is_answered(cls, v, info):
        """Compute is_answered based on answer_text."""
        if v is not None:
            return v
        # Access answer_text from the data dict
        answer_text = info.data.get('answer_text')
        return bool(answer_text and answer_text.strip())

    model_config = ConfigDict(from_attributes=True)


class SimilarAnswer(AnswerResponse):
    """Answer with similarity score for RAG retrieval."""
    similarity_score: float = Field(..., ge=0, le=1, description="Cosine similarity score")
