"""
RFP Pydantic schemas for validation and API responses.
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict
from datetime import date, datetime
import uuid

from ..enums import PriorityLevel, ProcessingStatus, RFPStatus
from .common import TimestampMixin

if TYPE_CHECKING:
    from .answer import AnswerResponse


class RFPBase(BaseModel):
    """Base RFP schema with common fields."""
    source_id: Optional[str] = None
    client_name: Optional[str] = Field(None, max_length=255)
    title: Optional[str] = Field(None, max_length=500)
    value: Optional[float] = Field(None, ge=0)
    currency: Optional[str] = Field(None, max_length=10)
    department: Optional[str] = Field(None, max_length=255)
    assigned_to: Optional[str] = Field(None, max_length=255)
    due_date: Optional[date] = None
    priority: Optional[PriorityLevel] = None
    format: Optional[str] = Field(None, max_length=50)
    context: Optional[str] = None
    attachments: Optional[Dict[str, Any]] = None


class RFPCreate(RFPBase):
    """Schema for creating a new RFP."""
    client_name: str = Field(..., min_length=1, max_length=255)  # Required on create


class RFPUpdate(RFPBase):
    """Schema for updating an RFP (all fields optional)."""
    status: Optional[RFPStatus] = None
    processing_status: Optional[ProcessingStatus] = None


class RFPResponse(RFPBase, TimestampMixin):
    """Schema for RFP response (includes all fields)."""
    id: uuid.UUID
    processing_status: ProcessingStatus
    status: RFPStatus
    questions_total: int
    questions_answered: int
    questions_approved: int

    model_config = ConfigDict(from_attributes=True)


class RFPWithAnswers(RFPResponse):
    """RFP response with nested answers."""
    answers: List["AnswerResponse"] = []
