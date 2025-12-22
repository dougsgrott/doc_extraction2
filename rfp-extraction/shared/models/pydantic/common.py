"""
Common Pydantic schemas and mixins.

Provides shared schemas used across multiple models for pagination,
timestamps, and other common patterns.
"""

from typing import Generic, TypeVar, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

T = TypeVar('T')


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(50, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""
    items: List[T]
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")


class TimestampMixin(BaseModel):
    """Mixin for created/updated timestamps."""
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)  # Pydantic v2 (was orm_mode in v1)
