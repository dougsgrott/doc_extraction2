"""
Document Pydantic schemas for validation and API responses.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import uuid

from ..enums import DocumentSourceType, ProcessingStatus
from .common import TimestampMixin


class DocumentBase(BaseModel):
    """Base document schema."""
    source_type: DocumentSourceType
    filename: str = Field(..., max_length=500)
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    doc_metadata: Optional[Dict[str, Any]] = None


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""
    blob_url: str  # Azure blob URL
    file_type: Optional[str] = Field(None, max_length=50)


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""
    title: Optional[str] = None
    description: Optional[str] = None
    processing_status: Optional[ProcessingStatus] = None
    total_chunks: Optional[int] = None
    total_pages: Optional[int] = None
    doc_metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase, TimestampMixin):
    """Schema for document response."""
    id: uuid.UUID
    blob_url: str
    file_type: Optional[str]
    total_pages: Optional[int]
    total_chunks: int
    processing_status: ProcessingStatus

    model_config = ConfigDict(from_attributes=True)
