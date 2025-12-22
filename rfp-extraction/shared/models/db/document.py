"""
Document database model for knowledge base.

Represents uploaded documents (policies, case studies, etc.) that are
chunked and embedded for RAG retrieval.
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from datetime import datetime
import uuid
from sqlalchemy import String, Text, Integer, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, uuid_pk, created_at, updated_at
from ..enums import DocumentSourceType, ProcessingStatus

if TYPE_CHECKING:
    from .chunk import Chunk


class Document(Base):
    """
    Knowledge base document entity.

    Represents uploaded documents (policies, case studies, etc.) that are
    chunked and embedded for RAG retrieval.
    """
    __tablename__ = "documents"

    # Primary key
    id: Mapped[uuid.UUID] = uuid_pk()

    # Document classification
    source_type: Mapped[DocumentSourceType] = mapped_column(
        nullable=False,
        index=True
    )

    # File information
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    blob_url: Mapped[str] = mapped_column(Text, nullable=False)  # Azure Blob Storage URL
    file_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # PDF, DOCX, etc.

    # Document metadata
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Processing metrics
    total_pages: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        default=ProcessingStatus.PENDING,
        nullable=False,
        index=True
    )

    # Extensible metadata (JSONB for flexibility)
    # Can store: {author, department, version, tags, custom_fields, etc.}
    # Note: Using 'doc_metadata' instead of 'metadata' to avoid SQLAlchemy reserved name
    doc_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()

    # Relationships
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select"
    )

    # Indexes for common queries
    __table_args__ = (
        Index('idx_document_source_status', 'source_type', 'processing_status'),
        Index('idx_document_filename', 'filename'),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename}, status={self.processing_status.value})>"
