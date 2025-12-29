"""
Chunk database model for document chunking.

Represents a semantically meaningful chunk of a document with
its embedding for vector similarity search.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime
import uuid
from sqlalchemy import String, Text, Integer, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .base import Base, uuid_pk, created_at

if TYPE_CHECKING:
    from .document import Document


class Chunk(Base):
    """
    Document chunk entity for RAG retrieval.

    Represents a semantically meaningful chunk of a document with
    its embedding for vector similarity search.
    """
    __tablename__ = "chunks"

    # Primary key
    id: Mapped[uuid.UUID] = uuid_pk()

    # Foreign key to document
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Chunk content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(1536),  # OpenAI embedding dimension
        nullable=True
    )

    # Chunk metadata
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # Sequential order
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_header: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Extensible metadata (JSONB for flexibility)
    # Can store: {table_data, image_ref, formatting, custom_fields, etc.}
    # Note: Using 'chunk_metadata' instead of 'metadata' to avoid SQLAlchemy reserved name
    chunk_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB, nullable=True)

    # Timestamp (no update - chunks are immutable)
    created_at: Mapped[datetime] = created_at()

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks"
    )

    # Indexes for common queries
    __table_args__ = (
        Index('idx_chunk_document_index', 'document_id', 'chunk_index'),
        Index('idx_chunk_page', 'page_number'),
        # Note: Vector index created separately via create_vector_indexes()
        # CREATE INDEX ON chunks USING ivfflat (content_embedding vector_cosine_ops);
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"
