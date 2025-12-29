"""
RFP (Request for Proposal) database model.

Represents an RFP document with metadata, processing status, and counters
for tracking question/answer progress.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import date, datetime
import uuid
from sqlalchemy import String, Text, Integer, Date, Index, Numeric
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, uuid_pk, created_at, updated_at
from ..enums import PriorityLevel, ProcessingStatus, RFPStatus

if TYPE_CHECKING:
    from .answer import Answer


class RFP(Base):
    """
    Request for Proposal entity.

    Represents an RFP document with metadata, processing status, and counters.
    The source_id is now just a text identifier (not a foreign key to Source table).
    """
    __tablename__ = "rfps"

    # Primary key
    id: Mapped[uuid.UUID] = uuid_pk()

    # Source identification (no FK, just identifier - could be filename, URL, etc.)
    source_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True, index=True)

    # RFP metadata
    client_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    value: Mapped[Optional[float]] = mapped_column(Numeric(precision=15, scale=2), nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # USD, EUR, etc.
    department: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    assigned_to: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # User/team
    due_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Enums for workflow tracking
    priority: Mapped[Optional[PriorityLevel]] = mapped_column(nullable=True, index=True)
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        default=ProcessingStatus.PENDING,
        nullable=False,
        index=True
    )
    status: Mapped[RFPStatus] = mapped_column(
        default=RFPStatus.DRAFT,
        nullable=False,
        index=True
    )

    # Document metadata
    format: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # PDF, DOCX, etc.

    # Question counters (denormalized for performance)
    questions_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    questions_answered: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    questions_approved: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Additional context/notes
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Attachments/metadata as JSONB (flexible structure)
    attachments: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()

    # Relationships
    answers: Mapped[List["Answer"]] = relationship(
        "Answer",
        back_populates="rfp",
        cascade="all, delete-orphan",
        lazy="select"
    )

    # Indexes for common queries
    __table_args__ = (
        Index('idx_rfp_client_name', 'client_name'),
        Index('idx_rfp_due_date', 'due_date'),
        Index('idx_rfp_status_priority', 'status', 'priority'),
    )

    def __repr__(self) -> str:
        return f"<RFP(id={self.id}, client={self.client_name}, status={self.status.value if self.status else None})>"
