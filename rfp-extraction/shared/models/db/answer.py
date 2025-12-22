"""
Answer database model (unified questions and answers).

Represents both unanswered questions and answered questions in a single table.
Can be linked to an RFP (for extracted questions) or standalone (for knowledge base).
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime
import uuid
from sqlalchemy import String, Text, Integer, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .base import Base, uuid_pk, created_at, updated_at
from ..enums import AnswerStatus, SourceType

if TYPE_CHECKING:
    from .rfp import RFP


class Answer(Base):
    """
    Unified Answer entity (combines questions and answers).

    Can represent:
    1. Unanswered RFP question (rfp_id set, answer_text NULL)
    2. Answered RFP question (rfp_id set, answer_text populated)
    3. Knowledge base Q&A pair (rfp_id NULL, source=KNOWLEDGE_BASE)
    4. Manually created Q&A (rfp_id NULL, source=MANUAL)
    """
    __tablename__ = "answers"

    # Primary key
    id: Mapped[uuid.UUID] = uuid_pk()

    # Foreign key to RFP (NULLABLE - not all answers come from RFPs)
    rfp_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rfps.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )

    # Question data
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(1536),  # OpenAI embedding dimension
        nullable=True
    )

    # Answer data (nullable for unanswered questions)
    answer_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status and classification
    status: Mapped[AnswerStatus] = mapped_column(
        default=AnswerStatus.PENDING,
        nullable=False,
        index=True
    )
    category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    section: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sequence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Order within RFP

    # Tags for flexible categorization
    tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)

    # Source tracking
    source: Mapped[SourceType] = mapped_column(
        default=SourceType.RFP_EXTRACTION,
        nullable=False,
        index=True
    )

    # Timestamps
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()

    # Relationships
    rfp: Mapped[Optional["RFP"]] = relationship(
        "RFP",
        back_populates="answers"
    )

    # Indexes for vector search and filtering
    __table_args__ = (
        Index('idx_answer_rfp_sequence', 'rfp_id', 'sequence'),
        Index('idx_answer_status_source', 'status', 'source'),
        Index('idx_answer_category', 'category'),
        # Note: Vector index created separately via create_vector_indexes()
        # CREATE INDEX ON answers USING ivfflat (question_embedding vector_cosine_ops);
    )

    def __repr__(self) -> str:
        return f"<Answer(id={self.id}, status={self.status.value}, source={self.source.value})>"

    @property
    def is_answered(self) -> bool:
        """Check if answer has been provided."""
        return self.answer_text is not None and self.answer_text.strip() != ""
