"""
Database enumeration types.

This module defines all enums used in the database schema, providing
type-safe values for workflow states, status tracking, and classifications.

Note: Extraction-specific enums (SectionType, QuestionType, etc.) remain in
extractors/base/models.py as they are domain logic, not persistence.
"""

from enum import Enum


class PriorityLevel(str, Enum):
    """Priority level for RFPs."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProcessingStatus(str, Enum):
    """Processing status for documents and RFPs."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


class RFPStatus(str, Enum):
    """Workflow status for RFPs."""
    DRAFT = "draft"
    ACTIVE = "active"
    IN_REVIEW = "in_review"
    SUBMITTED = "submitted"
    AWARDED = "awarded"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class AnswerStatus(str, Enum):
    """Status for answers (unified questions/answers workflow)."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ANSWERED = "answered"
    APPROVED = "approved"
    REJECTED = "rejected"


class SourceType(str, Enum):
    """Source/origin of an answer."""
    RFP_EXTRACTION = "rfp_extraction"
    KNOWLEDGE_BASE = "knowledge_base"
    MANUAL = "manual"
    AI_GENERATED = "ai_generated"


class DocumentSourceType(str, Enum):
    """Type/category of knowledge base document."""
    POLICY = "policy"
    CASE_STUDY = "case_study"
    TECHNICAL = "technical"
    PROPOSAL = "proposal"
    PRESENTATION = "presentation"
    CONTRACT = "contract"
    OTHER = "other"
