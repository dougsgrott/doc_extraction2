"""
Shared data models for extraction strategies.

This module defines the common data structures used across all extractor types,
ensuring consistency and interoperability.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SectionType(str, Enum):
    """Classification of document sections."""
    QUESTIONNAIRE = "questionnaire"
    INFORMATIONAL = "informational"
    MIXED = "mixed"
    PRICING = "pricing"
    APPENDIX = "appendix"
    INSTRUCTIONS = "instructions"
    DEFINITIONS = "definitions"
    UNKNOWN = "unknown"


class DocumentFormat(str, Enum):
    """Detected document format/template style."""
    FORMAL_NUMBERED = "formal_numbered"          # 1.1, 1.2, 1.2.1 style
    SECTION_HEADERS = "section_headers"          # "Section A:", "Part 1:" style
    QUESTION_NUMBERED = "question_numbered"      # Q1, Q2 or Question 1 style
    TABLE_BASED = "table_based"                  # Questions in tables
    MIXED_FORMAT = "mixed_format"                # Multiple formats in one doc
    FLAT_LIST = "flat_list"                      # Simple numbered/bulleted list
    UNSTRUCTURED = "unstructured"                # No clear structure
    CUSTOM = "custom"                            # Document-specific format


class ExtractionPriority(str, Enum):
    """Priority level for extraction."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SKIP = "skip"


class QuestionType(str, Enum):
    """Type of question detected."""
    DIRECT_QUESTION = "direct_question"           # Ends with ?
    IMPERATIVE_REQUEST = "imperative_request"     # "Describe...", "Provide..."
    CONFIRMATION_REQUEST = "confirmation_request" # "Confirm that..."
    CHECKBOX_ITEM = "checkbox_item"               # Checkbox or yes/no item
    TABLE_ENTRY = "table_entry"                   # Question in table format
    OTHER = "other"


class ConfidenceLevel(str, Enum):
    """Confidence level for extraction."""
    HIGH = "high"       # > 0.8
    MEDIUM = "medium"   # 0.5 - 0.8
    LOW = "low"         # < 0.5


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class DocumentSection:
    """
    Represents a detected section in the document.

    Sections are logical divisions of the document (e.g., "Instructions",
    "Questionnaire", "Pricing") with metadata about their type and
    extraction priority.
    """
    section_id: str
    section_number: Optional[str] = None
    section_title: str = ""
    section_type: SectionType = SectionType.UNKNOWN
    start_char: int = 0
    end_char: int = 0
    confidence: float = 1.0
    extraction_priority: ExtractionPriority = ExtractionPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate section boundaries."""
        if self.end_char < self.start_char:
            raise ValueError(f"Section {self.section_id}: end_char must be >= start_char")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Section {self.section_id}: confidence must be between 0 and 1")


@dataclass
class DocumentStructure:
    """
    Complete structural analysis of a document.

    Contains the document's sections, format information, and metadata
    about the document's organization.
    """
    document_title: Optional[str] = None
    document_type: Optional[str] = None
    document_format: DocumentFormat = DocumentFormat.UNSTRUCTURED
    sections: List[DocumentSection] = field(default_factory=list)
    format_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_section_by_id(self, section_id: str) -> Optional[DocumentSection]:
        """Get a section by its ID."""
        return next((s for s in self.sections if s.section_id == section_id), None)

    def get_sections_by_type(self, section_type: SectionType) -> List[DocumentSection]:
        """Get all sections of a specific type."""
        return [s for s in self.sections if s.section_type == section_type]

    def get_high_priority_sections(self) -> List[DocumentSection]:
        """Get all sections marked as high priority for extraction."""
        return [s for s in self.sections if s.extraction_priority == ExtractionPriority.HIGH]

    def __repr__(self) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("DOCUMENT STRUCTURE ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Type: {self.document_type}")
        lines.append(f"Format: {self.document_format.value}")
        lines.append(f"Sections: {len(self.sections)}")
        lines.append(f"Confidence: {self.format_confidence:.2f}")
        lines.append("")
        lines.append("SECTIONS:")
        lines.append("-" * 80)

        for i, section in enumerate(self.sections, 1):
            lines.append(f"\n{i}. {section.section_title}")
            lines.append(f"   ID: {section.section_id}")
            lines.append(f"   Type: {section.section_type.value}")
            lines.append(f"   Number: {section.section_number or 'N/A'}")
            lines.append(
                f"   Position: {section.start_char:,} - {section.end_char:,} "
                f"({section.end_char - section.start_char:,} chars)"
            )
            lines.append(f"   Priority: {section.extraction_priority.value}")
            lines.append(f"   Confidence: {section.confidence:.2f}")

            if section.metadata:
                meta = ", ".join(f"{k}={v}" for k, v in section.metadata.items())
                lines.append(f"   Metadata: {meta}")

        from collections import Counter
        type_counts = Counter(s.section_type.value for s in self.sections)

        lines.append("\n" + "=" * 80)
        lines.append("\nSections by Type:")
        for section_type, count in type_counts.most_common():
            lines.append(f"  {section_type}: {count}")

        return "\n".join(lines)


@dataclass
class ExtractedQuestion:
    """
    A single extracted question (without answer).

    Used by QuestionExtractor strategies that focus solely on
    identifying questions.
    """
    question_text: str
    original_number: Optional[str] = None
    question_type: Optional[QuestionType] = None
    category: Optional[str] = None
    confidence: float = 1.0
    section_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get the confidence level category."""
        if self.confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


@dataclass
class ExtractedQAPair:
    """
    A question-answer pair.

    Used by QuestionAnswerExtractor strategies that extract both
    questions and their associated answers.
    """
    question_text: str
    answer_text: Optional[str] = None
    original_number: Optional[str] = None
    question_type: Optional[QuestionType] = None
    category: Optional[str] = None
    confidence: float = 1.0
    section_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

    @property
    def has_answer(self) -> bool:
        """Check if this Q&A pair has an answer."""
        return self.answer_text is not None and self.answer_text.strip() != ""

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get the confidence level category."""
        if self.confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def to_question(self) -> ExtractedQuestion:
        """Convert this Q&A pair to an ExtractedQuestion (discarding answer)."""
        return ExtractedQuestion(
            question_text=self.question_text,
            original_number=self.original_number,
            question_type=self.question_type,
            category=self.category,
            confidence=self.confidence,
            section_id=self.section_id,
            metadata=self.metadata
        )


@dataclass
class ExtractionContext:
    """
    Context information passed between extractors.

    This allows extractors to build upon each other's results,
    e.g., question extraction can use pre-detected structure.
    """
    document_structure: Optional[DocumentStructure] = None
    llm_client: Optional[Any] = None
    config: Optional[Dict[str, Any]] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def has_structure(self) -> bool:
        """Check if document structure is available."""
        return self.document_structure is not None

    def get_structure(self) -> DocumentStructure:
        """Get document structure, raising error if not available."""
        if not self.has_structure():
            raise ValueError("Document structure not available in context")
        return self.document_structure
