"""
Base module for extraction strategies.

Exports core interfaces, models, and utilities used by all extractors.
"""

from .interfaces import (
    BaseExtractor,
    DocumentStructureExtractor,
    QuestionExtractor,
    QuestionAnswerExtractor
)

from .models import (
    SectionType,
    DocumentFormat,
    ExtractionPriority,
    QuestionType,
    ConfidenceLevel,
    DocumentSection,
    DocumentStructure,
    ExtractedQuestion,
    ExtractedQAPair,
    ExtractionContext
)

from .config import (
    ExtractorConfig,
    StructureExtractorConfig,
    QuestionExtractorConfig,
    QAExtractorConfig,
    StrategyConfig
)

from .utils import (
    chunk_text,
    chunk_text_by_paragraphs,
    deduplicate_questions,
    deduplicate_qa_pairs,
    normalize_text,
    text_similarity,
    extract_question_number,
    # is_likely_question,
    rate_limit,
    log_extraction_stats,
    QuestionToQAAdapter
)

__all__ = [
    # Interfaces
    "BaseExtractor",
    "DocumentStructureExtractor",
    "QuestionExtractor",
    "QuestionAnswerExtractor",
    # Models - Enums
    "SectionType",
    "DocumentFormat",
    "ExtractionPriority",
    "QuestionType",
    "ConfidenceLevel",
    # Models - Data Classes
    "DocumentSection",
    "DocumentStructure",
    "ExtractedQuestion",
    "ExtractedQAPair",
    "ExtractionContext",
    # Config
    "ExtractorConfig",
    "StructureExtractorConfig",
    "QuestionExtractorConfig",
    "QAExtractorConfig",
    "StrategyConfig",
    # Utils
    "chunk_text",
    "chunk_text_by_paragraphs",
    "deduplicate_questions",
    "deduplicate_qa_pairs",
    "normalize_text",
    "text_similarity",
    "extract_question_number",
    # "is_likely_question",
    "rate_limit",
    "log_extraction_stats",
    "QuestionToQAAdapter",
]
