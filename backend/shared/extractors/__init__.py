"""
Extraction strategies for RFP documents.

This package provides a clean, strategy-based architecture for extracting
information from RFP documents. It supports three types of extractors:

- DocumentStructureExtractor: Detects document structure and sections
- QuestionExtractor: Extracts questions from documents
- QuestionAnswerExtractor: Extracts question-answer pairs

Each extractor type supports multiple strategies:
- simple: Basic pattern-based extraction
- structure_aware: Uses document structure for targeted extraction
- context_aware: Uses context injection for better accuracy
- agentic: Multi-step extraction with self-validation

Usage:
    from shared.extractors import ExtractorFactory

    factory = ExtractorFactory()
    extractor = factory.create_qa_extractor("simple")
    qa_pairs = extractor.extract(document_text)
"""

# Export public API
from .base import (
    # Interfaces
    BaseExtractor,
    DocumentStructureExtractor,
    QuestionExtractor,
    QuestionAnswerExtractor,
    # Models - Enums
    SectionType,
    DocumentFormat,
    ExtractionPriority,
    QuestionType,
    ConfidenceLevel,
    # Models - Data Classes
    DocumentSection,
    DocumentStructure,
    ExtractedQuestion,
    ExtractedQAPair,
    ExtractionContext,
    # Config
    ExtractorConfig,
    StructureExtractorConfig,
    QuestionExtractorConfig,
    QAExtractorConfig,
    StrategyConfig,
)

from .factory import ExtractorFactory

__all__ = [
    # Main API
    "ExtractorFactory",
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
]

__version__ = "1.0.0"
