"""
Factory for creating extractor instances - simplified version without registry.
"""

from typing import Optional
import logging

from .base import (
    DocumentStructureExtractor,
    QuestionExtractor,
    QuestionAnswerExtractor,
    StructureExtractorConfig,
    QuestionExtractorConfig,
    QAExtractorConfig,
)

# Direct imports of implementations
from .structure.structure_aware import StructureAwareStructureExtractor
from .question.simple_llm import SimpleLLMQuestionExtractor
from .question.context_aware import ContextAwareQuestionExtractor
from .qa.simple import SimpleQAExtractor
from .qa.context_aware import ContextAwareQAExtractor

logger = logging.getLogger(__name__)


class ExtractorFactory:
    """
    Factory for creating extractor instances using simple dictionaries.

    Usage:
        factory = ExtractorFactory()
        extractor = factory.create_qa_extractor("simple")
        qa_pairs = extractor.extract(document_text)
    """

    # Strategy mappings - simple dictionaries instead of registry
    _STRUCTURE_EXTRACTORS = {
        "structure_aware": StructureAwareStructureExtractor,
    }

    _QUESTION_EXTRACTORS = {
        "simple_llm": SimpleLLMQuestionExtractor,
        "context_aware": ContextAwareQuestionExtractor,
    }

    _QA_EXTRACTORS = {
        "simple": SimpleQAExtractor,
        "context_aware": ContextAwareQAExtractor,
    }

    def create_structure_extractor(
        self,
        strategy: str = "structure_aware",
        config: Optional[StructureExtractorConfig] = None
    ) -> DocumentStructureExtractor:
        """Create a document structure extractor."""
        if strategy not in self._STRUCTURE_EXTRACTORS:
            raise ValueError(
                f"Unknown structure extractor: {strategy}. "
                f"Available: {list(self._STRUCTURE_EXTRACTORS.keys())}"
            )

        extractor_class = self._STRUCTURE_EXTRACTORS[strategy]
        config = config or StructureExtractorConfig()

        logger.debug(f"Creating structure extractor: {strategy}")
        return extractor_class(config=config)

    def create_question_extractor(
        self,
        strategy: str = "simple_llm",
        config: Optional[QuestionExtractorConfig] = None,
        structure_extractor: Optional[DocumentStructureExtractor] = None
    ) -> QuestionExtractor:
        """Create a question extractor."""
        if strategy not in self._QUESTION_EXTRACTORS:
            raise ValueError(
                f"Unknown question extractor: {strategy}. "
                f"Available: {list(self._QUESTION_EXTRACTORS.keys())}"
            )

        extractor_class = self._QUESTION_EXTRACTORS[strategy]
        config = config or QuestionExtractorConfig()

        logger.debug(f"Creating question extractor: {strategy}")
        instance = extractor_class(config=config)

        # Inject structure extractor if provided
        if structure_extractor and hasattr(instance, 'set_structure_extractor'):
            instance.set_structure_extractor(structure_extractor)

        return instance

    def create_qa_extractor(
        self,
        strategy: str = "simple",
        config: Optional[QAExtractorConfig] = None
    ) -> QuestionAnswerExtractor:
        """Create a question+answer extractor."""
        if strategy not in self._QA_EXTRACTORS:
            raise ValueError(
                f"Unknown Q&A extractor: {strategy}. "
                f"Available: {list(self._QA_EXTRACTORS.keys())}"
            )

        extractor_class = self._QA_EXTRACTORS[strategy]
        config = config or QAExtractorConfig()

        logger.debug(f"Creating Q&A extractor: {strategy}")
        return extractor_class(config=config)
