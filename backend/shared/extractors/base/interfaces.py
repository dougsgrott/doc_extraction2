"""
Base interfaces for extraction strategies.

This module defines the abstract base classes that all extractors must implement,
enabling the Strategy pattern for swappable extraction approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from .models import (
    DocumentStructure,
    ExtractedQuestion,
    ExtractedQAPair,
    ExtractionContext
)


class BaseExtractor(ABC):
    """ Base class for all extractors. """

    @abstractmethod
    def extract(self, document_text: str, context: Optional[ExtractionContext] = None) -> Any:
        """ Extract information from document text. """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass


class DocumentStructureExtractor(BaseExtractor):
    """ Base class for document structure extraction strategies. """

    @abstractmethod
    def extract(
        self,
        document_text: str,
        context: Optional[ExtractionContext] = None
    ) -> DocumentStructure:
        """ Extract document structure from text. """
        pass


class QuestionExtractor(BaseExtractor):
    """ Base class for question extraction strategies. """

    @abstractmethod
    def extract(
        self,
        document_text: str,
        context: Optional[ExtractionContext] = None
    ) -> List[ExtractedQuestion]:
        """ Extract questions from document text. """
        pass


class QuestionAnswerExtractor(BaseExtractor):
    """ Base class for question+answer extraction strategies. """

    @abstractmethod
    def extract(
        self,
        document_text: str,
        context: Optional[ExtractionContext] = None
    ) -> List[ExtractedQAPair]:
        """ Extract question-answer pairs from document text. """
        pass
