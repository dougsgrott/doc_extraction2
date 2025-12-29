"""Base classes for document parsing abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Union, BinaryIO


@dataclass
class ParsedDocument:
    """
    Standard output format from all document parsers.

    This provides a unified interface regardless of which parser was used.
    """
    content: str # Main text content (plain text or markdown)
    format: str # 'plain_text' or 'markdown'
    page_count: int
    metadata: Dict[str, Any] = field(default_factory=dict) # Parser-specific metadata


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_content: Union[bytes, BinaryIO]) -> ParsedDocument:
        """
        Parse document content and return standardized output.

        Args:
            file_content: PDF file as bytes or file-like object

        Returns:
            ParsedDocument with standardized format
        """
        pass
