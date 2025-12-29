"""Document parser abstraction for Azure Document Intelligence and pymupdf4llm."""

from .factory import create_parser
from .base import BaseDocumentParser, ParsedDocument

__all__ = ["create_parser", "BaseDocumentParser", "ParsedDocument"]
