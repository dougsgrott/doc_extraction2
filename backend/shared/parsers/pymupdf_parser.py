"""PyMuPDF4LLM parser wrapper."""

from typing import Union, BinaryIO
import fitz  # PyMuPDF
import pymupdf4llm
from .base import BaseDocumentParser, ParsedDocument


class PyMuPDFParser(BaseDocumentParser):
    """
    Wrapper around pymupdf4llm for markdown-based PDF parsing.

    Open-source alternative to Azure Document Intelligence.
    Returns markdown-formatted text optimized for LLM consumption.
    """

    def parse(self, file_content: Union[bytes, BinaryIO]) -> ParsedDocument:
        """
        Parse PDF using pymupdf4llm.

        Args:
            file_content: PDF file as bytes or file-like object

        Returns:
            ParsedDocument with markdown format
        """
        # Open PDF with PyMuPDF
        doc = fitz.open("pdf", file_content)

        # Convert to markdown using pymupdf4llm
        markdown_text = pymupdf4llm.to_markdown(doc)

        # Get page count
        page_count = len(doc)

        # Build metadata
        metadata = {
            'parser': 'pymupdf4llm',
            'page_count': page_count
        }

        # Add document metadata if available
        if doc.metadata:
            metadata['document_metadata'] = {
                k: v for k, v in doc.metadata.items()
                if v  # Only include non-empty values
            }

        doc.close()

        return ParsedDocument(
            content=markdown_text,
            format='markdown',
            page_count=page_count,
            metadata=metadata
        )
