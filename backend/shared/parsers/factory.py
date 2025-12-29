"""Factory for creating document parsers."""

import os
from typing import Optional
from .base import BaseDocumentParser
from .azure_parser import AzureDocumentParser
from .pymupdf_parser import PyMuPDFParser


def create_parser(parser_type: Optional[str] = None) -> BaseDocumentParser:
    """ Create document parser based on configuration. """
    if parser_type is None:
        parser_type = os.getenv("DOCUMENT_PARSER", "pymupdf").lower()
    else:
        parser_type = parser_type.lower()

    if parser_type == "azure":
        return AzureDocumentParser()

    elif parser_type == "pymupdf":
        return PyMuPDFParser()

    else:
        raise ValueError(
            f"Unknown document parser: {parser_type}. Use 'azure' or 'pymupdf'"
        )
