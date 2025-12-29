"""Azure Document Intelligence parser implementation."""

import os
from typing import Union, BinaryIO, Optional
from dataclasses import dataclass
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from .base import BaseDocumentParser, ParsedDocument


@dataclass
class MarginFilter:
    """Filter elements by page position (as percentage of page height/width)."""
    top: float = 0.10      # Filter top 10%
    bottom: float = 0.10   # Filter bottom 10%

    def is_in_margin(self, y_center: float, page_height: float) -> bool:
        """Check if y position is in header or footer margin."""
        relative_y = y_center / page_height
        return relative_y < self.top or relative_y > (1 - self.bottom)


class AzureDocumentParser(BaseDocumentParser):
    """
    Direct implementation using Azure Document Intelligence SDK.

    Provides OCR and structure detection using Azure's prebuilt-layout model.
    """

    def __init__(self, margin_filter: Optional[MarginFilter] = None):
        """
        Initialize Azure Document Intelligence parser.

        Args:
            margin_filter: Optional filter for removing headers/footers
        """
        endpoint = os.getenv("DI_ENDPOINT")
        key = os.getenv("DI_KEY")

        if not endpoint or not key:
            raise ValueError(
                "Azure Document Intelligence requires DI_ENDPOINT and DI_KEY environment variables"
            )

        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        self.margin_filter = margin_filter

    def parse(self, file_content: Union[bytes, BinaryIO]) -> ParsedDocument:
        """
        Parse PDF using Azure Document Intelligence.

        Args:
            file_content: PDF file as bytes or file-like object

        Returns:
            ParsedDocument with plain_text format and Azure metadata
        """
        # Call Azure Document Intelligence API
        poller = self.client.begin_analyze_document(
            "prebuilt-layout",
            file_content
        )
        result = poller.result()

        # Build page height lookup for margin filtering
        page_heights = {}
        for page in result.pages:
            page_heights[page.page_number] = page.height

        # Process paragraphs
        paragraphs = []
        for para in result.paragraphs:
            is_body = self._is_body_content(para, page_heights)
            para_data = {
                'content': para.content,
                'role': para.role if hasattr(para, 'role') else None,
                'is_body': is_body,
            }
            # Only include body paragraphs if filtering is enabled
            if not self.margin_filter or is_body:
                paragraphs.append(para_data)

        # Build content from filtered paragraphs
        content = '\n\n'.join(p['content'] for p in paragraphs)

        # Process tables
        tables = []
        if result.tables:
            for table in result.tables:
                tables.append({
                    'row_count': table.row_count,
                    'column_count': table.column_count,
                    'cells': [
                        {
                            'content': cell.content,
                            'row': cell.row_index,
                            'column': cell.column_index,
                        }
                        for cell in table.cells
                    ]
                })

        # Build metadata
        metadata = {
            'paragraphs_count': len(paragraphs),
            'tables_count': len(tables),
            'filtered_margins': self.margin_filter is not None,
            'parser': 'azure_document_intelligence'
        }

        # Include tables if present
        if tables:
            metadata['tables'] = tables

        return ParsedDocument(
            content=content,
            format='plain_text',
            page_count=len(result.pages),
            metadata=metadata
        )

    def _is_body_content(self, paragraph, page_heights: dict) -> bool:
        """Check if paragraph is in body (not header/footer)."""
        if not self.margin_filter or not hasattr(paragraph, 'bounding_regions') or not paragraph.bounding_regions:
            return True

        region = paragraph.bounding_regions[0]
        page_height = page_heights.get(region.page_number, 11.0)  # Default 11 inches

        # Get y center from polygon [x1,y1,x2,y2,x3,y3,x4,y4]
        polygon = region.polygon
        if len(polygon) >= 8:
            y_values = [polygon[i] for i in range(1, 8, 2)]
            y_center = sum(y_values) / len(y_values)
            return not self.margin_filter.is_in_margin(y_center, page_height)

        return True
