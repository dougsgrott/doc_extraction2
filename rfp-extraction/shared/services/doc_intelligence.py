import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarginFilter:
    """Filter elements by page position (as percentage of page height/width)."""
    top: float = 0.10      # Filter top 10%
    bottom: float = 0.10   # Filter bottom 10%
    
    def is_in_margin(self, y_center: float, page_height: float) -> bool:
        """Check if y position is in header or footer margin."""
        relative_y = y_center / page_height
        return relative_y < self.top or relative_y > (1 - self.bottom)


class DocumentParser:
    def __init__(self, margin_filter: Optional[MarginFilter] = None):
        self.endpoint = os.environ.get("DI_ENDPOINT")
        self.key = os.environ.get("DI_KEY")
        
        if not self.endpoint or not self.key:
            raise ValueError("Missing Document Intelligence configuration.")

        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint, 
            credential=AzureKeyCredential(self.key)
        )
        
        self.margin_filter = margin_filter

    def parse_stream(self, file_stream) -> AnalyzeResult:
        """Sends a file stream to Azure Document Intelligence."""
        poller = self.client.begin_analyze_document(
            "prebuilt-layout", 
            file_stream,
        )
        return poller.result()

    def filter_document(self, result):
        # Build page height lookup
        page_heights = {}
        for page in result.pages:
            page_heights[page.page_number] = page.height
        
        # Filter paragraphs
        filtered_paragraphs = []
        for para in result.paragraphs:
            if self._is_body_content(para, page_heights):
                filtered_paragraphs.append(para.content)
        
        return '\n\n'.join(filtered_paragraphs)

    def parse_to_text(self, file_stream, filter_margins: bool = True) -> str:
        """ Parse document and return clean text content. """
        result = self.parse_stream(file_stream)
        
        if not filter_margins or not self.margin_filter:
            return result.content
        return self.filter_document(result)

    def parse_to_json(self, file_stream, filter_margins: bool = True) -> dict:
        """
        Parse document and return structured JSON.
        
        Returns:
            Dict with 'content', 'paragraphs', 'tables', 'pages' keys
        """
        result = self.parse_stream(file_stream)
        
        # Build page height lookup
        page_heights = {}
        for page in result.pages:
            page_heights[page.page_number] = page.height
        
        # Process paragraphs
        paragraphs = []
        for para in result.paragraphs:
            is_body = self._is_body_content(para, page_heights) if self.margin_filter else True
            para_data = {
                'content': para.content,
                'role': para.role,
                'is_body': is_body,
            }
            if not filter_margins or is_body:
                paragraphs.append(para_data)
        
        # Process tables
        tables = []
        for table in (result.tables or []):
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
        
        return {
            'content': '\n\n'.join(p['content'] for p in paragraphs),
            'paragraphs': paragraphs,
            'tables': tables,
            'page_count': len(result.pages),
            'filtered_margins': filter_margins and self.margin_filter is not None,
        }

    def _is_body_content(self, paragraph, page_heights: dict) -> bool:
        """Check if paragraph is in body (not header/footer)."""
        if not self.margin_filter or not paragraph.bounding_regions:
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
