from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class DocumentItem(BaseModel):
    """A single item extracted from the document structure."""
    
    type: Literal["SECTION", "SUBSECTION", "QUESTION", "CONTEXT", "TOC"] = Field(
        description="The category of the text block."
    )
    clean_text: str = Field(
        description="The content with OCR artifacts removed and broken words fixed."
    )
    original_numbering: Optional[str] = Field(
        default=None,
        description="The extracted numbering ID (e.g., '1.2', 'Q3', '2.a'). Null if none exists."
    )
    is_mandatory: bool = Field(
        default=False, 
        description="True if the text implies a mandatory requirement or action."
    )

class SemanticChunkResult(BaseModel):
    """The extraction result for a specific chunk of text."""
    items: List[DocumentItem] = Field(description="List of structural items found in this chunk.")