from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# PYDANTIC MODELS FOR QUESTION EXTRACTION LLM
# =============================================================================

class PydanticQuestion(BaseModel):
    """Pydantic model for question extraction."""
    question_text: str = Field(..., description="The complete question text")
    original_number: Optional[str] = Field(None, description="Original question number if present")
    question_type: Optional[str] = Field(None, description="Type of question")
    category: Optional[str] = Field(None, description="Category or domain")
    confidence: float = Field(1.0, ge=0, le=1, description="Confidence in this extraction")


class QuestionExtractionResult(BaseModel):
    """Container for extraction results."""
    questions: List[PydanticQuestion]
