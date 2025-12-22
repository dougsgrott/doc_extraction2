"""
Simple LLM-based Question Extractor.

This extractor uses basic chunking and straightforward LLM calls to extract questions.
No structure awareness, no multi-step processing - just simple extraction.

Best for:
- Quick baseline extraction
- Documents without clear structure
- Comparing against more sophisticated methods

Approach:
1. Chunk document with simple overlap
2. Extract questions from each chunk using LLM
3. Deduplicate results
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import AzureOpenAI

from ..base import (
    QuestionExtractor,
    ExtractedQuestion,
    ExtractionContext,
    QuestionExtractorConfig,
    QuestionType,
    chunk_text,
    deduplicate_questions,
    rate_limit
)

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PydanticQuestion(BaseModel):
    """Pydantic model for question extraction."""
    question_text: str = Field(..., description="The complete question text, VERBATIM from document")
    original_number: Optional[str] = Field(None, description="Original question number if present (e.g., '8.1.1', 'Q5')")
    category: Optional[str] = Field(None, description="Category or domain")
    confidence: float = Field(1.0, ge=0, le=1, description="Confidence in this extraction")


class QuestionExtractionResult(BaseModel):
    """Container for extraction results."""
    questions: List[PydanticQuestion]


# =============================================================================
# SIMPLE LLM QUESTION EXTRACTOR
# =============================================================================

SYSTEM_PROMPT = """You are an expert at extracting questions from procurement documents.

Extract all questions that require a response from the supplier/vendor.

CRITICAL RULES - Textual Fidelity:
- Extract question text VERBATIM from the document - do NOT rephrase or rewrite
- Preserve the original wording, punctuation, and structure exactly as written
- Only clean up obvious OCR artifacts (page headers/footers, page numbers)
- Extract the question number exactly as it appears (e.g., "2.1", "8.1.1", "Q5")

What to EXTRACT:
✓ Questions ending with '?'
✓ Imperative statements requesting information (Describe, Explain, Provide, List, etc.)
✓ Numbered items that clearly request information
✓ "Please provide...", "State whether...", etc.

What NOT to extract:
✗ Statements or descriptions (e.g., "Secure hosting included")
✗ Instructions or guidelines
✗ Section headers
✗ Definitions or examples

If multiple sub-questions share the same number, group them into ONE entry."""


class SimpleLLMQuestionExtractor(QuestionExtractor):
    """
    Simple LLM-based question extraction.

    Uses basic chunking and straightforward LLM calls without
    structure awareness or sophisticated processing.
    """

    def __init__(self, config: Optional[QuestionExtractorConfig] = None):
        """Initialize the extractor."""
        self.config = config or QuestionExtractorConfig()

        if not self.config.validate():
            logger.warning("LLM configuration incomplete. Extraction may fail.")

        self.client = AzureOpenAI(
            azure_endpoint=self.config.llm_endpoint,
            api_key=self.config.llm_api_key,
            api_version=self.config.llm_api_version
        )

    @property
    def strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "simple_llm"

    def extract(
        self,
        document_text: str,
        context: Optional[ExtractionContext] = None
    ) -> List[ExtractedQuestion]:
        """
        Extract questions using simple LLM-based approach.

        Args:
            document_text: The full text of the document
            context: Optional extraction context (not used in simple approach)

        Returns:
            List of ExtractedQuestion objects
        """
        if not document_text or not document_text.strip():
            logger.warning("Empty document text provided")
            return []

        logger.info("Starting simple LLM question extraction...")

        # Chunk the document
        chunks = chunk_text(
            document_text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap
        )

        logger.info(f"Processing {len(chunks)} chunks...")

        all_questions = []

        for i, chunk in enumerate(chunks):
            logger.debug(f"Extracting from chunk {i+1}/{len(chunks)}...")

            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.config.llm_deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": f"Extract all questions from this text:\n\n{chunk}"
                        }
                    ],
                    response_format=QuestionExtractionResult,
                    temperature=self.config.temperature,
                )

                result = completion.choices[0].message.parsed

                if result and result.questions:
                    # Convert Pydantic models to ExtractedQuestion
                    for pyd_q in result.questions:
                        # Map to question type if possible
                        q_type = None
                        if pyd_q.question_text.endswith('?'):
                            q_type = QuestionType.DIRECT_QUESTION
                        elif any(verb in pyd_q.question_text.lower() for verb in ['describe', 'explain', 'provide', 'list', 'identify']):
                            q_type = QuestionType.IMPERATIVE_REQUEST

                        question = ExtractedQuestion(
                            question_text=pyd_q.question_text,
                            original_number=pyd_q.original_number,
                            question_type=q_type,
                            category=pyd_q.category,
                            confidence=pyd_q.confidence,
                            metadata={
                                'extraction_method': 'simple_llm',
                                'chunk_index': i
                            }
                        )
                        all_questions.append(question)

                rate_limit(self.config.rate_limit_delay)

            except Exception as e:
                logger.error(f"Failed to extract from chunk {i+1}: {str(e)}")
                continue

        # Filter by confidence threshold
        all_questions = [
            q for q in all_questions
            if q.confidence >= self.config.confidence_threshold
        ]

        # Deduplicate if configured
        if self.config.deduplicate_questions:
            all_questions = deduplicate_questions(
                all_questions,
                similarity_threshold=self.config.similarity_threshold
            )

        logger.info(f"Extracted {len(all_questions)} questions using simple LLM approach")

        return all_questions
