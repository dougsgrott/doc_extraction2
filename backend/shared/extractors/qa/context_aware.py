"""
Context-Aware Q+A Extractor.

This extractor enhances Q+A extraction with deeper contextual understanding.
It uses the context-aware question extractor internally and pairs questions
with answers using contextual analysis.

Approach:
1. Uses context-aware question extraction
2. Attempts to match questions with nearby answers using context
3. Provides richer metadata about the extraction
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from shared.llm import create_llm_client

from ..base import (
    QuestionAnswerExtractor,
    ExtractedQAPair,
    ExtractedQuestion,
    ExtractionContext,
    QAExtractorConfig,
    deduplicate_qa_pairs,
    rate_limit
)
from ..question.context_aware import ContextAwareQuestionExtractor
from ..base import QuestionExtractorConfig

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR ANSWER EXTRACTION
# =============================================================================

class AnswerExtractionResult(BaseModel):
    """Result of answer extraction for a question."""
    answer_found: bool = Field(..., description="Whether an answer was found")
    answer_text: Optional[str] = Field(None, description="The extracted answer text (VERBATIM)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the answer extraction")


SYSTEM_PROMPT = """You are an expert at identifying answers to questions in procurement documents.

Given a question and surrounding text, determine if an answer is present and extract it.

# Input Format Handling

The input text may be in **plain text** OR **markdown format**, depending on the PDF parsing tool used.

## Markdown Syntax Guide
Common markdown formatting you may encounter:
- **Bold:** `**text**` or `__text__`
- **Italic:** `*text*` or `_text_`
- **Headers:** `#` (h1), `##` (h2), `###` (h3), etc.
- **Tables:** `| col1 | col2 |` with `|---|---|` separators
- **Lists:** `- item` or `* item` or `1. item`
- **Links:** `[text](url)`

## Critical Rules for Markdown Text

1. **Ignore Markdown Syntax - Focus on Content**
   - Markdown symbols (`**`, `#`, `|`, `-`, `*`) are FORMATTING ONLY
   - Extract the semantic content, not the formatting markers

2. **Clean Output - NO Markdown in Extracted Text**
   - Input: `**We implement 2FA and encryption.**`
   - Output: `We implement 2FA and encryption.` (remove `**`)
   - Input: `| Our security approach is... |`
   - Output: `Our security approach is...` (remove `|`)

3. **Table Format Answers**
   - Answers may appear in markdown tables with `| Question | Answer |` format
   - Extract content from table cells, ignore `|` and `---` separators
   - Input: `| Describe security. | We use 2FA. |`
   - Extract answer: `We use 2FA.`

4. **Links**
   - Extract link text, ignore URL
   - Input: `See our [privacy policy](https://example.com)`
   - Extract: `See our privacy policy`

CRITICAL - Textual Fidelity:
- Extract answers VERBATIM from the document
- Do NOT rephrase or rewrite
- Preserve exact wording, punctuation, and structure
- Remove markdown formatting syntax (`**`, `#`, `|`, etc.)
- Remove OCR artifacts (page headers/footers, page numbers)

An answer may be:
- Explicit text following the question
- A filled-in response field
- Text in a table cell (markdown format: `| answer |`)
- A checkbox or selection
- May be blank/empty (answer_found=false)

Common patterns indicating NO answer:
- "Enter response here"
- "[Insert details]"
- Blank or empty fields
- `**Enter response here:**` (markdown bold placeholder)

If no answer is present (e.g., blank response field, unanswered question), set answer_found=false."""

class ContextAwareQAExtractor(QuestionAnswerExtractor):
    """
    Context-aware Q+A extraction.

    Combines context-aware question extraction with answer detection
    to produce high-quality Q+A pairs.
    """

    def __init__(self, config: Optional[QAExtractorConfig] = None):
        """Initialize the extractor."""
        self.config = config or QAExtractorConfig()

        if not self.config.validate():
            logger.warning("LLM configuration incomplete. Extraction may fail.")

        self.client = create_llm_client(self.config)

        # Store structure for debugging/auditing
        self.structure = None

    @property
    def strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "context_aware"

    def extract(
        self,
        document_text: str,
        context: Optional[ExtractionContext] = None
    ) -> List[ExtractedQAPair]:
        """
        Extract Q+A pairs using context-aware analysis.

        Args:
            document_text: The full text of the document
            context: Optional extraction context

        Returns:
            List of ExtractedQAPair objects
        """
        if not document_text or not document_text.strip():
            logger.warning("Empty document text provided")
            return []

        logger.info("Starting context-aware Q+A extraction...")

        # Get document structure first
        if context and context.has_structure():
            self.structure = context.get_structure()
        elif hasattr(self, 'structure_extractor') and self.structure_extractor:
            logger.info("Detecting document structure for answer extraction...")
            self.structure = self.structure_extractor.extract(document_text)
        else:
            self.structure = None

        # Step 1: Extract questions using context-aware question extractor
        question_config = QuestionExtractorConfig(
            llm_endpoint=self.config.llm_endpoint,
            llm_api_key=self.config.llm_api_key,
            llm_api_version=self.config.llm_api_version,
            llm_deployment=self.config.llm_deployment,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            confidence_threshold=self.config.confidence_threshold,
            allowed_section_types=self.config.allowed_section_types
        )

        question_extractor = ContextAwareQuestionExtractor(question_config)

        # Set structure extractor if available
        if hasattr(self, 'structure_extractor') and self.structure_extractor:
            question_extractor.set_structure_extractor(self.structure_extractor)

        questions = question_extractor.extract(document_text, context)

        logger.info(f"Extracted {len(questions)} questions, now attempting to find answers...")

        # Step 2: Attempt to find answers for each question
        qa_pairs = self._match_questions_with_answers(questions, document_text, self.structure)

        # Deduplicate if configured
        if self.config.deduplicate_pairs:
            qa_pairs = deduplicate_qa_pairs(
                qa_pairs,
                similarity_threshold=self.config.similarity_threshold
            )

        logger.info(f"Produced {len(qa_pairs)} Q+A pairs")

        return qa_pairs

    def _match_questions_with_answers(
        self,
        questions: List[ExtractedQuestion],
        document_text: str,
        structure=None
    ) -> List[ExtractedQAPair]:
        """
        Attempt to match questions with answers in the document using LLM.

        For each question, extracts surrounding context and uses LLM to identify
        if an answer is present.
        """
        qa_pairs = []

        # Build section map for quick lookup
        section_map = {}
        if structure and structure.sections:
            for section in structure.sections:
                section_map[section.section_id] = section

        for i, question in enumerate(questions):
            logger.debug(f"Finding answer for question {i+1}/{len(questions)}: {question.question_text[:50]}...")

            # Extract context around the question using section boundaries if available
            context = self._extract_question_context(question, document_text, section_map)

            if not context:
                # No context found, create pair without answer
                pair = ExtractedQAPair(
                    question_text=question.question_text,
                    answer_text=None,
                    original_number=question.original_number,
                    category=question.category,
                    confidence=question.confidence,
                    section_id=question.section_id,
                    metadata=question.metadata
                )
                qa_pairs.append(pair)
                continue

            # Use LLM to extract answer from context
            answer_result = self._extract_answer_from_context(question, context)

            # Create QA pair with extracted answer
            pair = ExtractedQAPair(
                question_text=question.question_text,
                answer_text=answer_result.answer_text if answer_result.answer_found else None,
                original_number=question.original_number,
                category=question.category,
                confidence=min(question.confidence, answer_result.confidence),
                section_id=question.section_id,
                metadata={
                    **question.metadata,
                    'answer_found': answer_result.answer_found,
                    'answer_confidence': answer_result.confidence
                }
            )

            qa_pairs.append(pair)

            # Rate limiting
            if i > 0 and i % 5 == 0:
                rate_limit(self.config.rate_limit_delay)

        return qa_pairs

    def _extract_question_context(
        self,
        question: ExtractedQuestion,
        document_text: str,
        section_map: dict = None
    ) -> Optional[str]:
        """
        Extract context around a question for answer detection.

        Uses section boundaries if available, otherwise tries to find question text.
        """
        # Strategy 1: Use section boundaries if available
        if section_map and question.section_id and question.section_id in section_map:
            section = section_map[question.section_id]
            context = document_text[section.start_char:section.end_char]
            logger.debug(f"Using section {section.section_id} as context ({len(context)} chars)")
            return context

        # Strategy 2: Try to find the question text in the document
        question_lower = question.question_text.lower().strip()[:100]  # Use first 100 chars for matching
        doc_lower = document_text.lower()

        # Try to find question position
        pos = doc_lower.find(question_lower)

        if pos == -1:
            # Try partial match (first significant words)
            words = question_lower.split()[:10]
            partial = ' '.join(words)
            pos = doc_lower.find(partial)

        if pos == -1:
            logger.debug(f"Could not locate question in document: {question.question_text[:50]}...")
            # As a last resort, use the full document (not ideal but better than nothing)
            logger.debug("Using full document as context (fallback)")
            return document_text

        # Extract context (question + following text)
        context_start = max(0, pos - 200)  # Some text before for context
        context_end = min(len(document_text), pos + 2000)  # Generous amount after
        context = document_text[context_start:context_end]

        return context

    def _extract_answer_from_context(
        self,
        question: ExtractedQuestion,
        context: str
    ) -> AnswerExtractionResult:
        """
        Use LLM to extract answer from context.

        Args:
            question: The question to find an answer for
            context: Text containing the question and potential answer

        Returns:
            AnswerExtractionResult with answer if found
        """
        try:
            completion = self.client.chat_completions_parse(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Question: {question.question_text}

Context (containing question and potential answer):
{context}

Extract the answer if present."""
                    }
                ],
                response_format=AnswerExtractionResult,
                temperature=0.0,  # Deterministic for answer extraction
            )

            result = completion.choices[0].message.parsed
            return result

        except Exception as e:
            logger.error(f"Failed to extract answer: {str(e)}")
            return AnswerExtractionResult(
                answer_found=False,
                answer_text=None,
                confidence=0.0
            )

    def set_structure_extractor(self, extractor):
        """Inject a structure extractor dependency."""
        self.structure_extractor = extractor
