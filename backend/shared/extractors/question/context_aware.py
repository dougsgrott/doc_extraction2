"""
Context-Aware Question Extractor.

This extractor uses document structure and context injection to improve question extraction.
It chunks the document respecting section boundaries and injects contextual information
about the current section, surrounding sections, and extraction guidance.
"""

import logging
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from shared.llm import create_llm_client

from .models import PydanticQuestion, QuestionExtractionResult
from ..base import (
    QuestionExtractor,
    ExtractedQuestion,
    ExtractionContext,
    QuestionExtractorConfig,
    DocumentSection,
    SectionType,
    QuestionType,
    chunk_text_by_paragraphs,
    deduplicate_questions,
    rate_limit
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT-AWARE QUESTION EXTRACTOR
# =============================================================================

SYSTEM_PROMPT = """You are an expert at extracting questions from procurement documents. Use the provided context to guide your extraction.

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

2. **Bold/Headers Are NOT Automatically Questions**
   - `**Does your platform...?**` → This IS a question (has question mark)
   - `**TECHNICAL REQUIREMENTS**` → This is a HEADER, NOT a question
   - `# 3 Requirements` → This is a section header, NOT a question
   - `## Security` → This is a header, NOT a question
   - Look for question marks or imperative verbs, NOT formatting

3. **Clean Output - NO Markdown in Extracted Text**
   - Input: `**8.1.1** Does your platform allow multiple accounts?`
   - Output question_text: `Does your platform allow multiple accounts?` (remove `**`)
   - Output original_number: `8.1.1` (remove `**`)

4. **Table Format Questions**
   - Questions may appear in markdown tables
   - Extract content from table cells, ignore `|` and `---` separators

## Format Examples

### Example 1: Plain Text
```
8.1.1 What is your company name?
```
Output: question_text: "What is your company name?", original_number: "8.1.1"

### Example 2: Markdown Bold Question
```
**8.1 Describe your approach to security**
```
Output: question_text: "Describe your approach to security", original_number: "8.1" (no `**`)

### Example 3: Markdown Header (NOT a question)
```
## 3 TECHNICAL REQUIREMENTS
```
Output: SKIP - This is a section header, not a question

CRITICAL - Extract Questions ONLY, Not Statements:
- Extract questions VERBATIM as they appear in the document
- Remove markdown formatting syntax (`**`, `#`, `|`, etc.) from output
- Do NOT create questions from statements or descriptions
- Do NOT turn answers into questions
- Only extract text that is explicitly asking for information (ends with '?', starts with interrogatives, uses imperative verbs)

Examples of what to EXTRACT:
✓ "What is your company name?"
✓ "8.1 Describe your approach to security"
✓ "Please provide details about..."
✓ `**What is your company name?**` → Extract: "What is your company name?" (remove `**`)
✓ `| Describe your approach to security |` → Extract: "Describe your approach to security" (remove `|`)

Examples of what NOT to extract:
✗ "Secure image hosting included" (this is a statement/answer, NOT a question)
✗ `**Secure image hosting included**` (this is a bold statement, NOT a question)
✗ "Company has 100 employees" (statement, not a question)
✗ "ISO 27001 certified" (statement, not a question)
✗ `# 3 Requirements` (section header, not a question)
✗ `## Security Compliance` (section header, not a question)

If text is a statement or answer, do NOT convert it into a question."""

class ContextAwareQuestionExtractor(QuestionExtractor):
    """
    Context-aware question extraction strategy.

    This extractor uses document structure (if available) to provide
    rich context to the LLM, improving extraction accuracy. It respects
    section boundaries and provides section-specific extraction guidance.
    """

    def __init__(self, config: Optional[QuestionExtractorConfig] = None):
        """Initialize the extractor."""
        self.config = config or QuestionExtractorConfig()

        if not self.config.validate():
            logger.warning("LLM configuration incomplete. Extraction may fail.")

        self.client = create_llm_client(self.config)

        self.structure_extractor = None
        self.structure = None

    @property
    def strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "context_aware"

    def set_structure_extractor(self, extractor):
        """Inject a structure extractor for dependency."""
        self.structure_extractor = extractor

    def extract(
        self,
        document_text: str,
        context: Optional[ExtractionContext] = None
    ) -> List[ExtractedQuestion]:
        """
        Extract questions using context-aware chunking.

        Args:
            document_text: The full text of the document
            context: Optional extraction context with document structure

        Returns:
            List of ExtractedQuestion objects
        """
        if not document_text or not document_text.strip():
            logger.warning("Empty document text provided")
            return []

        # Get or detect document structure
        structure = None
        if context and context.has_structure():
            structure = context.get_structure()
        elif self.structure_extractor:
            logger.info("Detecting document structure...")
            structure = self.structure_extractor.extract(document_text)

        # Store structure for later retrieval (debugging/auditing)
        self.structure = structure

        # Extract questions with or without structure
        if structure and structure.sections:
            logger.info(f"Extracting with structure awareness ({len(structure.sections)} sections)")
            all_questions = self._extract_with_structure(document_text, structure)
        else:
            logger.info("Extracting without structure (fallback to chunking)")
            all_questions = self._extract_without_structure(document_text)

        # Deduplicate if configured
        if self.config.deduplicate_questions:
            all_questions = deduplicate_questions(
                all_questions,
                similarity_threshold=self.config.similarity_threshold
            )
            logger.info(f"After deduplication: {len(all_questions)} questions")

        return all_questions

    def _should_extract_from_section(self, section: DocumentSection) -> bool:
        """
        Determine if we should extract from this section based on config.

        Uses the allowed_section_types configuration to filter sections.
        - If allowed_section_types is an empty list [], extract from ALL section types
        - If allowed_section_types is a list with values, use it as a whitelist
        - If allowed_section_types is None (not set), use default behavior (QUESTIONNAIRE and PRICING only)
        """
        section_type = section.section_type

        # If allowed_section_types is an empty list, allow ALL sections
        if self.config.allowed_section_types is not None:
            # Empty list means allow all
            if len(self.config.allowed_section_types) == 0:
                return True
            # Non-empty list means whitelist
            return section_type in self.config.allowed_section_types

        # Default behavior when allowed_section_types is None (not configured)
        default_allowed = [
            SectionType.QUESTIONNAIRE,
            SectionType.PRICING
        ]

        return section_type in default_allowed

    def _extract_with_structure(
        self,
        document_text: str,
        structure
    ) -> List[ExtractedQuestion]:
        """Extract questions using document structure for context."""
        all_questions = []

        # Process high-priority sections first
        high_priority_sections = structure.get_high_priority_sections()
        other_sections = [s for s in structure.sections if s not in high_priority_sections]

        for sections_group in [high_priority_sections, other_sections]:
            for section in sections_group:
                # Check if we should extract from this section type
                if not self._should_extract_from_section(section):
                    logger.debug(f"Skipping section {section.section_id} (type: {section.section_type.value})")
                    continue

                # Skip sections marked to skip by priority
                if section.extraction_priority.value == "skip":
                    continue

                # Extract section text
                section_text = document_text[section.start_char:section.end_char]

                if not section_text.strip():
                    continue

                # Build context for this section
                context_info = self._build_section_context(section, structure)

                # Chunk the section if it's too large
                if len(section_text) > self.config.chunk_size:
                    chunks = chunk_text_by_paragraphs(
                        section_text,
                        chunk_size=self.config.chunk_size,
                        overlap=self.config.overlap
                    )
                else:
                    chunks = [section_text]

                # Extract from each chunk
                for i, chunk in enumerate(chunks):
                    chunk_context = f"{context_info}\n\nChunk {i+1} of {len(chunks)} in this section.\n"
                    questions = self._extract_from_chunk(chunk, chunk_context, section.section_id)
                    all_questions.extend(questions)

                logger.debug(f"Section {section.section_id}: extracted {len(questions)} questions")

        return all_questions

    def _extract_without_structure(self, document_text: str) -> List[ExtractedQuestion]:
        """Fallback extraction without structure (basic chunking)."""
        chunks = chunk_text_by_paragraphs(
            document_text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap
        )

        all_questions = []
        for i, chunk in enumerate(chunks):
            context = f"Processing document chunk {i+1} of {len(chunks)}.\n"
            questions = self._extract_from_chunk(chunk, context)
            all_questions.extend(questions)

        return all_questions

    def _build_section_context(self, section: DocumentSection, structure) -> str:
        """Build contextual information for a section."""
        lines = ["=== SECTION CONTEXT ==="]

        # Document info
        if structure.document_title:
            lines.append(f"Document: {structure.document_title}")
        if structure.document_type:
            lines.append(f"Type: {structure.document_type}")

        # Current section
        lines.append(f"\nCurrent Section: {section.section_title}")
        if section.section_number:
            lines.append(f"Section Number: {section.section_number}")
        lines.append(f"Section Type: {section.section_type.value}")
        lines.append(f"Extraction Priority: {section.extraction_priority.value}")

        # Extraction guidance
        if section.section_type.value == "questionnaire":
            lines.append("\nThis is a QUESTIONNAIRE section - extract all questions.")
        elif section.section_type.value == "pricing":
            lines.append("\nThis is a PRICING section - extract all questions.")
        elif section.section_type.value == "informational":
            lines.append("\nThis is an INFORMATIONAL section - do not extract questions.")
        elif section.section_type.value == "mixed":
            lines.append("\nThis is a MIXED section - do not extract questions.")
        else:
            # For other types (appendix, unknown, etc.)
            lines.append(f"\nThis is a {section.section_type.value.upper()} section - extract clear questions only.")

        return "\n".join(lines)

    def _extract_from_chunk(
        self,
        chunk: str,
        context_text: str,
        section_id: Optional[str] = None
    ) -> List[ExtractedQuestion]:
        """Extract questions from a single chunk with context."""
        try:
            # Build prompt with context
            prompt = f"{context_text}\n\nExtract all questions from the following text:\n\n{chunk}"

            completion = self.client.chat_completions_parse(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format=QuestionExtractionResult,
                temperature=self.config.temperature,
            )

            result = completion.choices[0].message.parsed

            if result and result.questions:
                # Convert Pydantic models to our dataclass
                questions = []
                for pydantic_q in result.questions:
                    # Map question_type string to enum if possible
                    q_type = None
                    if pydantic_q.question_type:
                        try:
                            q_type = QuestionType(pydantic_q.question_type.lower().replace(" ", "_"))
                        except ValueError:
                            pass

                    q = ExtractedQuestion(
                        question_text=pydantic_q.question_text,
                        original_number=pydantic_q.original_number,
                        question_type=q_type,
                        category=pydantic_q.category,
                        confidence=pydantic_q.confidence,
                        section_id=section_id
                    )
                    questions.append(q)

                return questions

            rate_limit(self.config.rate_limit_delay)

        except Exception as e:
            logger.error(f"Failed to extract from chunk: {str(e)}")

        return []
