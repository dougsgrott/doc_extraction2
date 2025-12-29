"""
Simple LLM-based Question Extractor.

This extractor uses basic chunking and straightforward LLM calls to extract questions.
No structure awareness, no multi-step processing - just simple extraction.

Approach:
1. Chunk document with simple overlap
2. Extract questions from each chunk using LLM
3. Deduplicate results
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from shared.llm import create_llm_client

from .models import PydanticQuestion, QuestionExtractionResult
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
# SIMPLE LLM QUESTION EXTRACTOR
# =============================================================================

SYSTEM_PROMPT = """You are an expert at extracting questions from procurement documents.

Extract all questions that require a response from the supplier/vendor.

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
   - Bold/italic/headers do NOT change the meaning of text

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
   - Input: `**Account Structure:** Describe your approach.`
   - Output question_text: `Describe your approach.` (remove `**` and category prefix)

4. **Table Format Questions**
   - Questions may appear in markdown tables
   - Extract content from table cells, ignore `|` and `---` separators
   - Input: `| Does your platform support SSO? | ... |`
   - Output question_text: `Does your platform support SSO?` (remove `|`)

5. **Links**
   - Extract link text, ignore URL
   - Input: `Describe [your approach](http://example.com)`
   - Extract: `Describe your approach`

## Format Examples

### Example 1: Plain Text
```
8.1.1 Account Structure. Does your platform allow multiple user accounts?
```
Output: question_text: "Does your platform allow multiple user accounts?", original_number: "8.1.1"

### Example 2: Markdown Bold Question
```
**8.1.1 Account Structure.** Does your platform allow multiple user accounts?
```
Output: question_text: "Does your platform allow multiple user accounts?", original_number: "8.1.1" (no `**`)

### Example 3: Markdown Header (NOT a question)
```
## 3 TECHNICAL REQUIREMENTS
```
Output: SKIP - This is a section header, not a question

### Example 4: Markdown Table
```
| Does your platform support API integration? |
```
Output: question_text: "Does your platform support API integration?" (no `|`)

CRITICAL RULES - Textual Fidelity:
- Extract question text VERBATIM from the document - do NOT rephrase or rewrite
- Preserve the original wording, punctuation, and structure exactly as written
- Remove markdown formatting syntax (`**`, `#`, `|`, etc.) from output
- Only clean up obvious OCR artifacts (page headers/footers, page numbers)
- Extract the question number exactly as it appears, without markdown (e.g., "2.1", "8.1.1", "Q5")

What to EXTRACT:
✓ Questions ending with '?'
✓ Imperative statements requesting information (Describe, Explain, Provide, List, etc.)
✓ Numbered items that clearly request information
✓ "Please provide...", "State whether...", etc.

What NOT to extract:
✗ Statements or descriptions (e.g., "Secure hosting included" or `**Secure hosting included**`)
✗ Instructions or guidelines
✗ Section headers (plain or markdown: "3 Requirements", `# 3 Requirements`, `## Security`)
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

        self.client = create_llm_client(self.config)

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
                completion = self.client.chat_completions_parse(
                    model=self.config.llm_model,
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
