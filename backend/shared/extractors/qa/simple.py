"""
Simple Q+A Extractor.

This extractor implements a straightforward approach to extracting question-answer pairs:
1. Chunk the document into manageable pieces
2. Use LLM to extract Q&A pairs from each chunk
3. Deduplicate and return results

This is adapted from the original SemanticAnalyzer in shared/services/semantic.py.
"""

import logging
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from shared.llm import create_llm_client

from ..base import (
    QuestionAnswerExtractor,
    ExtractedQAPair,
    ExtractionContext,
    QAExtractorConfig,
    chunk_text,
    deduplicate_qa_pairs,
    rate_limit
)

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class PydanticQAPair(BaseModel):
    """Pydantic model for Q&A pair (for LLM structured output)."""
    question_text: str = Field(..., description="The complete question text, including all sub-questions if present.")
    answer_text: Optional[str] = Field(None, description="The answer text if present (for previous RFPs).")
    category: Optional[str] = Field(None, description="The domain or section topic (e.g., 'Security', 'Legal').")
    original_number: Optional[str] = Field(None, description="The original question number (e.g., '8.1.1', 'Q5')")


class SemanticExtractionResult(BaseModel):
    """Container for the list of items found in a chunk."""
    qa_pairs: List[PydanticQAPair]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
# Role
You are an expert Document Parsing AI specialized in Request for Proposals (RFPs) and Due Diligence Questionnaires (DDQs). Your objective is to parse raw, messy OCR text into a structured JSON format.

**CRITICAL REQUIREMENT - Textual Fidelity:**
- Extract question and answer text VERBATIM from the document
- Do NOT rephrase, rewrite, or paraphrase
- Preserve the original wording, punctuation, and structure exactly as written
- Only remove obvious OCR artifacts (page headers, footers, page numbers) AND markdown formatting syntax

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
- **Horizontal rules:** `---` or `***`

## Critical Rules for Markdown Text

1. **Ignore Markdown Syntax - Focus on Content**
   - Markdown symbols (`**`, `#`, `|`, `-`, `*`) are FORMATTING ONLY
   - Extract the semantic content, not the formatting markers
   - Bold/italic/headers do NOT change the meaning of text

2. **Bold/Headers Are NOT Automatically Questions**
   - `**Does your platform...?**` → This IS a question (has question mark)
   - `**TECHNICAL REQUIREMENTS**` → This is a HEADER, NOT a question
   - `# 3 Requirements` → This is a section header, NOT a question
   - Look for question marks or imperative verbs, NOT formatting

3. **Clean Output - NO Markdown in Extracted Text**
   - Input: `**8.1.1** Does your platform allow multiple accounts?`
   - Output question_text: `Does your platform allow multiple accounts?` (remove `**`)
   - Output original_number: `8.1.1` (remove `**`)
   - Input: `**Account Structure:** Describe your approach.`
   - Output question_text: `Describe your approach.` (remove `**` and category prefix)

4. **Table Format Questions**
   - Questions may appear in markdown tables with `| Question | Answer |` format
   - Extract content from table cells, ignore `|` and `---` separators
   - Input: `| Describe your security measures. | We implement 2FA... |`
   - Output: question_text: `Describe your security measures.`, answer_text: `We implement 2FA...`

5. **Links**
   - Extract link text, ignore URL
   - Input: `See our [privacy policy](https://example.com)`
   - Extract: `See our privacy policy`

## Format Examples

### Example 1: Plain Text (Azure Document Intelligence)
```
LONDON & PARTNERS
REQUEST FOR PROPOSAL (RFP)
8.1.1 Account Structure. Does your platform allow multiple user accounts?
Enter response here:
```

### Example 2: Markdown Format (pymupdf4llm, etc.)
```
**LONDON & PARTNERS**

**REQUEST FOR PROPOSAL (RFP)**

**8.1.1 Account Structure.** Does your platform allow multiple user accounts?

**Enter response here:**
```

### Example 3: Markdown Header (NOT a question)
```
## 3 TECHNICAL REQUIREMENTS

This section covers technical specifications.
```
**Action:** SKIP - This is a section header, not a question

### Example 4: Markdown Table
```
| Question | Answer |
|----------|--------|
| Describe your security approach. | We use industry-standard encryption. |
```
**Action:** EXTRACT as question + answer, remove `|` characters

# Context & Input Data
You will be processing text chunks from PDFs that may be parsed by:
- **Azure Document Intelligence:** Plain text with OCR artifacts
- **Open-source parsers (pymupdf4llm, PyPDF, etc.):** Markdown-formatted text
- **OCR Noise:** The text contains artifacts like page headers (e.g., "LONDON & PARTNERS"), page numbers, and footers inserted randomly in the middle of sentences. Remove these artifacts but DO NOT rephrase the actual content.
- **Structure:** The document contains varying sections. Some are **Informational** (Instructions, Terms, Legal Definitions) and some are **Questionnaires** (Technical Requirements, Pricing, Compliance) requiring supplier input.

# Core Objectives
1. **Analyze Structure:** Use the Table of Contents (if visible) or Section Headers to determine your current location in the document.
2. **Classify Intent:** For every numbered item, determine if it is:
   - `INFORMATIONAL`: A legal term, instruction, or deadline (e.g., "2.1 Closing Date"). -> **IGNORE OR TAG AS INFO.**
   - `QUESTION`: A specific request for information, confirmation, or description from the supplier (e.g., "8.1.1 Does your system offer..."). -> **EXTRACT VERBATIM.**
3. **Extract Content:** Remove OCR artifacts and extract the question and answer text exactly as written (verbatim).

# Global Rules

### 1. Handling Sections (The "Section Type" Logic)
You must explicitly categorize the section you are in.
- **Information Sections:** Introduction, Specification, Functional Requirements, Commercial Submission, Form of Tender, Special Conditions, Instructions to Tenderers, Background.
  - *Action:* Do NOT extract numbered lists here as questions. These are terms.
- **Questionnaire Sections:** Procurement Questionnaire.
  - *Action:* Extract all numbered items and imperative statements as questions.

### 2. Identifying Questions
A "Question" is defined as any text that:
- Ends in a question mark.
- Uses imperative language requiring a response (e.g., "Describe your methodology...", "Provide details of...", "Confirm that...").
- Is an item in a questionnaire section.

**IMPORTANT - Handling Sub-Questions:**
- If a numbered item (e.g., "8.2.5") contains MULTIPLE sub-questions, combine them into a SINGLE question entry
- Keep all sub-questions together under the same original_number
- Example: "8.2.5 Platform scope - Reporting: What is the platform's ability to cover removals and suppressions? What is the platform's ability to cover integration with Google Analytics?"
  Should be ONE entry with original_number="8.2.5" and question_text containing both questions joined together
- Do NOT split sub-questions into separate entries with duplicate numbers

### 3. Handling Answers
- **Unanswered:** If the text says "Enter response here", "[Insert details]", or is blank -> Set Answer to `null`.
- **Answered:** If there is text immediately following a question that appears to be written by a supplier -> Extract as `answer_text`.

### 4. Handling OCR Noise
- **Headers/Footers:** If a sentence is split by "LONDON & PARTNERS" or a page number, join the sentence parts.
- **Broken Numbers:** If "8.1. \n 7" appears, treat it as "8.1.7".

# Step-by-Step Processing Instructions (Chain of Thought)
1. **Scan for Section Headers:** Look for bolded, capitalized headers (e.g., "2 SPECIAL CONDITIONS", "8 PROCUREMENT QUESTIONNAIRE"). Update your internal state regarding the "Current Section".
2. **Evaluate Item:** When you encounter a numbered item (e.g., "2.1" or "8.1.1"):
   - Ask: "Does this require the supplier to write something?"
   - If YES -> Extract as Question.
   - If NO (it's just a rule/date) -> Mark as `requires_response: false`.
3. **Format Output:** JSON only.

# Few-Shot Examples

**Example 1: Informational Section (Do NOT extract as Question) - Plain Text**
*Input:* "2.1 Closing Date. Responses must be submitted by Friday. LONDON & PARTNERS 2.2 Information provided..."
*Reasoning:* This is Section 2 "Special Conditions". "Closing Date" is a rule, not a question for the supplier.
*Output:* `{"item_type": "informational", "text": "Closing Date...", "requires_response": false}`

**Example 2: Questionnaire Section (Extract) - Plain Text**
*Input:* "8.1.3 Account Structure. Does your platform allow multiple user accounts? Enter response here:"
*Reasoning:* This is Section 8. It asks "Does your platform...". It requires input.
*Output:* `{"item_type": "question", "original_number": "8.1.3", "question_text": "Does your platform allow multiple user accounts?", "answer": null, "requires_response": true}`

**Example 3: Imperative Question - Plain Text**
*Input:* "Describe your approach to dedicated IP warming."
*Reasoning:* No question mark, but "Describe" is an imperative command in a requirement section.
*Output:* `{"item_type": "question", "question_text": "Describe your approach to dedicated IP warming.", "requires_response": true}`

**Example 4: Multiple Sub-Questions Under Same Number (KEEP TOGETHER) - Plain Text**
*Input:* "8.2.5 Platform scope - Reporting. What is the platform's ability to cover removals and suppressions? What is the platform's ability to cover integration with Google Analytics?"
*Reasoning:* This is question 8.2.5 with TWO sub-questions. They share the same number, so combine into ONE entry.
*Output:* `{"item_type": "question", "original_number": "8.2.5", "question_text": "Platform scope - Reporting. What is the platform's ability to cover removals and suppressions? What is the platform's ability to cover integration with Google Analytics?", "answer": null, "requires_response": true}`

**Example 5: Markdown Bold Question (Extract, Remove Formatting)**
*Input:* "**8.1.1 Account Structure.** Does your platform allow multiple user accounts?"
*Reasoning:* This is a question (ends with ?). The `**` is just markdown bold formatting - strip it from output.
*Output:* `{"item_type": "question", "original_number": "8.1.1", "question_text": "Does your platform allow multiple user accounts?", "answer": null, "requires_response": true}`

**Example 6: Markdown Header (NOT a Question)**
*Input:* "## 3 TECHNICAL REQUIREMENTS"
*Reasoning:* This is a section header with `##` markdown. It's not a question - no question mark, no imperative verb.
*Output:* SKIP - Do not extract

**Example 7: Markdown Table with Q&A**
*Input:* "| Describe your security measures. | We implement 2FA and encryption. |"
*Reasoning:* Markdown table format. "Describe" is imperative. Answer provided in second column.
*Output:* `{"item_type": "question", "question_text": "Describe your security measures.", "answer_text": "We implement 2FA and encryption.", "requires_response": true}`

**Example 8: Markdown Bold Header vs Question**
*Input:* "**Does your platform support API integration?**"
*Reasoning:* Even though it's bold, it ends with a question mark - it IS a question. Remove `**` from output.
*Output:* `{"item_type": "question", "question_text": "Does your platform support API integration?", "requires_response": true}`

### 3. OUTPUT FORMAT
- Return a list of items.
- Strictly maintain `original_numbering` - ONE entry per unique question number
- If a number has multiple sub-questions, combine them into a single entry
"""


# =============================================================================
# SIMPLE Q+A EXTRACTOR
# =============================================================================

class SimpleQAExtractor(QuestionAnswerExtractor):
    """
    Simple Q+A extraction strategy.

    Uses basic text chunking and direct LLM prompting to extract
    question-answer pairs. This is the baseline approach.
    """

    def __init__(self, config: Optional[QAExtractorConfig] = None):
        """ Initialize the extractor. """
        self.config = config or QAExtractorConfig()

        # Validate configuration
        if not self.config.validate():
            logger.warning("LLM configuration incomplete. Extraction may fail.")

        # Initialize LLM client
        self.client = create_llm_client(self.config)

    @property
    def strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "simple"

    def extract(
        self,
        document_text: str,
        context: Optional[ExtractionContext] = None
    ) -> List[ExtractedQAPair]:
        """
        Extract Q&A pairs from document text.

        Args:
            document_text: The full text of the document
            context: Optional extraction context (not used in simple strategy)

        Returns:
            List of ExtractedQAPair objects
        """
        if not document_text or not document_text.strip():
            logger.warning("Empty document text provided")
            return []

        # Chunk the text
        chunks = chunk_text(
            document_text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap
        )

        logger.info(
            f"Split document into {len(chunks)} chunks for extraction "
            f"(chunk_size={self.config.chunk_size}, overlap={self.config.overlap})"
        )

        # Extract Q&A pairs from each chunk
        all_qa_pairs: List[ExtractedQAPair] = []

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}...")

            try:
                # Call LLM with structured output
                completion = self.client.chat_completions_parse(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Extract Q&A pairs from this text:\n\n{chunk}"}
                    ],
                    response_format=SemanticExtractionResult,
                    temperature=self.config.temperature,
                )

                result = completion.choices[0].message.parsed

                if result and result.qa_pairs:
                    # Convert Pydantic models to our dataclass models
                    for pydantic_pair in result.qa_pairs:
                        qa_pair = ExtractedQAPair(
                            question_text=pydantic_pair.question_text,
                            answer_text=pydantic_pair.answer_text,
                            original_number=pydantic_pair.original_number,
                            category=pydantic_pair.category
                        )
                        all_qa_pairs.append(qa_pair)

                # Rate limiting
                rate_limit(self.config.rate_limit_delay)

            except Exception as e:
                logger.error(f"Failed to process chunk {i+1}: {str(e)}")
                # Continue to next chunk rather than failing completely
                continue

        logger.info(f"Extracted {len(all_qa_pairs)} Q&A pairs (before deduplication)")

        # Deduplicate if configured
        if self.config.deduplicate_pairs:
            all_qa_pairs = deduplicate_qa_pairs(
                all_qa_pairs,
                similarity_threshold=self.config.similarity_threshold
            )
            logger.info(f"After deduplication: {len(all_qa_pairs)} Q&A pairs")

        return all_qa_pairs
