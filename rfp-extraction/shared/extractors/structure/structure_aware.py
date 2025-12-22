"""
Structure-Aware Document Structure Extractor.

This extractor uses AI-powered analysis to detect document structure across
diverse formats. It samples multiple parts of the document, analyzes format
patterns, and identifies sections with their types and extraction priorities.

Adapted from the AIStructureDetector in shared/services/semantic_1.py.
"""

import re
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from openai import AzureOpenAI

from ..base import (
    DocumentStructureExtractor,
    DocumentStructure,
    DocumentSection,
    SectionType,
    DocumentFormat,
    ExtractionPriority,
    ExtractionContext,
    StructureExtractorConfig
)

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR LLM STRUCTURED OUTPUT
# =============================================================================

class DocumentFormatAnalysis(BaseModel):
    """Analysis of the document's overall format and structure."""
    detected_format: DocumentFormat = Field(..., description="Primary format of the document")
    format_confidence: float = Field(..., ge=0, le=1, description="Confidence in format detection")

    # Structure patterns detected
    numbering_pattern: Optional[str] = Field(None, description="Pattern like '1.1.1', 'Q1', 'Section A'")
    header_style: Optional[str] = Field(None, description="How headers are formatted (caps, bold markers, etc)")

    # Document metadata
    document_type: str = Field(..., description="RFP, DDQ, Security Questionnaire, etc.")
    issuing_organization: Optional[str] = Field(None, description="Organization name if detected")
    document_title: Optional[str] = Field(None, description="Document title if detected")

    # Structure characteristics
    has_table_of_contents: bool = Field(False, description="Whether TOC was detected")
    has_clear_sections: bool = Field(..., description="Whether document has clear section divisions")
    estimated_section_count: int = Field(..., description="Rough estimate of major sections")

    # Special characteristics
    has_response_fields: bool = Field(..., description="Whether response placeholders are present")
    response_field_pattern: Optional[str] = Field(None, description="Pattern of response fields")

    # Analysis notes
    structure_notes: str = Field(..., description="Notes about the document structure")
    potential_challenges: List[str] = Field(default_factory=list, description="Potential extraction challenges")


class DetectedSectionModel(BaseModel):
    """A section detected by AI analysis (Pydantic model for LLM)."""
    section_id: str = Field(..., description="Unique identifier for this section")

    # Location
    start_marker: str = Field(..., description="Text that marks the start of this section")
    approximate_start_position: str = Field(..., description="Description of where this section starts")

    # Identification
    section_number: Optional[str] = Field(None, description="Section number if present")
    section_title: str = Field(..., description="Section title or heading")

    # Classification
    section_type: SectionType = Field(..., description="Type of content in this section")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in this detection")

    # Content indicators
    contains_questions: bool = Field(..., description="Whether section contains extractable questions")
    estimated_question_count: int = Field(0, description="Rough estimate of questions in section")
    question_indicators: List[str] = Field(default_factory=list, description="Patterns indicating questions")

    # Hierarchy
    level: int = Field(1, description="Hierarchy level (1=top, 2=subsection, etc)")
    parent_section_id: Optional[str] = Field(None, description="Parent section if this is a subsection")

    # Extraction guidance
    extraction_priority: ExtractionPriority = Field(..., description="Priority for extraction")
    extraction_notes: str = Field("", description="Notes to guide extraction from this section")


class SectionListResponse(BaseModel):
    """Container for list of detected sections."""
    sections: List[DetectedSectionModel]


# =============================================================================
# LLM PROMPTS
# =============================================================================

FORMAT_ANALYSIS_PROMPT = """You are an expert document analyst specializing in procurement documents (RFPs, DDQs, questionnaires).

Analyze this document sample to understand its structure and format. This document could be:
- A government RFP with formal section numbering
- A corporate due diligence questionnaire (DDQ)
- A security/compliance questionnaire
- A vendor assessment form
- Or any other procurement-related document

Focus on:
1. **Format Detection**: How is the document structured? (numbered sections, question lists, tables, etc.)
2. **Numbering Patterns**: What numbering scheme is used? (1.1.1, Q1, Section A, etc.)
3. **Section Identification**: Are there clear section divisions?
4. **Question Indicators**: How are questions formatted? (ending in ?, imperative statements, response fields)
5. **Response Fields**: How are response areas marked? ([Enter here], ___, checkboxes, tables)

Be specific about patterns you observe. This analysis will guide extraction from the full document."""


SECTION_DETECTION_PROMPT = """You are mapping the structure of a procurement document (RFP/DDQ/Questionnaire).

Based on the format analysis provided and the document samples, identify ALL major sections in this document.

FORMAT CONTEXT:
{format_context}

For each section you identify:
1. Provide a unique ID
2. Note the text that marks its start (exact words that begin the section)
3. Classify its type (questionnaire, informational, instructions, etc.)
4. Assess whether it contains extractable questions
5. Assign extraction priority:
   - HIGH: Definite questionnaire sections with questions to answer
   - MEDIUM: May contain questions (mixed content, appendices)
   - LOW: Unlikely to have questions but worth checking
   - SKIP: Definitely informational only (terms, instructions, background)

IMPORTANT:
- Identify sections at the appropriate granularity (not too fine, not too coarse)
- A "questionnaire" section may have subsections - identify the main questionnaire container
- Look for ALL questionnaire-type sections, not just ones labeled "questionnaire"
- Tables with response columns are questionnaires
- Sections titled "Requirements" often contain questions despite the name"""


# =============================================================================
# STRUCTURE-AWARE STRUCTURE EXTRACTOR
# =============================================================================

class StructureAwareStructureExtractor(DocumentStructureExtractor):
    """
    AI-powered document structure detection that handles diverse formats.

    Unlike rule-based detection, this approach:
    1. Samples multiple parts of the document
    2. Uses AI to understand the document's specific format
    3. Detects sections regardless of formatting conventions
    4. Provides confidence scores and extraction guidance
    """

    def __init__(self, config: Optional[StructureExtractorConfig] = None):
        """
        Initialize the extractor.

        Args:
            config: Configuration for the extractor
        """
        self.config = config or StructureExtractorConfig()

        # Validate configuration
        if not self.config.validate():
            logger.warning("LLM configuration incomplete. Extraction may fail.")

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.config.llm_endpoint,
            api_key=self.config.llm_api_key,
            api_version=self.config.llm_api_version
        )

        # Statistics
        self.stats = {
            "api_calls": 0,
            "tokens_used": 0,
            "sections_detected": 0
        }

    @property
    def strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "structure_aware"

    def extract(
        self,
        document_text: str,
        context: Optional[ExtractionContext] = None
    ) -> DocumentStructure:
        """
        Extract document structure from text.

        This is a multi-step process:
        1. Sample the document strategically
        2. Analyze format and structure patterns
        3. Detect and classify sections
        4. Refine boundaries
        5. Generate extraction guidance

        Args:
            document_text: The full text of the document
            context: Optional extraction context

        Returns:
            DocumentStructure with detected sections and metadata
        """
        if not document_text or not document_text.strip():
            logger.warning("Empty document text provided")
            return DocumentStructure()

        logger.info("Starting AI-powered structure detection...")

        # Step 1: Strategic sampling
        samples = self._create_strategic_samples(document_text)
        logger.info(f"Created {len(samples)} strategic samples")

        # Step 2: Analyze document format
        format_analysis = self._analyze_document_format(samples)
        logger.info(
            f"Format detected: {format_analysis.detected_format.value} "
            f"(confidence: {format_analysis.format_confidence:.2f})"
        )

        # Step 3: Detect sections
        detected_sections = self._detect_sections(samples, format_analysis)
        logger.info(f"Detected {len(detected_sections)} sections")

        # Step 4: Refine boundaries with actual positions
        refined_sections = self._refine_section_boundaries(detected_sections, document_text)

        # Step 5: Build final structure
        structure = self._build_document_structure(format_analysis, refined_sections)

        self.stats["sections_detected"] = len(structure.sections)
        logger.info(f"Structure detection complete. Sections: {len(structure.sections)}")

        return structure

    def _create_strategic_samples(self, text: str) -> List[Dict[str, Any]]:
        """
        Create strategic samples from different parts of the document.

        Sampling strategy:
        - Beginning: Catch title, TOC, introduction
        - Early-middle: Often where questionnaires start
        - Middle: Core content
        - Late-middle: Additional questionnaire sections
        - End: Appendices, final sections
        """
        text_length = len(text)
        sample_size = self.config.sample_size
        samples = []

        # Calculate sample positions
        positions = [
            ("beginning", 0),
            ("early", int(text_length * 0.2)),
            ("middle", int(text_length * 0.45)),
            ("late", int(text_length * 0.7)),
            ("end", max(0, text_length - sample_size))
        ]

        for name, start in positions:
            end = min(start + sample_size, text_length)

            # Try to start/end at paragraph boundaries
            adjusted_start = self._find_paragraph_boundary(text, start, direction="backward")
            adjusted_end = self._find_paragraph_boundary(text, end, direction="forward")

            sample_text = text[adjusted_start:adjusted_end]

            samples.append({
                "name": name,
                "start": adjusted_start,
                "end": adjusted_end,
                "text": sample_text,
                "position_percent": (adjusted_start / text_length) * 100
            })

        return samples

    def _find_paragraph_boundary(self, text: str, pos: int, direction: str = "forward") -> int:
        """Find the nearest paragraph boundary."""
        if direction == "forward":
            # Look for next double newline or end
            match = re.search(r'\n\s*\n', text[pos:])
            if match:
                return pos + match.end()
            return min(pos + 200, len(text))
        else:
            # Look for previous double newline or start
            search_text = text[:pos]
            match = re.search(r'\n\s*\n(?!.*\n\s*\n)', search_text)
            if match:
                return match.end()
            return max(0, pos - 200)

    def _analyze_document_format(self, samples: List[Dict[str, Any]]) -> DocumentFormatAnalysis:
        """
        Analyze the document's format using AI.

        Examines multiple samples to understand:
        - Document type and structure
        - Numbering conventions
        - Section patterns
        - Question/response formats
        """
        self.stats["api_calls"] += 1

        # Combine samples for analysis
        combined_sample = "\n\n".join([
            f"=== SAMPLE FROM {s['name'].upper()} ({s['position_percent']:.0f}% into document) ===\n{s['text']}"
            for s in samples[:3]  # Use first 3 samples for format analysis
        ])

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.config.llm_deployment,
                messages=[
                    {"role": "system", "content": FORMAT_ANALYSIS_PROMPT},
                    {"role": "user", "content": f"Analyze this document's format and structure:\n\n{combined_sample}"}
                ],
                response_format=DocumentFormatAnalysis,
                temperature=self.config.structure_analysis_temperature
            )

            if hasattr(completion, 'usage') and completion.usage:
                self.stats["tokens_used"] += completion.usage.total_tokens

            time.sleep(self.config.rate_limit_delay)
            return completion.choices[0].message.parsed

        except Exception as e:
            logger.error(f"Format analysis failed: {e}")
            # Return a safe default
            return DocumentFormatAnalysis(
                detected_format=DocumentFormat.UNSTRUCTURED,
                format_confidence=0.3,
                document_type="Unknown",
                has_clear_sections=False,
                estimated_section_count=5,
                has_response_fields=True,
                structure_notes="Format analysis failed, using fallback",
                potential_challenges=["Could not analyze document format"]
            )

    def _detect_sections(
        self,
        samples: List[Dict[str, Any]],
        format_analysis: DocumentFormatAnalysis
    ) -> List[DetectedSectionModel]:
        """Detect all sections in the document using AI."""
        self.stats["api_calls"] += 1

        # Build format context
        format_context = f"""
Document Type: {format_analysis.document_type}
Format: {format_analysis.detected_format.value}
Numbering Pattern: {format_analysis.numbering_pattern or 'Not detected'}
Header Style: {format_analysis.header_style or 'Not detected'}
Has TOC: {format_analysis.has_table_of_contents}
Has Clear Sections: {format_analysis.has_clear_sections}
Estimated Sections: {format_analysis.estimated_section_count}
Response Field Pattern: {format_analysis.response_field_pattern or 'Not detected'}
Structure Notes: {format_analysis.structure_notes}
"""

        # Use all samples for section detection
        combined_samples = "\n\n".join([
            f"=== {s['name'].upper()} SECTION ({s['position_percent']:.0f}% into document) ===\n{s['text']}"
            for s in samples
        ])

        prompt = SECTION_DETECTION_PROMPT.format(format_context=format_context)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.config.llm_deployment,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Identify all sections in this document:\n\n{combined_samples}"}
                ],
                response_format=SectionListResponse,
                temperature=self.config.section_detection_temperature
            )

            if hasattr(completion, 'usage') and completion.usage:
                self.stats["tokens_used"] += completion.usage.total_tokens

            time.sleep(self.config.rate_limit_delay)
            return completion.choices[0].message.parsed.sections

        except Exception as e:
            logger.error(f"Section detection failed: {e}")
            return self._fallback_section_detection(samples)

    def _fallback_section_detection(self, samples: List[Dict[str, Any]]) -> List[DetectedSectionModel]:
        """
        Fallback section detection when AI detection fails.
        Creates a simple structure based on document chunks.
        """
        logger.warning("Using fallback section detection")

        sections = []

        # Create sections based on samples
        for i, sample in enumerate(samples):
            sections.append(DetectedSectionModel(
                section_id=f"fallback_section_{i+1}",
                start_marker=sample['text'][:50],
                approximate_start_position=f"{sample['position_percent']:.0f}% into document",
                section_title=f"Section {i+1} ({sample['name']})",
                section_type=SectionType.UNKNOWN,
                confidence=0.3,
                contains_questions=True,  # Assume yes for fallback
                extraction_priority=ExtractionPriority.MEDIUM,
                extraction_notes="Fallback detection - process with caution"
            ))

        return sections

    def _refine_section_boundaries(
        self,
        sections: List[DetectedSectionModel],
        full_text: str
    ) -> List[Tuple[DetectedSectionModel, int, int]]:
        """
        Refine section boundaries to get exact character positions.

        Uses string matching for section markers.
        """
        refined = []
        text_lower = full_text.lower()

        for section in sections:
            # Try to find the section start marker
            start_marker = section.start_marker.lower().strip()

            # Try exact match first
            start_pos = text_lower.find(start_marker)

            if start_pos == -1:
                # Try partial match (first significant words)
                words = start_marker.split()[:5]
                partial_marker = ' '.join(words)
                start_pos = text_lower.find(partial_marker)

            if start_pos == -1:
                # Try title match
                title_lower = section.section_title.lower()
                start_pos = text_lower.find(title_lower)

            if start_pos == -1:
                logger.warning(f"Could not find section: {section.section_title}")
                # Use approximate position
                if "beginning" in section.approximate_start_position.lower():
                    start_pos = 0
                elif "end" in section.approximate_start_position.lower():
                    start_pos = int(len(full_text) * 0.8)
                else:
                    # Try to parse percentage
                    pct_match = re.search(r'(\d+)%', section.approximate_start_position)
                    if pct_match:
                        start_pos = int(len(full_text) * int(pct_match.group(1)) / 100)
                    else:
                        start_pos = 0

            refined.append((section, start_pos, None))  # End will be calculated

        # Sort by start position
        refined.sort(key=lambda x: x[1])

        # Calculate end positions (start of next section)
        result = []
        for i, (section, start, _) in enumerate(refined):
            if i + 1 < len(refined):
                end = refined[i + 1][1]
            else:
                end = len(full_text)

            # Ensure minimum section length
            if end - start < self.config.min_section_length:
                continue

            result.append((section, start, end))

        return result

    def _build_document_structure(
        self,
        format_analysis: DocumentFormatAnalysis,
        refined_sections: List[Tuple[DetectedSectionModel, int, int]]
    ) -> DocumentStructure:
        """Build the final document structure."""
        # Convert Pydantic models to our DocumentSection dataclass
        sections = []
        for pydantic_section, start, end in refined_sections:
            section = DocumentSection(
                section_id=pydantic_section.section_id,
                section_number=pydantic_section.section_number,
                section_title=pydantic_section.section_title,
                section_type=pydantic_section.section_type,
                start_char=start,
                end_char=end,
                confidence=pydantic_section.confidence,
                extraction_priority=pydantic_section.extraction_priority,
                metadata={
                    "contains_questions": pydantic_section.contains_questions,
                    "estimated_question_count": pydantic_section.estimated_question_count,
                    "question_indicators": pydantic_section.question_indicators,
                    "level": pydantic_section.level,
                    "parent_section_id": pydantic_section.parent_section_id,
                    "extraction_notes": pydantic_section.extraction_notes
                }
            )
            sections.append(section)

        # Build structure
        return DocumentStructure(
            document_title=format_analysis.document_title,
            document_type=format_analysis.document_type,
            document_format=format_analysis.detected_format,
            sections=sections,
            format_confidence=format_analysis.format_confidence,
            metadata={
                "numbering_pattern": format_analysis.numbering_pattern,
                "header_style": format_analysis.header_style,
                "has_table_of_contents": format_analysis.has_table_of_contents,
                "has_response_fields": format_analysis.has_response_fields,
                "response_field_pattern": format_analysis.response_field_pattern,
                "structure_notes": format_analysis.structure_notes,
                "potential_challenges": format_analysis.potential_challenges,
                "issuing_organization": format_analysis.issuing_organization
            }
        )
