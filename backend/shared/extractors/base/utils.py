"""
Shared utility functions for extraction strategies.

This module provides common functionality used across different extractors,
such as text chunking, deduplication, and similarity detection.
"""

import re
import time
import logging
from typing import List, Tuple, Set, Optional
from .models import ExtractedQuestion, ExtractedQAPair

logger = logging.getLogger(__name__)


# =============================================================================
# TEXT CHUNKING
# =============================================================================

def chunk_text(text: str, chunk_size: int = 8000, overlap: int = 500) -> List[str]:
    """ Split text into overlapping chunks. """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move forward, accounting for overlap
        start += (chunk_size - overlap)

    return chunks


def chunk_text_by_paragraphs(
    text: str,
    chunk_size: int = 8000,
    overlap: int = 500
) -> List[str]:
    """
    Split text into chunks respecting paragraph boundaries.

    This is smarter than simple character-based chunking as it avoids
    splitting in the middle of sentences or paragraphs. """
    if not text:
        return []

    # Split into paragraphs
    paragraphs = text.split('\n')

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_len = len(para) + 1  # +1 for newline

        # If adding this paragraph exceeds chunk_size and we have content, finalize chunk
        if current_size + para_len > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))

            # Start new chunk with overlap (keep last few paragraphs)
            overlap_paras = []
            overlap_size = 0
            for p in reversed(current_chunk):
                if overlap_size + len(p) < overlap:
                    overlap_paras.insert(0, p)
                    overlap_size += len(p)
                else:
                    break

            current_chunk = overlap_paras
            current_size = overlap_size

        # Add paragraph to current chunk
        current_chunk.append(para)
        current_size += para_len

    # Add final chunk if any content remains
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks


# =============================================================================
# DEDUPLICATION
# =============================================================================

def deduplicate_questions(
    questions: List[ExtractedQuestion],
    similarity_threshold: float = 0.9
) -> List[ExtractedQuestion]:
    """ Remove duplicate questions using similarity matching. """
    if not questions:
        return []

    unique_questions = []
    seen_texts = set()

    for q in questions:
        # Normalize text for comparison
        normalized = normalize_text(q.question_text)

        # Check if we've seen this exact text
        if normalized in seen_texts:
            continue

        # Check similarity with existing questions
        is_duplicate = False
        for existing in unique_questions:
            if text_similarity(q.question_text, existing.question_text) >= similarity_threshold:
                # Keep the one with higher confidence
                if q.confidence > existing.confidence:
                    unique_questions.remove(existing)
                    seen_texts.discard(normalize_text(existing.question_text))
                    break
                else:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_questions.append(q)
            seen_texts.add(normalized)

    return unique_questions


def deduplicate_qa_pairs(
    qa_pairs: List[ExtractedQAPair],
    similarity_threshold: float = 0.9
) -> List[ExtractedQAPair]:
    """ Remove duplicate Q&A pairs using similarity matching. """
    if not qa_pairs:
        return []

    unique_pairs = []
    seen_texts = set()

    for pair in qa_pairs:
        # Normalize text for comparison
        normalized = normalize_text(pair.question_text)

        # Check if we've seen this exact text
        if normalized in seen_texts:
            continue

        # Check similarity with existing pairs
        is_duplicate = False
        for existing in unique_pairs:
            if text_similarity(pair.question_text, existing.question_text) >= similarity_threshold:
                # Keep the one with higher confidence or with an answer
                if pair.has_answer and not existing.has_answer:
                    unique_pairs.remove(existing)
                    seen_texts.discard(normalize_text(existing.question_text))
                    break
                elif pair.confidence > existing.confidence and pair.has_answer == existing.has_answer:
                    unique_pairs.remove(existing)
                    seen_texts.discard(normalize_text(existing.question_text))
                    break
                else:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_pairs.append(pair)
            seen_texts.add(normalized)

    return unique_pairs


# =============================================================================
# TEXT SIMILARITY
# =============================================================================

def normalize_text(text: str) -> str:
    """ Normalize text for comparison (whitespace, lowecase, punctuation) """
    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove common punctuation at the end
    text = text.rstrip('?.!:;,')

    return text


def text_similarity(text1: str, text2: str) -> float:
    """ Calculate similarity between two texts using simple word overlap. """
    # Normalize texts
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)

    # Exact match
    if t1 == t2:
        return 1.0

    # Convert to word sets
    words1 = set(t1.split())
    words2 = set(t2.split())

    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


# =============================================================================
# PATTERN MATCHING
# =============================================================================

def extract_question_number(text: str) -> Tuple[Optional[str], str]:
    """
    Extract question numbering from text.

    Handles various numbering formats:
    - "1.1 Question text"
    - "Q1: Question text"
    - "Question 5. Question text"
    - "[1] Question text"

    Args:
        text: The text to parse

    Returns:
        Tuple of (number, text_without_number)
    """
    # Pattern 1: "1.1" or "1.1.1" at start
    match = re.match(r'^(\d+(?:\.\d+)*)[.\s:]+(.+)$', text)
    if match:
        return match.group(1), match.group(2).strip()

    # Pattern 2: "Q1" or "Question 1"
    match = re.match(r'^(?:Q|Question)\s*(\d+)[.\s:]+(.+)$', text, re.IGNORECASE)
    if match:
        return f"Q{match.group(1)}", match.group(2).strip()

    # Pattern 3: "[1]" at start
    match = re.match(r'^\[(\d+)\]\s*(.+)$', text)
    if match:
        return match.group(1), match.group(2).strip()

    # No number found
    return None, text

# =============================================================================
# RATE LIMITING
# =============================================================================

def rate_limit(delay: float = 0.2):
    """ Simple rate limiting via sleep. """
    time.sleep(delay)


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def log_extraction_stats(
    extractor_name: str,
    num_items: int,
    processing_time: float,
    api_calls: int = 0
):
    """
    Log extraction statistics.

    Args:
        extractor_name: Name of the extractor
        num_items: Number of items extracted
        processing_time: Time taken in seconds
        api_calls: Number of API calls made
    """
    logger.info(
        f"{extractor_name} extracted {num_items} items in {processing_time:.2f}s "
        f"({api_calls} API calls)"
    )


# =============================================================================
# EXTRACTOR ADAPTERS
# =============================================================================

class QuestionToQAAdapter:
    """
    Adapter that wraps a QuestionExtractor to make it compatible with DocumentProcessor.

    Converts ExtractedQuestion objects to ExtractedQAPair objects with answer_text=None.
    This allows DocumentProcessor to work with both question-only and Q&A extraction
    without code duplication.

    Usage:
        question_extractor = factory.create_question_extractor("context_aware")
        qa_extractor = QuestionToQAAdapter(question_extractor)
        processor = DocumentProcessor(session, parser, extractor=qa_extractor)
    """

    def __init__(self, question_extractor):
        """ Initialize with a QuestionExtractor instance. """
        self.question_extractor = question_extractor

    @property
    def strategy_name(self) -> str:
        """Return the wrapped extractor's strategy name."""
        return f"question_to_qa_adapter({self.question_extractor.strategy_name})"

    def extract(self, document_text: str, context=None) -> List[ExtractedQAPair]:
        """ Extract questions and convert them to Q&A pairs with null answers. """
        # Extract questions using the wrapped extractor
        questions: List[ExtractedQuestion] = self.question_extractor.extract(document_text, context)

        # Convert to Q&A pairs with null answers
        qa_pairs = [
            ExtractedQAPair(
                question_text=q.question_text,
                answer_text=None,
                original_number=q.original_number,
                question_type=q.question_type,
                category=q.category,
                confidence=q.confidence,
                section_id=q.section_id,
                metadata=q.metadata
            )
            for q in questions
        ]

        return qa_pairs

    def set_structure_extractor(self, extractor):
        """ Pass through structure extractor injection to the wrapped question extractor. """
        if hasattr(self.question_extractor, 'set_structure_extractor'):
            self.question_extractor.set_structure_extractor(extractor)
