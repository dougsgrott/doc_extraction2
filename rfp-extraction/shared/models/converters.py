"""
Converters between extraction models and database models.

This module provides functions to convert between:
1. Extraction models (ExtractedQuestion, ExtractedQAPair) -> DB models (Answer)
2. Extraction enums -> DB enums
3. DB models -> Pydantic schemas (handled by Pydantic's from_attributes)

This creates a clean separation between the extraction layer (business logic)
and the persistence layer (database).
"""

from typing import Optional, List
import uuid

from shared.extractors.base.models import (
    ExtractedQuestion,
    ExtractedQAPair,
    QuestionType,
    ConfidenceLevel
)
from .db.answer import Answer
from .enums import AnswerStatus, SourceType


# =============================================================================
# EXTRACTION -> DATABASE CONVERSIONS
# =============================================================================

def extracted_question_to_answer(
    question: ExtractedQuestion,
    rfp_id: Optional[uuid.UUID] = None,
    source: SourceType = SourceType.RFP_EXTRACTION,
    sequence: Optional[int] = None
) -> Answer:
    """
    Convert ExtractedQuestion to Answer database model.

    Args:
        question: Extracted question from document
        rfp_id: Optional RFP ID to link to
        source: Source of the question
        sequence: Optional sequence number in RFP

    Returns:
        Answer model (unsaved, needs to be added to session)
    """
    return Answer(
        rfp_id=rfp_id,
        question_text=question.question_text,
        answer_text=None,  # No answer for ExtractedQuestion
        status=AnswerStatus.PENDING,
        category=question.category,
        section=question.section_id,  # Map section_id to section
        sequence=sequence,
        tags=_extract_tags_from_question_metadata(question),
        source=source,
        # question_embedding will be generated separately (if needed)
    )


def extracted_qa_to_answer(
    qa_pair: ExtractedQAPair,
    rfp_id: Optional[uuid.UUID] = None,
    source: SourceType = SourceType.RFP_EXTRACTION,
    sequence: Optional[int] = None
) -> Answer:
    """
    Convert ExtractedQAPair to Answer database model.

    Args:
        qa_pair: Extracted Q&A pair from document
        rfp_id: Optional RFP ID to link to
        source: Source of the Q&A pair
        sequence: Optional sequence number in RFP

    Returns:
        Answer model (unsaved, needs to be added to session)
    """
    # Determine status based on whether answer exists
    if qa_pair.has_answer:
        # If from knowledge base, mark as APPROVED; if from RFP extraction, mark as ANSWERED
        status = AnswerStatus.APPROVED if source == SourceType.KNOWLEDGE_BASE else AnswerStatus.ANSWERED
    else:
        status = AnswerStatus.PENDING

    return Answer(
        rfp_id=rfp_id,
        question_text=qa_pair.question_text,
        answer_text=qa_pair.answer_text,
        status=status,
        category=qa_pair.category,
        section=qa_pair.section_id,
        sequence=sequence,
        tags=_extract_tags_from_qa_metadata(qa_pair),
        source=source,
    )


def bulk_extracted_qa_to_answers(
    qa_pairs: List[ExtractedQAPair],
    rfp_id: Optional[uuid.UUID] = None,
    source: SourceType = SourceType.RFP_EXTRACTION
) -> List[Answer]:
    """
    Convert a list of ExtractedQAPair to Answer models with sequence numbers.

    Args:
        qa_pairs: List of extracted Q&A pairs
        rfp_id: Optional RFP ID to link to
        source: Source of the pairs

    Returns:
        List of Answer models (unsaved)
    """
    answers = []
    for i, qa_pair in enumerate(qa_pairs, start=1):
        answer = extracted_qa_to_answer(
            qa_pair,
            rfp_id=rfp_id,
            source=source,
            sequence=i
        )
        answers.append(answer)
    return answers


def bulk_extracted_questions_to_answers(
    questions: List[ExtractedQuestion],
    rfp_id: Optional[uuid.UUID] = None,
    source: SourceType = SourceType.RFP_EXTRACTION
) -> List[Answer]:
    """
    Convert a list of ExtractedQuestion to Answer models with sequence numbers.

    Args:
        questions: List of extracted questions
        rfp_id: Optional RFP ID to link to
        source: Source of the questions

    Returns:
        List of Answer models (unsaved)
    """
    answers = []
    for i, question in enumerate(questions, start=1):
        answer = extracted_question_to_answer(
            question,
            rfp_id=rfp_id,
            source=source,
            sequence=i
        )
        answers.append(answer)
    return answers


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_tags_from_question_metadata(question: ExtractedQuestion) -> Optional[List[str]]:
    """Extract tags from ExtractedQuestion metadata."""
    tags = []

    # Add question type as tag
    if question.question_type:
        tags.append(f"type:{question.question_type.value}")

    # Add original number as tag
    if question.original_number:
        tags.append(f"num:{question.original_number}")

    # Add confidence level
    tags.append(f"confidence:{question.confidence_level.value}")

    # Add category if present
    if question.category:
        tags.append(f"category:{question.category}")

    # Add any custom metadata
    if question.metadata:
        for key, value in question.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                tags.append(f"{key}:{value}")

    return tags if tags else None


def _extract_tags_from_qa_metadata(qa_pair: ExtractedQAPair) -> Optional[List[str]]:
    """Extract tags from ExtractedQAPair metadata."""
    tags = []

    # Add question type as tag
    if qa_pair.question_type:
        tags.append(f"type:{qa_pair.question_type.value}")

    # Add original number as tag
    if qa_pair.original_number:
        tags.append(f"num:{qa_pair.original_number}")

    # Add confidence level
    tags.append(f"confidence:{qa_pair.confidence_level.value}")

    # Add category if present
    if qa_pair.category:
        tags.append(f"category:{qa_pair.category}")

    # Add answered status
    tags.append(f"answered:{qa_pair.has_answer}")

    # Add any custom metadata
    if qa_pair.metadata:
        for key, value in qa_pair.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                tags.append(f"{key}:{value}")

    return tags if tags else None


# =============================================================================
# DATABASE -> EXTRACTION CONVERSIONS (for backward compatibility)
# =============================================================================

def answer_to_extracted_qa(answer: Answer) -> ExtractedQAPair:
    """
    Convert Answer database model back to ExtractedQAPair.

    Useful for feeding historical data back into extraction pipeline
    or for testing purposes.

    Args:
        answer: Answer database model

    Returns:
        ExtractedQAPair
    """
    return ExtractedQAPair(
        question_text=answer.question_text,
        answer_text=answer.answer_text,
        category=answer.category,
        original_number=_extract_original_number_from_tags(answer.tags),
        confidence=_estimate_confidence_from_status(answer.status),
        section_id=answer.section,
        metadata={'source': answer.source.value, 'status': answer.status.value}
    )


def answer_to_extracted_question(answer: Answer) -> ExtractedQuestion:
    """
    Convert Answer database model to ExtractedQuestion.

    Args:
        answer: Answer database model

    Returns:
        ExtractedQuestion
    """
    return ExtractedQuestion(
        question_text=answer.question_text,
        category=answer.category,
        original_number=_extract_original_number_from_tags(answer.tags),
        confidence=_estimate_confidence_from_status(answer.status),
        section_id=answer.section,
        metadata={'source': answer.source.value, 'status': answer.status.value}
    )


def _extract_original_number_from_tags(tags: Optional[List[str]]) -> Optional[str]:
    """Extract original number from tags."""
    if not tags:
        return None
    for tag in tags:
        if tag.startswith('num:'):
            return tag[4:]  # Remove 'num:' prefix
    return None


def _estimate_confidence_from_status(status: AnswerStatus) -> float:
    """
    Estimate confidence score from answer status.

    This is a heuristic mapping for backward compatibility.
    """
    confidence_map = {
        AnswerStatus.APPROVED: 1.0,
        AnswerStatus.ANSWERED: 0.8,
        AnswerStatus.IN_PROGRESS: 0.6,
        AnswerStatus.PENDING: 0.5,
        AnswerStatus.REJECTED: 0.3,
    }
    return confidence_map.get(status, 0.5)
