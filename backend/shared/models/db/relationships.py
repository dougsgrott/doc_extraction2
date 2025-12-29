"""
Helper functions for managing relationships and denormalized counters.

Provides utilities for maintaining consistency between related entities,
particularly for RFP question counters.
"""

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from .rfp import RFP
from .answer import Answer
from ..enums import AnswerStatus


def update_rfp_counters(session: Session, rfp_id) -> None:
    """
    Update denormalized counters on RFP based on related answers.

    Recomputes:
    - questions_total: Total number of answers linked to this RFP
    - questions_answered: Number of answers with status ANSWERED or APPROVED
    - questions_approved: Number of answers with status APPROVED

    Args:
        session: SQLAlchemy session
        rfp_id: RFP ID to update
    """
    rfp = session.get(RFP, rfp_id)
    if not rfp:
        return

    # Get all answers for this RFP
    stmt = select(Answer).where(Answer.rfp_id == rfp_id)
    answers = session.execute(stmt).scalars().all()

    # Update counters
    rfp.questions_total = len(answers)
    rfp.questions_answered = sum(
        1 for a in answers
        if a.status in (AnswerStatus.ANSWERED, AnswerStatus.APPROVED)
    )
    rfp.questions_approved = sum(
        1 for a in answers
        if a.status == AnswerStatus.APPROVED
    )

    session.commit()


def create_answer_with_rfp_update(
    session: Session,
    answer: Answer,
    commit: bool = True
) -> Answer:
    """
    Create an answer and update parent RFP counters atomically.

    Args:
        session: SQLAlchemy session
        answer: Answer to create
        commit: Whether to commit the transaction (default: True)

    Returns:
        Created answer with ID populated
    """
    session.add(answer)
    session.flush()  # Get answer ID

    # Update RFP counters if answer is linked to an RFP
    if answer.rfp_id:
        update_rfp_counters(session, answer.rfp_id)

    if commit:
        session.commit()

    return answer


def bulk_create_answers_with_rfp_update(
    session: Session,
    answers: list[Answer],
    commit: bool = True
) -> list[Answer]:
    """
    Create multiple answers and update parent RFP counters efficiently.

    This is more efficient than calling create_answer_with_rfp_update multiple
    times because it updates counters only once at the end.

    Args:
        session: SQLAlchemy session
        answers: List of answers to create
        commit: Whether to commit the transaction (default: True)

    Returns:
        Created answers with IDs populated
    """
    # Add all answers
    session.add_all(answers)
    session.flush()  # Get all answer IDs

    # Update RFP counters for all affected RFPs
    rfp_ids = {answer.rfp_id for answer in answers if answer.rfp_id}
    for rfp_id in rfp_ids:
        update_rfp_counters(session, rfp_id)

    if commit:
        session.commit()

    return answers


def update_answer_status(
    session: Session,
    answer_id,
    new_status: AnswerStatus,
    answer_text: Optional[str] = None,
    commit: bool = True
) -> Answer:
    """
    Update answer status and optionally answer text, then update RFP counters.

    Args:
        session: SQLAlchemy session
        answer_id: Answer ID to update
        new_status: New status for the answer
        answer_text: Optional new answer text
        commit: Whether to commit the transaction (default: True)

    Returns:
        Updated answer

    Raises:
        ValueError: If answer not found
    """
    answer = session.get(Answer, answer_id)
    if not answer:
        raise ValueError(f"Answer with id {answer_id} not found")

    # Update answer
    answer.status = new_status
    if answer_text is not None:
        answer.answer_text = answer_text

    # Update RFP counters if linked to RFP
    if answer.rfp_id:
        update_rfp_counters(session, answer.rfp_id)

    if commit:
        session.commit()

    return answer
