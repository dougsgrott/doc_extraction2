"""
Database base configuration and utilities.

Provides the declarative base class, common column helpers, and database
engine factory functions.
"""

import os
import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.dialects.postgresql import UUID

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:password@localhost/rfp_db")


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


# =============================================================================
# COMMON COLUMN HELPERS
# =============================================================================

def uuid_pk() -> Mapped[uuid.UUID]:
    """ Primary key UUID column, auto-generated on insert. """
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def created_at() -> Mapped[datetime]:
    """ Created timestamp column, auto-set on insert. """
    return mapped_column(default=datetime.utcnow, nullable=False)


def updated_at() -> Mapped[datetime]:
    """ Updated timestamp column, auto-updated on modification """
    return mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )


# =============================================================================
# ENGINE FACTORY
# =============================================================================

def create_db_engine(url: Optional[str] = None, echo: bool = False):
    """ Create database engine with proper configuration. """
    return create_engine(url or DATABASE_URL, echo=echo)


def get_session_maker(engine):
    """ Get sessionmaker for the given engine. """
    return sessionmaker(bind=engine)
