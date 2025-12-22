"""
Database models package.

Exports all models and provides initialization utilities.
"""

from sqlalchemy import text

from .base import Base, create_db_engine, get_session_maker
from .rfp import RFP
from .answer import Answer
from .document import Document
from .chunk import Chunk

__all__ = [
    'Base',
    'RFP',
    'Answer',
    'Document',
    'Chunk',
    'create_db_engine',
    'get_session_maker',
    'init_database',
]


def init_database(database_url: str, echo: bool = False):
    """
    Initialize database: create tables and enable extensions.

    Args:
        database_url: PostgreSQL connection string
        echo: Whether to echo SQL statements

    Returns:
        SQLAlchemy engine
    """
    engine = create_db_engine(database_url, echo=echo)

    # Enable pgvector extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(engine)

    return engine
