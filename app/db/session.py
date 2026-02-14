"""
Database session management.

NOTE: This project uses ChromaDB (vector database) as primary storage.
      SQLAlchemy / PostgreSQL is NOT used.
      This module is kept as a stub to avoid import errors from legacy code.
"""

# Stub Base for any legacy model imports
try:
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()
except ImportError:
    Base = object  # type: ignore


def get_db():
    """Legacy stub – not used in the ChromaDB architecture."""
    raise NotImplementedError("This project uses ChromaDB, not PostgreSQL.")
