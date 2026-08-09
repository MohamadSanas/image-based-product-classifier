"""
persistence/database.py — SQLAlchemy ORM setup for SQLite.

Defines the ORM models (tables) and provides a Database class that
manages engine creation and session lifecycle.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Generator

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class ScanRecord(Base):
    """ORM model representing one scan session stored in the database."""

    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String(512), nullable=False)
    total_products = Column(Integer, nullable=False, default=0)
    # category_counts stored as JSON: [{"label": "apple", "count": 3, "percentage": 30.0}, ...]
    category_counts_json = Column(Text, nullable=False, default="[]")
    # detected_products stored as JSON array for full traceability
    detected_products_json = Column(Text, nullable=False, default="[]")
    scan_duration_ms = Column(Float, nullable=True)
    scan_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<ScanRecord id={self.id} path={self.image_path!r} "
            f"total={self.total_products} ts={self.scan_timestamp}>"
        )

    @property
    def category_counts(self) -> list[dict]:
        return json.loads(self.category_counts_json)

    @property
    def detected_products(self) -> list[dict]:
        return json.loads(self.detected_products_json)


class Database:
    """
    Manages SQLAlchemy engine and session factory.

    Usage
    -----
    >>> db = Database("sqlite:///output/scan_history.db")
    >>> with db.session() as session:
    ...     session.add(...)
    """

    def __init__(self, database_url: str = "sqlite:///output/scan_history.db") -> None:
        db_path = database_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},  # SQLite threading
            echo=False,
        )
        self._SessionFactory = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._create_tables()
        logger.info("Database initialised at: %s", database_url)

    def _create_tables(self) -> None:
        Base.metadata.create_all(self._engine)

    def session(self) -> Generator[Session, None, None]:
        """Context manager that provides a database session."""
        session = self._SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @property
    def engine(self):
        return self._engine

    @classmethod
    def from_config(cls, cfg: dict) -> "Database":
        db_path = cfg["paths"].get("database", "output/scan_history.db")
        return cls(f"sqlite:///{db_path}")
