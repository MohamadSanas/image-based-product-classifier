"""
persistence/repositories.py — CRUD operations for ScanResult persistence.

Implements IScanRepository using SQLAlchemy + ScanRecord ORM model.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from src.core.entities import BoundingBox, CategoryCount, DetectedProduct, ScanResult
from src.core.interfaces import IScanRepository
from src.persistence.database import Database, ScanRecord

logger = logging.getLogger(__name__)


class ScanRepository(IScanRepository):
    """
    Concrete scan history repository backed by SQLite via SQLAlchemy.

    Parameters
    ----------
    database : Database
        Shared Database instance providing session factory.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    # ------------------------------------------------------------------
    # IScanRepository implementation
    # ------------------------------------------------------------------

    def save(self, result: ScanResult) -> ScanResult:
        """Persist a ScanResult and return it with its db-assigned id."""
        record = ScanRecord(
            image_path=result.image_path,
            total_products=result.total_products,
            category_counts_json=self._serialise_counts(result.category_counts),
            detected_products_json=self._serialise_detections(result.detected_products),
            scan_duration_ms=result.scan_duration_ms,
            scan_timestamp=result.scan_timestamp,
        )
        with self._db.session() as session:
            session.add(record)
            session.flush()  # Populate record.id
            result.id = record.id

        logger.info("Saved ScanResult id=%d to database.", result.id)
        return result

    def get_by_id(self, scan_id: int) -> Optional[ScanResult]:
        with self._db.session() as session:
            record: Optional[ScanRecord] = session.get(ScanRecord, scan_id)
            if record is None:
                return None
            return self._record_to_domain(record)

    def get_all(self, limit: int = 50, offset: int = 0) -> list[ScanResult]:
        with self._db.session() as session:
            records = (
                session.query(ScanRecord)
                .order_by(ScanRecord.scan_timestamp.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._record_to_domain(r) for r in records]

    def delete(self, scan_id: int) -> bool:
        with self._db.session() as session:
            record = session.get(ScanRecord, scan_id)
            if record is None:
                return False
            session.delete(record)
        logger.info("Deleted ScanRecord id=%d.", scan_id)
        return True

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_counts(counts: list[CategoryCount]) -> str:
        return json.dumps([
            {"label": c.label, "count": c.count, "percentage": c.percentage}
            for c in counts
        ])

    @staticmethod
    def _serialise_detections(detected: list[DetectedProduct]) -> str:
        return json.dumps([
            {
                "label": d.label,
                "confidence": d.confidence,
                "bounding_box": {
                    "x": d.bounding_box.x, "y": d.bounding_box.y,
                    "width": d.bounding_box.width, "height": d.bounding_box.height,
                },
                "top_k_predictions": d.top_k_predictions,
            }
            for d in detected
        ])

    @staticmethod
    def _record_to_domain(record: ScanRecord) -> ScanResult:
        """Reconstruct a ScanResult domain object from an ORM record."""
        raw_counts = json.loads(record.category_counts_json)
        category_counts = [
            CategoryCount(label=c["label"], count=c["count"], percentage=c["percentage"])
            for c in raw_counts
        ]

        raw_detections = json.loads(record.detected_products_json)
        detected_products = [
            DetectedProduct(
                label=d["label"],
                confidence=d["confidence"],
                bounding_box=BoundingBox(
                    x=d["bounding_box"]["x"],
                    y=d["bounding_box"]["y"],
                    width=d["bounding_box"]["width"],
                    height=d["bounding_box"]["height"],
                ),
                top_k_predictions=d.get("top_k_predictions", []),
            )
            for d in raw_detections
        ]

        return ScanResult(
            id=record.id,
            image_path=record.image_path,
            detected_products=detected_products,
            category_counts=category_counts,
            total_products=record.total_products,
            scan_timestamp=record.scan_timestamp,
            scan_duration_ms=record.scan_duration_ms,
        )
