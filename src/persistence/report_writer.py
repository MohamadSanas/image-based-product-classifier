"""
persistence/report_writer.py — Save scan results as JSON, CSV, or plain text.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from src.core.entities import ScanResult

logger = logging.getLogger(__name__)


class ReportWriter:
    """
    Writes ScanResult data to disk in multiple formats.

    Parameters
    ----------
    output_dir : str
        Directory where reports are saved.
    """

    def __init__(self, output_dir: str = "output/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, result: ScanResult, filename: str | None = None) -> Path:
        """Serialise a ScanResult to a JSON file."""
        filename = filename or self._auto_filename(result, "json")
        path = self.output_dir / filename

        data = {
            "id": result.id,
            "image_path": result.image_path,
            "scan_timestamp": result.scan_timestamp.isoformat(),
            "scan_duration_ms": result.scan_duration_ms,
            "total_products": result.total_products,
            "category_counts": [
                {"label": c.label, "count": c.count, "percentage": c.percentage}
                for c in result.category_counts
            ],
            "detected_products": [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "top_k_predictions": d.top_k_predictions,
                    "bounding_box": d.bounding_box.as_xywh(),
                }
                for d in result.detected_products
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("JSON report saved: %s", path)
        return path

    def write_csv(self, result: ScanResult, filename: str | None = None) -> Path:
        """Write category-count summary as a CSV file."""
        filename = filename or self._auto_filename(result, "csv")
        path = self.output_dir / filename

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "count", "percentage"])
            writer.writeheader()
            for c in result.category_counts:
                writer.writerow({
                    "label": c.label,
                    "count": c.count,
                    "percentage": c.percentage,
                })

        logger.info("CSV report saved: %s", path)
        return path

    def write_text(self, result: ScanResult, filename: str | None = None) -> Path:
        """Write a human-readable plain text summary."""
        from src.analytics.statistics import StatisticsEngine

        filename = filename or self._auto_filename(result, "txt")
        path = self.output_dir / filename

        engine = StatisticsEngine()
        summary = engine.summary_text(result.category_counts, result.total_products)
        header = (
            f"Smart Supermarket — Scan Report\n"
            f"{'=' * 40}\n"
            f"Image      : {result.image_path}\n"
            f"Timestamp  : {result.scan_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Duration   : {result.scan_duration_ms:.1f} ms\n"
            f"{'=' * 40}\n\n"
        )

        path.write_text(header + summary + "\n", encoding="utf-8")
        logger.info("Text report saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_filename(result: ScanResult, ext: str) -> str:
        ts = result.scan_timestamp.strftime("%Y%m%d_%H%M%S")
        scan_id = result.id or "unsaved"
        return f"scan_{scan_id}_{ts}.{ext}"
