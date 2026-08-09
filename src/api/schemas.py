"""
api/schemas.py — Pydantic request and response models.

Used for API input validation and response serialisation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Shared sub-models
# ------------------------------------------------------------------

class BoundingBoxSchema(BaseModel):
    x: int
    y: int
    width: int
    height: int


class PredictionSchema(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class DetectedProductSchema(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBoxSchema
    top_k_predictions: list[PredictionSchema] = []


class CategoryCountSchema(BaseModel):
    label: str
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0.0, le=100.0)


# ------------------------------------------------------------------
# Classify endpoint
# ------------------------------------------------------------------

class ClassifyResponse(BaseModel):
    """Response returned by POST /api/v1/classify."""
    scan_id: Optional[int] = None
    image_source: str
    total_products: int
    scan_duration_ms: Optional[float] = None
    category_counts: list[CategoryCountSchema]
    detected_products: list[DetectedProductSchema]


# ------------------------------------------------------------------
# History endpoints
# ------------------------------------------------------------------

class ScanSummary(BaseModel):
    """Lightweight record for history list views."""
    id: int
    image_path: str
    total_products: int
    scan_timestamp: datetime
    scan_duration_ms: Optional[float] = None
    category_counts: list[CategoryCountSchema]


class HistoryResponse(BaseModel):
    """Response for GET /api/v1/history."""
    total: int
    limit: int
    offset: int
    items: list[ScanSummary]


class DeleteResponse(BaseModel):
    """Response for DELETE /api/v1/history/{id}."""
    success: bool
    message: str


# ------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
