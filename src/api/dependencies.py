"""
api/dependencies.py — FastAPI dependency injection providers.

Provides singleton-style access to shared services (pipeline, repository)
via FastAPI's Depends() mechanism.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from fastapi import Depends

from src.analytics.visualization import Visualizer
from src.persistence.database import Database
from src.persistence.repositories import ScanRepository
from src.persistence.report_writer import ReportWriter
from src.pipeline.classifier import CNNClassifier
from src.pipeline.pipeline import ClassificationPipeline
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.segmentor import Segmentor

logger = logging.getLogger(__name__)

_CONFIG_PATH = "config/config.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _get_database() -> Database:
    cfg = _load_config()
    return Database.from_config(cfg)


@lru_cache(maxsize=1)
def _get_pipeline() -> ClassificationPipeline:
    cfg = _load_config()

    preprocessor = Preprocessor.from_config(cfg)
    segmentor = Segmentor.from_config(cfg)
    classifier = CNNClassifier.from_config(cfg)

    weights_path = cfg["paths"]["model_weights"]
    if Path(weights_path).exists():
        classifier.load(weights_path)
    else:
        logger.warning(
            "Model weights not found at '%s'. Predictions will fail until trained.",
            weights_path,
        )

    return ClassificationPipeline(
        preprocessor=preprocessor,
        segmentor=segmentor,
        classifier=classifier,
        confidence_threshold=cfg["inference"]["confidence_threshold"],
    )


# ------------------------------------------------------------------
# FastAPI-injectable dependency functions
# ------------------------------------------------------------------

def get_config() -> dict:
    return _load_config()


def get_pipeline() -> ClassificationPipeline:
    return _get_pipeline()


def get_repository() -> ScanRepository:
    return ScanRepository(_get_database())


def get_report_writer() -> ReportWriter:
    cfg = _load_config()
    return ReportWriter(output_dir=cfg["paths"]["output_reports"])


def get_visualizer() -> Visualizer:
    cfg = _load_config()
    return Visualizer.from_config(cfg)
