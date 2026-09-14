# ============================================================
# Module  : Configuration Loader
# Owner   : SANAS M.M.
# Course  : EC9570 – Digital Image Processing
# Desc    : Loads config/config.yaml and returns a nested
#           SimpleNamespace so callers can access settings as
#           plain attributes:  cfg.preprocessing.target_size
# ============================================================

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

logger = logging.getLogger(__name__)

# Default location of the master config file
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _dict_to_namespace(d: Any) -> Any:
    """
    Recursively convert a dict (and nested dicts) to SimpleNamespace
    so that YAML values are accessible as attributes rather than keys.
    Lists and scalars are returned unchanged.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(item) for item in d]
    return d


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """
    Load the YAML configuration file and return it as a nested
    ``SimpleNamespace``.

    Parameters
    ----------
    path : str | Path | None
        Path to the YAML file.  Defaults to ``config/config.yaml``
        relative to the project root.

    Returns
    -------
    SimpleNamespace
        Nested namespace matching the YAML structure.  Example access::

            cfg = load_config()
            cfg.preprocessing.target_size   # [224, 224]
            cfg.inference.confidence_threshold  # 0.6
            cfg.paths.output_images         # "output/detected_images"

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the resolved path.
    ImportError
        If the ``pyyaml`` package is not installed.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required for config_loader. "
            "Install it with:  pip install pyyaml"
        ) from exc

    config_path = Path(path).resolve() if path is not None else _DEFAULT_CONFIG_PATH

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected location: config/config.yaml (project root)"
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    cfg = cast(SimpleNamespace, _dict_to_namespace(raw))
    logger.info("Config loaded from '%s'", config_path)
    return cfg
