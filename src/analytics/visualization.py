"""
analytics/visualization.py — Chart and annotated image generation.

Provides:
  - draw_bounding_boxes() — overlay boxes + labels on the original image
  - bar_chart()           — category-count bar chart
  - pie_chart()           — category-distribution pie chart
  - save_chart()          — save a Matplotlib figure to disk
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.core.entities import DetectedProduct, CategoryCount

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Generates visual outputs for a scan result.

    Parameters
    ----------
    bbox_color : tuple[int, int, int]
        BGR colour for bounding boxes.
    bbox_thickness : int
    font_scale : float
    chart_dpi : int
    chart_style : str
        Matplotlib style name.
    output_dir : str
        Default directory to save charts.
    """

    def __init__(
        self,
        bbox_color: tuple[int, int, int] = (0, 255, 0),
        bbox_thickness: int = 2,
        font_scale: float = 0.7,
        chart_dpi: int = 150,
        chart_style: str = "seaborn-v0_8-darkgrid",
        output_dir: str = "output",
    ) -> None:
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.chart_dpi = chart_dpi
        self.chart_style = chart_style
        self.output_dir = Path(output_dir)
        try:
            plt.style.use(chart_style)
        except OSError:
            logger.warning("Matplotlib style '%s' not found, using default.", chart_style)

    # ------------------------------------------------------------------
    # Bounding box overlay
    # ------------------------------------------------------------------

    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        detected: list[DetectedProduct],
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the image.

        Parameters
        ----------
        image : np.ndarray
            Original BGR uint8 image.
        detected : list[DetectedProduct]

        Returns
        -------
        Annotated image (copy, does not modify original).
        """
        annotated = image.copy()

        for product in detected:
            bb = product.bounding_box
            label = f"{product.label} {product.confidence:.0%}"

            # Draw rectangle
            cv2.rectangle(
                annotated,
                (bb.x, bb.y),
                (bb.x2, bb.y2),
                self.bbox_color,
                self.bbox_thickness,
            )

            # Draw label background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
            )
            label_y = max(bb.y - 5, text_h + 5)
            cv2.rectangle(
                annotated,
                (bb.x, label_y - text_h - baseline),
                (bb.x + text_w, label_y + baseline),
                self.bbox_color,
                cv2.FILLED,
            )

            # Draw label text
            cv2.putText(
                annotated, label,
                (bb.x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 0, 0),          # Black text
                2, cv2.LINE_AA,
            )

        return annotated

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def bar_chart(
        self,
        counts: list[CategoryCount],
        title: str = "Detected Products — Category Count",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Generate a horizontal bar chart of product category counts."""
        if not counts:
            logger.warning("bar_chart called with empty counts.")
            return plt.figure()

        labels = [c.label.capitalize() for c in counts]
        values = [c.count for c in counts]
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

        fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.6)))
        bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8)

        ax.bar_label(bars, fmt="%d", padding=4, fontsize=10)
        ax.set_xlabel("Count", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if save_path:
            self._save(fig, save_path)

        return fig

    def pie_chart(
        self,
        counts: list[CategoryCount],
        title: str = "Detected Products — Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Generate a pie chart of product category distribution."""
        if not counts:
            logger.warning("pie_chart called with empty counts.")
            return plt.figure()

        labels = [c.label.capitalize() for c in counts]
        sizes = [c.count for c in counts]
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        explode = [0.05] * len(labels)

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            explode=explode,
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(10)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()

        if save_path:
            self._save(fig, save_path)

        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=self.chart_dpi, bbox_inches="tight")
        logger.info("Chart saved to %s", save_path)
        plt.close(fig)

    @classmethod
    def from_config(cls, cfg: dict) -> "Visualizer":
        vis = cfg.get("visualization", {})
        return cls(
            bbox_color=tuple(vis.get("bbox_color", [0, 255, 0])),
            bbox_thickness=vis.get("bbox_thickness", 2),
            font_scale=vis.get("font_scale", 0.7),
            chart_dpi=vis.get("chart_dpi", 150),
            chart_style=vis.get("chart_style", "seaborn-v0_8-darkgrid"),
            output_dir=cfg["paths"]["output_charts"],
        )
