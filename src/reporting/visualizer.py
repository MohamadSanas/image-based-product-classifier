# ============================================================
# Module  : Product Visualization
# Owner   : AHAMED A.A.
# Course  : EC9570 – Digital Image Processing
# Desc    : Generates bar charts, pie charts, and a confidence
#           distribution plot from ClassificationResult data.
#           All charts are saved to output/charts/ by default.
# ============================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Colour palette (17 RPC categories -> distinct colours)
# ──────────────────────────────────────────────────────────────

# Carefully chosen HSL-based palette — distinct, visually rich
_CATEGORY_COLORS: list[str] = [
    "#E63946",  # vivid red
    "#F4A261",  # warm orange
    "#2A9D8F",  # teal
    "#457B9D",  # steel blue
    "#A8DADC",  # light cyan
    "#6A0572",  # deep purple
    "#F1C40F",  # golden yellow
    "#27AE60",  # emerald
    "#E67E22",  # carrot
    "#2980B9",  # cobalt blue
    "#8E44AD",  # amethyst
    "#16A085",  # green sea
    "#D35400",  # pumpkin
    "#C0392B",  # pomegranate
    "#1ABC9C",  # turquoise
    "#2C3E50",  # midnight blue
    "#F39C12",  # orange (amber)
]


def _get_color(idx: int) -> str:
    """Return a colour hex string, cycling through the palette."""
    return _CATEGORY_COLORS[idx % len(_CATEGORY_COLORS)]


# ──────────────────────────────────────────────────────────────
# ProductVisualizer
# ──────────────────────────────────────────────────────────────

class ProductVisualizer:
    """
    Generates Matplotlib charts from product classification data.

    Typical usage
    -------------
    >>> from reporting import ProductVisualizer
    >>> viz = ProductVisualizer(output_dir="output/charts")
    >>> bar_path = viz.bar_chart(classified, stem="shelf")
    >>> pie_path = viz.pie_chart(classified, stem="shelf")
    >>> conf_path = viz.confidence_distribution(classified, stem="shelf")

    All methods return the ``Path`` of the saved file.
    """

    def __init__(
        self,
        output_dir: str | Path = "output/charts",
        dpi: int = 150,
        fig_width: float = 10.0,
        fig_height: float = 6.0,
    ) -> None:
        """
        Initialise the visualizer.

        Parameters
        ----------
        output_dir : str | Path
            Directory where charts are saved.  Created automatically.
        dpi : int
            Output image resolution (dots per inch). Default 150.
        fig_width : float
            Figure width in inches. Default 10.
        fig_height : float
            Figure height in inches. Default 6.
        """
        self._output_dir = Path(output_dir)
        self._dpi = dpi
        self._fig_w = fig_width
        self._fig_h = fig_height
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ───────────────────────────────────────────────

    def _count_by_category(self, results: list) -> dict[str, int]:
        """Count results grouped by category, sorted descending."""
        from collections import Counter
        counter: Counter[str] = Counter(r.category for r in results)
        return dict(sorted(counter.items(), key=lambda kv: -kv[1]))

    def _import_matplotlib(self) -> Any:
        """Import and configure matplotlib; raise ImportError with hint if missing."""
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend (safe for all envs)
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
            return plt, ticker
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for ProductVisualizer. "
                "Install it with:  pip install matplotlib"
            ) from exc

    def _apply_style(self, fig: Any, ax: Any, title: str) -> None:
        """Apply a consistent, polished style to a figure."""
        fig.patch.set_facecolor("#1A1A2E")          # dark navy background
        ax.set_facecolor("#16213E")                  # slightly lighter axes bg
        ax.set_title(title, fontsize=15, fontweight="bold",
                     color="white", pad=14)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3A3A5C")
        ax.title.set_color("white")

    # ── 1. bar_chart ──────────────────────────────────────────

    def bar_chart(
        self,
        results: list,
        stem: str = "output",
        title: str = "Product Count by Category",
    ) -> Path:
        """
        Save a horizontal bar chart of product counts by category.

        Parameters
        ----------
        results : list[ClassificationResult]
            Output of ``ProductClassifier.classify_detections()``.
        stem : str
            Filename stem (no extension); e.g. ``"shelf"`` ->
            ``output/charts/shelf_bar.png``.
        title : str
            Chart title.

        Returns
        -------
        Path
            Path of the saved PNG file.

        Raises
        ------
        ValueError
            If *results* is empty.
        ImportError
            If matplotlib is not installed.
        """
        if not results:
            raise ValueError("bar_chart: results list is empty.")

        plt, ticker = self._import_matplotlib()

        counts = self._count_by_category(results)
        categories = list(counts.keys())
        values     = list(counts.values())
        total      = sum(values)
        colors     = [_get_color(i) for i in range(len(categories))]

        fig, ax = plt.subplots(figsize=(self._fig_w, max(4, len(categories) * 0.55)))
        self._apply_style(fig, ax, title)

        bars = ax.barh(categories, values, color=colors, edgecolor="#1A1A2E",
                       linewidth=0.6, height=0.65)

        # Value labels on each bar
        for bar, val in zip(bars, values):
            pct = val / total * 100
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{val}  ({pct:.1f}%)",
                va="center", ha="left",
                fontsize=8.5, color="white", fontweight="bold",
            )

        ax.set_xlabel("Number of Detections", color="white", labelpad=8)
        ax.set_ylabel("Product Category",     color="white", labelpad=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.invert_yaxis()   # highest count at top
        ax.set_xlim(0, max(values) * 1.25)

        # Subtle grid
        ax.xaxis.grid(True, color="#3A3A5C", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout(pad=1.5)
        out_path = self._output_dir / f"{stem}_bar.png"
        plt.savefig(str(out_path), dpi=self._dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("bar_chart saved -> %s", out_path)
        return out_path

    # ── 2. pie_chart ──────────────────────────────────────────

    def pie_chart(
        self,
        results: list,
        stem: str = "output",
        title: str = "Product Category Distribution",
    ) -> Path:
        """
        Save a donut-style pie chart of category distribution.

        Parameters
        ----------
        results : list[ClassificationResult]
            Output of ``ProductClassifier.classify_detections()``.
        stem : str
            Filename stem; e.g. ``"shelf"`` -> ``output/charts/shelf_pie.png``.
        title : str
            Chart title.

        Returns
        -------
        Path
            Path of the saved PNG file.

        Raises
        ------
        ValueError
            If *results* is empty.
        ImportError
            If matplotlib is not installed.
        """
        if not results:
            raise ValueError("pie_chart: results list is empty.")

        plt, _ = self._import_matplotlib()

        counts = self._count_by_category(results)
        labels = list(counts.keys())
        values = list(counts.values())
        colors = [_get_color(i) for i in range(len(labels))]
        total  = sum(values)

        fig, ax = plt.subplots(figsize=(self._fig_w, self._fig_h))
        self._apply_style(fig, ax, title)

        wedge_props = dict(width=0.55, edgecolor="#1A1A2E", linewidth=1.2)

        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
            pctdistance=0.75,
            startangle=140,
            wedgeprops=wedge_props,
            textprops={"color": "white", "fontsize": 8.5, "fontweight": "bold"},
        )

        # Centre label
        ax.text(0, 0, f"{total}\ndetected", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")

        # External legend (truncate long names)
        max_label_len = 28
        legend_labels = [
            f"{lbl[:max_label_len]}{'…' if len(lbl) > max_label_len else ''}  ({v})"
            for lbl, v in zip(labels, values)
        ]
        ax.legend(
            wedges, legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8.5,
            facecolor="#16213E",
            edgecolor="#3A3A5C",
            labelcolor="white",
            framealpha=0.85,
        )

        plt.tight_layout(pad=1.5)
        out_path = self._output_dir / f"{stem}_pie.png"
        plt.savefig(str(out_path), dpi=self._dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("pie_chart saved -> %s", out_path)
        return out_path

    # ── 3. confidence_distribution ────────────────────────────

    def confidence_distribution(
        self,
        results: list,
        stem: str = "output",
        title: str = "Confidence Score Distribution",
        bins: int = 20,
    ) -> Path:
        """
        Save a histogram of YOLO confidence scores across all detections.

        Parameters
        ----------
        results : list[ClassificationResult]
            Output of ``ProductClassifier.classify_detections()``.
        stem : str
            Filename stem; e.g. ``"shelf"`` ->
            ``output/charts/shelf_conf.png``.
        title : str
            Chart title.
        bins : int
            Number of histogram bins. Default 20.

        Returns
        -------
        Path
            Path of the saved PNG file.

        Raises
        ------
        ValueError
            If *results* is empty.
        ImportError
            If matplotlib is not installed.
        """
        if not results:
            raise ValueError("confidence_distribution: results list is empty.")

        plt, ticker = self._import_matplotlib()

        confidences = [r.confidence for r in results]
        mean_conf   = sum(confidences) / len(confidences)

        fig, ax = plt.subplots(figsize=(self._fig_w, self._fig_h))
        self._apply_style(fig, ax, title)

        n, bin_edges, patches = ax.hist(
            confidences, bins=bins, range=(0.0, 1.0),
            color="#2A9D8F", edgecolor="#1A1A2E", linewidth=0.6,
            alpha=0.85,
        )

        # Colour bars by confidence zone (red -> yellow -> green gradient)
        for patch, left_edge in zip(patches, bin_edges[:-1]):
            mid = left_edge + (bin_edges[1] - bin_edges[0]) / 2
            if mid < 0.5:
                patch.set_facecolor("#E63946")
            elif mid < 0.75:
                patch.set_facecolor("#F4A261")
            else:
                patch.set_facecolor("#2A9D8F")

        # Mean confidence vertical line
        ax.axvline(mean_conf, color="#F1C40F", linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_conf:.1%}")

        ax.set_xlabel("Confidence Score", color="white", labelpad=8)
        ax.set_ylabel("Number of Detections", color="white", labelpad=8)
        ax.set_xlim(0, 1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.xaxis.grid(True, color="#3A3A5C", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.yaxis.grid(True, color="#3A3A5C", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        legend = ax.legend(
            fontsize=9.5, facecolor="#16213E", edgecolor="#3A3A5C",
            labelcolor="white", framealpha=0.85,
        )

        # Annotation: count above each bar
        for bar_n, bar_patch in zip(n, patches):
            if bar_n > 0:
                ax.text(
                    bar_patch.get_x() + bar_patch.get_width() / 2,
                    bar_patch.get_height() + 0.05,
                    str(int(bar_n)),
                    ha="center", va="bottom",
                    fontsize=7, color="white",
                )

        plt.tight_layout(pad=1.5)
        out_path = self._output_dir / f"{stem}_conf.png"
        plt.savefig(str(out_path), dpi=self._dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("confidence_distribution saved -> %s", out_path)
        return out_path

    # ── repr ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ProductVisualizer("
            f"output_dir='{self._output_dir}', "
            f"dpi={self._dpi})"
        )
