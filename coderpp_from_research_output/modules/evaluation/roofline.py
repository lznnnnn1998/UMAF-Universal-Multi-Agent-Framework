"""
Roofline Plot — Publication-quality arithmetic intensity vs FLOP/s figures.

Generates roofline plots showing the performance limits of GPU hardware
(peak compute, memory bandwidth ceiling) with measured attention kernel
performance points overlaid.

Usage:
    from evaluation.roofline import RooflinePlot, RooflineMeasurement, GPU_ROOFLINE_SPECS

    plotter = RooflinePlot(gpu_model="A100")
    plotter.add_measurement("FlashAttention v3", arithmetic_intensity=10.5, flops_per_sec=250)
    plotter.save("roofline.pdf")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# ─────────────────────────────────────────────────────────────────────────
# GPU Roofline Specifications
# ─────────────────────────────────────────────────────────────────────────

GPU_ROOFLINE_SPECS: dict[str, dict[str, float]] = {
    "H100": {
        "peak_tflops_fp16": 989.0,
        "peak_tflops_fp32": 67.0,
        "peak_tflops_tc": 1979.0,  # Tensor Core (sparse)
        "bandwidth_gb_s": 3350.0,
        "l2_bandwidth_gb_s": 9000.0,
        "hbm_gb": 80.0,
    },
    "A100": {
        "peak_tflops_fp16": 312.0,
        "peak_tflops_fp32": 19.5,
        "peak_tflops_tc": 624.0,
        "bandwidth_gb_s": 2039.0,
        "l2_bandwidth_gb_s": 6000.0,
        "hbm_gb": 80.0,
    },
    "H800": {
        "peak_tflops_fp16": 756.0,
        "peak_tflops_fp32": 60.0,
        "peak_tflops_tc": 1512.0,
        "bandwidth_gb_s": 3000.0,
        "l2_bandwidth_gb_s": 8000.0,
        "hbm_gb": 80.0,
    },
    "L40S": {
        "peak_tflops_fp16": 181.0,
        "peak_tflops_fp32": 90.5,
        "peak_tflops_tc": 362.0,
        "bandwidth_gb_s": 576.0,
        "l2_bandwidth_gb_s": 1500.0,
        "hbm_gb": 48.0,
    },
    "RTX 4090": {
        "peak_tflops_fp16": 165.0,
        "peak_tflops_fp32": 82.6,
        "peak_tflops_tc": 330.0,
        "bandwidth_gb_s": 1008.0,
        "l2_bandwidth_gb_s": 2000.0,
        "hbm_gb": 24.0,
    },
}

VALID_ROOFLINE_GPU_NAMES: set[str] = set(GPU_ROOFLINE_SPECS.keys())


# ─────────────────────────────────────────────────────────────────────────
# Roofline Measurement
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class RooflineMeasurement:
    """A single measurement point on the roofline plot.

    Attributes:
        label: Description label (e.g. "FlashAttention v3").
        arithmetic_intensity: FLOPs per byte of memory traffic.
        flops_per_sec: Achieved FLOP/s (in TFLOPS).
        achieved_tflops: Same as flops_per_sec, in TFLOPS.
        color: Plot color (matplotlib-compatible).
        marker: Matplotlib marker style.
        marker_size: Marker size in points.
        note: Optional annotation text.
    """

    label: str
    arithmetic_intensity: float
    flops_per_sec: float
    achieved_tflops: float = 0.0
    color: str = "blue"
    marker: str = "o"
    marker_size: float = 80.0
    note: str = ""

    def __post_init__(self):
        if self.achieved_tflops == 0.0:
            self.achieved_tflops = self.flops_per_sec

    def __repr__(self) -> str:
        return (
            f"RooflineMeasurement({self.label}, "
            f"AI={self.arithmetic_intensity:.1f}, "
            f"TFLOPS={self.achieved_tflops:.1f})"
        )


# ─────────────────────────────────────────────────────────────────────────
# Roofline Plot
# ─────────────────────────────────────────────────────────────────────────


class RooflinePlot:
    """Generate publication-quality roofline plots for GPU performance analysis.

    Draws the memory bandwidth ceiling and compute ceiling as straight lines
    on a log-log plot of arithmetic intensity (FLOP/byte) vs FLOP/s.

    Args:
        gpu_model: GPU model name (must be in GPU_ROOFLINE_SPECS).
        precision: "fp16", "fp32", or "tc" (Tensor Core, default: "fp16").
        figure_size: (width, height) in inches for the figure.
        dpi: Figure DPI.
    """

    # Standard color cycle for measurements
    DEFAULT_COLORS: list[str] = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    # Standard marker cycle
    DEFAULT_MARKERS: list[str] = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "*"]

    def __init__(
        self,
        gpu_model: str = "H100",
        precision: str = "fp16",
        figure_size: tuple[float, float] = (8.0, 6.0),
        dpi: int = 150,
    ):
        if gpu_model not in VALID_ROOFLINE_GPU_NAMES:
            raise ValueError(
                f"Unknown GPU model '{gpu_model}'. "
                f"Valid: {sorted(VALID_ROOFLINE_GPU_NAMES)}"
            )
        if precision not in ("fp16", "fp32", "tc"):
            raise ValueError(f"Unknown precision '{precision}'. Use fp16, fp32, or tc.")

        self.gpu_model = gpu_model
        self.precision = precision
        self.figure_size = figure_size
        self.dpi = dpi
        self.measurements: list[RooflineMeasurement] = []

        self._specs = GPU_ROOFLINE_SPECS[gpu_model]
        peak_key = f"peak_tflops_{precision}"
        self._peak_tflops = self._specs.get(
            peak_key, self._specs.get("peak_tflops_fp16", 0.0)
        )
        self._bandwidth_gb_s = self._specs["bandwidth_gb_s"]
        self._ridge_point = self._peak_tflops / (self._bandwidth_gb_s / 1000.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def peak_tflops(self) -> float:
        """Peak TFLOPS for this GPU and precision."""
        return self._peak_tflops

    @property
    def bandwidth_gb_s(self) -> float:
        """HBM bandwidth in GB/s."""
        return self._bandwidth_gb_s

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at the ridge point (compute/memory boundary)."""
        return self._ridge_point

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_measurement(
        self,
        label: str,
        arithmetic_intensity: float,
        flops_per_sec: float,
        color: str | None = None,
        marker: str | None = None,
        note: str = "",
    ) -> RooflineMeasurement:
        """Add a measurement point to the plot.

        Args:
            label: Human-readable label for the legend.
            arithmetic_intensity: FLOP/byte ratio.
            flops_per_sec: Achieved TFLOPS.
            color: Matplotlib color (auto-cycled if None).
            marker: Matplotlib marker (auto-cycled if None).
            note: Optional annotation text.

        Returns:
            The added RooflineMeasurement.
        """
        idx = len(self.measurements)
        if color is None:
            color = self.DEFAULT_COLORS[idx % len(self.DEFAULT_COLORS)]
        if marker is None:
            marker = self.DEFAULT_MARKERS[idx % len(self.DEFAULT_MARKERS)]

        m = RooflineMeasurement(
            label=label,
            arithmetic_intensity=arithmetic_intensity,
            flops_per_sec=flops_per_sec,
            achieved_tflops=flops_per_sec,
            color=color,
            marker=marker,
            note=note,
        )
        self.measurements.append(m)
        return m

    def add_measurement_from_attention(
        self,
        label: str,
        seq_len: int,
        d_model: int,
        batch_size: int = 1,
        elapsed_seconds: float = 1.0,
        bytes_moved: int | None = None,
        color: str | None = None,
        marker: str | None = None,
    ) -> RooflineMeasurement:
        """Add a measurement derived from attention computation parameters.

        Args:
            label: Human-readable label.
            seq_len: Sequence length.
            d_model: Model dimension.
            batch_size: Batch size.
            elapsed_seconds: Wall-clock time for one forward pass (seconds).
            bytes_moved: Total bytes moved between HBM and compute
                (auto-estimated if None).
            color: Matplotlib color (optional).
            marker: Matplotlib marker (optional).

        Returns:
            RooflineMeasurement.
        """
        n_heads = 8
        d_head = d_model // n_heads

        # FLOP count
        flops = (
            batch_size
            * n_heads
            * seq_len
            * seq_len
            * (4 * d_head + 5)
        )
        tflops = flops / (elapsed_seconds * 1e12)

        # Bytes moved
        if bytes_moved is None:
            elem_size = 2  # fp16
            bytes_moved = (
                batch_size
                * n_heads
                * seq_len
                * d_head
                * 4  # Q, K, V, O
                * elem_size
            )

        arithmetic_intensity = flops / bytes_moved if bytes_moved > 0 else 0.0

        return self.add_measurement(
            label=label,
            arithmetic_intensity=arithmetic_intensity,
            flops_per_sec=tflops,
            color=color,
            marker=marker,
        )

    def classify(self, measurement: RooflineMeasurement) -> str:
        """Classify a measurement as bandwidth-bound or compute-bound.

        Args:
            measurement: The measurement to classify.

        Returns:
            "bandwidth_bound" or "compute_bound".
        """
        bw_tflops = measurement.arithmetic_intensity * (self._bandwidth_gb_s / 1000.0)
        if bw_tflops < self._peak_tflops:
            return "bandwidth_bound"
        return "compute_bound"

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def get_roofline_data(self, num_points: int = 100) -> dict[str, Any]:
        """Get the roofline curve data for external plotting.

        Args:
            num_points: Number of sample points on the curve.

        Returns:
            Dict with keys: arithmetic_intensity, achievable_tflops,
            ridge_point, peak_tflops, bandwidth_gb_s.
        """
        # Generate arithmetic intensity range around the ridge point
        ai_min = self._ridge_point * 0.01
        ai_max = self._ridge_point * 100.0
        ai_values = [
            ai_min * (ai_max / ai_min) ** (i / (num_points - 1))
            for i in range(num_points)
        ]

        bw_tb_s = self._bandwidth_gb_s / 1000.0  # Convert to TB/s
        tflops_values = [
            min(self._peak_tflops, ai * bw_tb_s) for ai in ai_values
        ]

        return {
            "arithmetic_intensity": ai_values,
            "achievable_tflops": tflops_values,
            "ridge_point": self._ridge_point,
            "peak_tflops": self._peak_tflops,
            "bandwidth_gb_s": self._bandwidth_gb_s,
            "gpu_model": self.gpu_model,
            "precision": self.precision,
        }

    def get_measurements_data(self) -> list[dict[str, Any]]:
        """Export measurement data as a list of dicts."""
        return [
            {
                "label": m.label,
                "arithmetic_intensity": m.arithmetic_intensity,
                "achieved_tflops": m.achieved_tflops,
                "classification": self.classify(m),
                "color": m.color,
                "marker": m.marker,
                "note": m.note,
            }
            for m in self.measurements
        ]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, title: str | None = None) -> Any:
        """Generate the roofline plot figure.

        Args:
            title: Custom title (default: auto-generated).

        Returns:
            matplotlib Figure object.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        data = self.get_roofline_data()

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Roofline curve
        ax.loglog(
            data["arithmetic_intensity"],
            data["achievable_tflops"],
            "k-",
            linewidth=2,
            label=f'{self.gpu_model} ({self.precision})',
        )

        # Ridge point marker
        ax.axvline(
            x=self._ridge_point,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f'Ridge: {self._ridge_point:.1f} FLOP/byte',
        )

        # Measurements
        for m in self.measurements:
            ax.scatter(
                m.arithmetic_intensity,
                m.achieved_tflops,
                c=m.color,
                marker=m.marker,
                s=m.marker_size,
                label=m.label,
                edgecolors="black",
                linewidths=0.5,
            )
            if m.note:
                ax.annotate(
                    m.note,
                    (m.arithmetic_intensity, m.achieved_tflops),
                    textcoords="offset points",
                    xytext=(10, 5),
                    fontsize=8,
                )

        # Labels
        if title is None:
            title = f"Roofline Plot — {self.gpu_model} ({self.precision.upper()})"

        ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
        ax.set_ylabel("Performance (TFLOPS)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        return fig

    def save(self, filepath: str, title: str | None = None) -> str:
        """Save the roofline plot to a file.

        Args:
            filepath: Output file path (e.g. "roofline.pdf", "roofline.png").
            title: Custom title (default: auto-generated).

        Returns:
            Absolute path to the saved file.
        """
        fig = self.plot(title=title)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pass
        return str(Path(filepath).resolve())

    def __repr__(self) -> str:
        return (
            f"RooflinePlot(gpu={self.gpu_model}, precision={self.precision}, "
            f"peak={self._peak_tflops:.0f} TFLOPS, "
            f"bw={self._bandwidth_gb_s:.0f} GB/s, "
            f"ridge={self._ridge_point:.1f}, "
            f"measurements={len(self.measurements)})"
        )
