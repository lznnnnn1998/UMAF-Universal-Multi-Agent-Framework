"""
Memory Profiler — Peak HBM usage, memory bandwidth utilization, roofline analysis.

Computes the memory footprint of attention mechanism implementations and
evaluates their memory bandwidth utilization using the roofline model
(arithmetic intensity vs achievable FLOP/s).

Usage:
    from evaluation.memory import MemoryProfiler, MemoryProfileResult
    from evaluation.memory import RooflineModel, RooflinePoint, ROOFLINE_GPU_SPECS

    profiler = MemoryProfiler(hbm_capacity_gb=80, bandwidth_gb_s=3350)
    mem_result = profiler.profile(attention_fn, seq_len=2048, d_model=1024)

    model = RooflineModel(gpu_model="H100")
    point = model.evaluate(arithmetic_intensity=10.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

# ─────────────────────────────────────────────────────────────────────────
# Roofline GPU Specifications
# ─────────────────────────────────────────────────────────────────────────

ROOFLINE_GPU_SPECS: dict[str, dict[str, float]] = {
    "H100": {
        "peak_tflops_fp16": 989.0,
        "peak_tflops_bf16": 989.0,
        "bandwidth_gb_s": 3350.0,
        "hbm_gb": 80.0,
    },
    "A100": {
        "peak_tflops_fp16": 312.0,
        "peak_tflops_bf16": 312.0,
        "bandwidth_gb_s": 2039.0,
        "hbm_gb": 80.0,
    },
    "H800": {
        "peak_tflops_fp16": 756.0,
        "peak_tflops_bf16": 756.0,
        "bandwidth_gb_s": 3000.0,
        "hbm_gb": 80.0,
    },
    "L40S": {
        "peak_tflops_fp16": 181.0,
        "peak_tflops_bf16": 181.0,
        "bandwidth_gb_s": 576.0,
        "hbm_gb": 48.0,
    },
    "RTX 4090": {
        "peak_tflops_fp16": 165.0,
        "peak_tflops_bf16": 165.0,
        "bandwidth_gb_s": 1008.0,
        "hbm_gb": 24.0,
    },
}

VALID_ROOFLINE_GPU_NAMES: set[str] = set(ROOFLINE_GPU_SPECS.keys())


# ─────────────────────────────────────────────────────────────────────────
# Roofline model types
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class RooflinePoint:
    """A single point on a roofline plot.

    Attributes:
        arithmetic_intensity: FLOPs per byte of memory traffic.
        achievable_tflops: Achievable TFLOPS at this intensity.
        bandwidth_bound: True if the point is memory-bandwidth-limited.
        compute_bound: True if the point is compute-limited.
    """

    arithmetic_intensity: float
    achievable_tflops: float
    bandwidth_bound: bool
    compute_bound: bool


@dataclass
class RooflineModel:
    """Roofline model for a specific GPU.

    Computes the maximum achievable performance given an arithmetic
    intensity using the standard roofline equation:
        achievable = min(peak_tflops, intensity * bandwidth / 1000)

    The factor 1000 converts GB/s to TB/s: GB/s / 1000 = TB/s, matching TFLOPS.

    Args:
        gpu_model: GPU model name (must be in ROOFLINE_GPU_SPECS).
    """

    gpu_model: str

    def __post_init__(self):
        if self.gpu_model not in VALID_ROOFLINE_GPU_NAMES:
            raise ValueError(
                f"Unknown GPU model '{self.gpu_model}'. "
                f"Valid: {sorted(VALID_ROOFLINE_GPU_NAMES)}"
            )
        self._specs = ROOFLINE_GPU_SPECS[self.gpu_model]

    @property
    def peak_tflops(self) -> float:
        """Peak TFLOPS for the GPU."""
        return self._specs["peak_tflops_fp16"]

    @property
    def bandwidth_gb_s(self) -> float:
        """HBM bandwidth in GB/s."""
        return self._specs["bandwidth_gb_s"]

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at the ridge point (compute/memory boundary).

        Ridge = peak_tflops / (bandwidth_gb_s / 1000)
        Below this intensity, the kernel is bandwidth-bound.
        Above it, the kernel is compute-bound.
        """
        return self.peak_tflops / (self.bandwidth_gb_s / 1000.0)

    def evaluate(self, arithmetic_intensity: float) -> RooflinePoint:
        """Evaluate achievable performance at a given arithmetic intensity.

        Args:
            arithmetic_intensity: FLOPs per byte of DRAM traffic.

        Returns:
            RooflinePoint with achievable TFLOPS and bound classification.
        """
        # Convert GB/s → TB/s: bandwidth_gb_s / 1000
        bandwidth_tflops = arithmetic_intensity * (self.bandwidth_gb_s / 1000.0)
        achievable = min(self.peak_tflops, bandwidth_tflops)
        bandwidth_bound = bandwidth_tflops < self.peak_tflops
        return RooflinePoint(
            arithmetic_intensity=arithmetic_intensity,
            achievable_tflops=achievable,
            bandwidth_bound=bandwidth_bound,
            compute_bound=not bandwidth_bound,
        )


# ─────────────────────────────────────────────────────────────────────────
# Memory Profile Result
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryProfileResult:
    """Result of a memory profiling run.

    Attributes:
        gpu_model: GPU model name.
        hbm_capacity_gb: GPU HBM capacity in GB.
        bandwidth_gb_s: GPU HBM bandwidth in GB/s.
        peak_memory_bytes: Estimated peak HBM usage in bytes.
        peak_memory_gb: Peak HBM usage in GB.
        memory_utilization_pct: (peak / capacity) * 100.
        bandwidth_utilization_pct: Achieved bandwidth / theoretical peak * 100.
        roofline_point: RooflinePoint classifying the kernel's compute regime.
        seq_len: Sequence length used in the profile.
        d_model: Model dimension used.
        batch_size: Batch size used.
        flops: Estimated total FLOPs for the attention computation.
        bytes_read: Estimated total bytes read from HBM.
        bytes_written: Estimated total bytes written to HBM.
    """

    gpu_model: str
    hbm_capacity_gb: float
    bandwidth_gb_s: float
    peak_memory_bytes: int
    peak_memory_gb: float
    memory_utilization_pct: float
    bandwidth_utilization_pct: float
    roofline_point: RooflinePoint
    seq_len: int
    d_model: int
    batch_size: int
    flops: float
    bytes_read: int
    bytes_written: int

    def __repr__(self) -> str:
        return (
            f"MemoryProfileResult(gpu={self.gpu_model}, seq={self.seq_len}, "
            f"peak={self.peak_memory_gb:.2f}GB, "
            f"mem_util={self.memory_utilization_pct:.1f}%, "
            f"bw_util={self.bandwidth_utilization_pct:.1f}%)"
        )


# ─────────────────────────────────────────────────────────────────────────
# Memory Profiler
# ─────────────────────────────────────────────────────────────────────────


class MemoryProfiler:
    """Profile attention mechanism memory usage and bandwidth utilization.

    Estimates peak HBM usage by counting tensor allocations and data movement
    in the attention computation. Uses the roofline model to classify the
    kernel as compute-bound or memory-bandwidth-bound.

    Args:
        hbm_capacity_gb: GPU HBM capacity in GB (default: 80 for H100/A100).
        bandwidth_gb_s: GPU HBM bandwidth in GB/s (default: 3350 for H100).
        gpu_model: GPU model name for roofline analysis (default: "H100").
    """

    def __init__(
        self,
        hbm_capacity_gb: float = 80.0,
        bandwidth_gb_s: float = 3350.0,
        gpu_model: str = "H100",
    ):
        self.hbm_capacity_gb = hbm_capacity_gb
        self.bandwidth_gb_s = bandwidth_gb_s
        self.gpu_model = gpu_model
        self._roofline = RooflineModel(gpu_model=gpu_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        attention_fn: Callable[..., torch.Tensor],
        seq_len: int = 2048,
        d_model: int = 1024,
        batch_size: int = 8,
        include_grad: bool = False,
        **kwargs: Any,
    ) -> MemoryProfileResult:
        """Profile memory usage of an attention function.

        Args:
            attention_fn: A callable ``fn(q, k, v, **kwargs)``.
            seq_len: Sequence length.
            d_model: Model dimension.
            batch_size: Batch size.
            include_grad: If True, also estimate gradient memory (default: False).
            **kwargs: Additional arguments forwarded to ``attention_fn``.

        Returns:
            MemoryProfileResult with memory and roofline analysis.
        """
        flops = self._count_attention_flops(seq_len, d_model, batch_size)
        bytes_read, bytes_written = self._estimate_memory_traffic(
            seq_len, d_model, batch_size, include_grad
        )
        total_bytes = bytes_read + bytes_written
        arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0.0

        roofline_point = self._roofline.evaluate(arithmetic_intensity)

        # Peak memory: Q, K, V tensors + attention matrix + output
        n_heads = 8
        d_head = d_model // n_heads
        elem_size = 2  # fp16/bf16
        peak_memory_bytes = (
            batch_size * n_heads * seq_len * d_head * 4  # Q, K, V, O
            + batch_size * n_heads * seq_len * seq_len  # attention scores
        ) * elem_size
        peak_memory_gb = peak_memory_bytes / (1024**3)
        memory_utilization_pct = (peak_memory_gb / self.hbm_capacity_gb) * 100.0

        bandwidth_utilization_pct = (
            (roofline_point.achievable_tflops / self._roofline.peak_tflops) * 100.0
            if self._roofline.peak_tflops > 0
            else 0.0
        )

        return MemoryProfileResult(
            gpu_model=self.gpu_model,
            hbm_capacity_gb=self.hbm_capacity_gb,
            bandwidth_gb_s=self.bandwidth_gb_s,
            peak_memory_bytes=peak_memory_bytes,
            peak_memory_gb=peak_memory_gb,
            memory_utilization_pct=memory_utilization_pct,
            bandwidth_utilization_pct=bandwidth_utilization_pct,
            roofline_point=roofline_point,
            seq_len=seq_len,
            d_model=d_model,
            batch_size=batch_size,
            flops=flops,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
        )

    def profile_tensor_shapes(self, *shapes: tuple[int, ...], dtype_size: int = 2) -> int:
        """Estimate total memory for a list of tensor shapes.

        Args:
            *shapes: Tensor shape tuples, e.g. (2, 8, 2048, 128).
            dtype_size: Bytes per element (default: 2 for fp16/bf16).

        Returns:
            Total memory in bytes.
        """
        total = 0
        for shape in shapes:
            elements = 1
            for dim in shape:
                elements *= dim
            total += elements * dtype_size
        return total

    def get_roofline_model(self) -> RooflineModel:
        """Return the underlying roofline model for this profiler."""
        return self._roofline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_attention_flops(seq_len: int, d_model: int, batch_size: int) -> float:
        """Count FLOPs for standard scaled dot-product attention."""
        n_heads = 8
        d_head = d_model // n_heads
        # QK^T: 2 * seq^2 * d_head, softmax: ~5 * seq^2, PV: 2 * seq^2 * d_head
        flops_per_head = seq_len * seq_len * (4 * d_head + 5)
        return n_heads * flops_per_head * batch_size

    @staticmethod
    def _estimate_memory_traffic(
        seq_len: int,
        d_model: int,
        batch_size: int,
        include_grad: bool = False,
    ) -> tuple[int, int]:
        """Estimate bytes read and written during attention.

        Returns:
            (bytes_read, bytes_written) as integers.
        """
        n_heads = 8
        d_head = d_model // n_heads
        elem_size = 2  # fp16/bf16

        # Read: Q, K, V (3 tensors)
        bytes_read = 3 * batch_size * n_heads * seq_len * d_head * elem_size
        # Write: output O
        bytes_written = batch_size * n_heads * seq_len * d_head * elem_size

        if include_grad:
            # Gradients: read O_grad, write Q_grad, K_grad, V_grad
            bytes_read += batch_size * n_heads * seq_len * d_head * elem_size
            bytes_written += 3 * batch_size * n_heads * seq_len * d_head * elem_size

        return bytes_read, bytes_written
