"""
Throughput Benchmark — Measures tokens/sec, TFLOPS, and compute utilization.

Evaluates attention mechanism implementations against GPU peak specifications
using standard FLOP counting and wall-clock timing. Supports per-GPU roofline
comparison (H100, A100, H800, L40S, RTX 4090, MI300X).

Usage:
    from evaluation.throughput import ThroughputBenchmark, ThroughputResult, GPU_SPECS

    bench = ThroughputBenchmark(gpu_model="H100")
    result = bench.benchmark(attention_fn, seq_len=2048, d_model=1024, batch_size=8)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# ─────────────────────────────────────────────────────────────────────────
# GPU Specifications (peak TFLOPS for FP16/BF16, bandwidth GB/s)
# ─────────────────────────────────────────────────────────────────────────

GPU_SPECS: dict[str, dict[str, float]] = {
    "H100": {
        "peak_tflops_fp16": 989.0,
        "peak_tflops_bf16": 989.0,
        "peak_tflops_fp32": 67.0,
        "memory_bandwidth_gb_s": 3350.0,
        "hbm_capacity_gb": 80.0,
    },
    "A100": {
        "peak_tflops_fp16": 312.0,
        "peak_tflops_bf16": 312.0,
        "peak_tflops_fp32": 19.5,
        "memory_bandwidth_gb_s": 2039.0,
        "hbm_capacity_gb": 80.0,
    },
    "H800": {
        "peak_tflops_fp16": 756.0,
        "peak_tflops_bf16": 756.0,
        "peak_tflops_fp32": 60.0,
        "memory_bandwidth_gb_s": 3000.0,
        "hbm_capacity_gb": 80.0,
    },
    "L40S": {
        "peak_tflops_fp16": 181.0,
        "peak_tflops_bf16": 181.0,
        "peak_tflops_fp32": 90.5,
        "memory_bandwidth_gb_s": 576.0,
        "hbm_capacity_gb": 48.0,
    },
    "RTX 4090": {
        "peak_tflops_fp16": 165.0,
        "peak_tflops_bf16": 165.0,
        "peak_tflops_fp32": 82.6,
        "memory_bandwidth_gb_s": 1008.0,
        "hbm_capacity_gb": 24.0,
    },
    "MI300X": {
        "peak_tflops_fp16": 1307.0,
        "peak_tflops_bf16": 1307.0,
        "peak_tflops_fp32": 81.7,
        "memory_bandwidth_gb_s": 5300.0,
        "hbm_capacity_gb": 192.0,
    },
}

VALID_GPU_NAMES: set[str] = set(GPU_SPECS.keys())


# ─────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ThroughputResult:
    """Result of a single throughput benchmark run.

    Attributes:
        gpu_model: Name of the GPU model used for peak comparisons.
        seq_len: Sequence length tested.
        d_model: Model dimension (d_head or d_model).
        batch_size: Batch size.
        tokens_per_second: Achieved throughput in tokens/second.
        achieved_tflops: Achieved TFLOPS (FP16/BF16).
        peak_tflops: GPU peak TFLOPS for the precision used.
        compute_utilization_pct: (achieved / peak) * 100 as percentage.
        elapsed_seconds: Total wall-clock time for the benchmark run.
        total_flops: Total FLOPs counted for the attention computation.
        memory_bytes: Estimated HBM memory footprint in bytes.
        precision: Computation precision used (fp16, bf16, fp32).
    """

    gpu_model: str
    seq_len: int
    d_model: int
    batch_size: int
    tokens_per_second: float
    achieved_tflops: float
    peak_tflops: float
    compute_utilization_pct: float
    elapsed_seconds: float
    total_flops: float
    memory_bytes: int
    precision: str = "fp16"

    def __repr__(self) -> str:
        return (
            f"ThroughputResult(gpu={self.gpu_model}, seq={self.seq_len}, "
            f"d={self.d_model}, batch={self.batch_size}, "
            f"tok/s={self.tokens_per_second:.1f}, "
            f"TFLOPS={self.achieved_tflops:.2f}, "
            f"util={self.compute_utilization_pct:.1f}%)"
        )


# ─────────────────────────────────────────────────────────────────────────
# Throughput Benchmark
# ─────────────────────────────────────────────────────────────────────────


class ThroughputBenchmark:
    """Benchmark attention mechanism throughput against GPU peak specs.

    Measures wall-clock time for a forward (and optionally backward) pass,
    counts FLOPs, and reports compute utilization as a percentage of the
    GPU's theoretical peak throughput.

    Args:
        gpu_model: GPU model name (must be in GPU_SPECS, default: "H100").
        warmup_iters: Number of warmup iterations before timing (default: 5).
        bench_iters: Number of timed iterations (default: 20).
        precision: "fp16", "bf16", or "fp32" (default: "fp16").
    """

    def __init__(
        self,
        gpu_model: str = "H100",
        warmup_iters: int = 5,
        bench_iters: int = 20,
        precision: str = "fp16",
    ):
        if gpu_model not in VALID_GPU_NAMES:
            raise ValueError(
                f"Unknown GPU model '{gpu_model}'. Valid: {sorted(VALID_GPU_NAMES)}"
            )
        if precision not in ("fp16", "bf16", "fp32"):
            raise ValueError(f"Unknown precision '{precision}'. Use fp16, bf16, or fp32.")

        self.gpu_model = gpu_model
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.precision = precision
        self._specs = GPU_SPECS[gpu_model]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def benchmark(
        self,
        attention_fn: Callable[..., torch.Tensor],
        seq_len: int = 2048,
        d_model: int = 1024,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> ThroughputResult:
        """Benchmark an attention function's throughput.

        Args:
            attention_fn: A callable ``fn(q, k, v, **kwargs)`` returning
                an output tensor of shape ``[batch, heads, seq, d_head]``.
            seq_len: Sequence length.
            d_model: Model dimension.
            batch_size: Batch size.
            **kwargs: Additional arguments forwarded to ``attention_fn``.

        Returns:
            ThroughputResult with timing, TFLOPS, and utilization data.
        """
        total_flops = self._count_attention_flops(seq_len, d_model, batch_size)

        # Create random test tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self._torch_dtype()
        n_heads = 8
        d_head = d_model // n_heads

        q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)

        # Warmup
        for _ in range(self.warmup_iters):
            _ = attention_fn(q, k, v, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(self.bench_iters):
            _ = attention_fn(q, k, v, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        elapsed_per_iter = elapsed / self.bench_iters
        tokens_per_second = (batch_size * seq_len) / elapsed_per_iter
        flops_per_second = total_flops / elapsed_per_iter
        achieved_tflops = flops_per_second / 1e12

        peak_key = f"peak_tflops_{self.precision}"
        peak_tflops = self._specs.get(peak_key, self._specs["peak_tflops_fp16"])
        compute_utilization_pct = (achieved_tflops / peak_tflops) * 100.0

        # Estimate memory footprint (batch * heads * seq * seq * 2 bytes for fp16)
        memory_bytes = batch_size * n_heads * seq_len * seq_len * 2

        return ThroughputResult(
            gpu_model=self.gpu_model,
            seq_len=seq_len,
            d_model=d_model,
            batch_size=batch_size,
            tokens_per_second=tokens_per_second,
            achieved_tflops=achieved_tflops,
            peak_tflops=peak_tflops,
            compute_utilization_pct=compute_utilization_pct,
            elapsed_seconds=elapsed_per_iter,
            total_flops=total_flops,
            memory_bytes=memory_bytes,
            precision=self.precision,
        )

    def get_peak_tflops(self) -> float:
        """Return the peak TFLOPS for the configured GPU and precision."""
        peak_key = f"peak_tflops_{self.precision}"
        return self._specs.get(peak_key, self._specs["peak_tflops_fp16"])

    def get_memory_bandwidth(self) -> float:
        """Return HBM bandwidth in GB/s for the configured GPU."""
        return self._specs["memory_bandwidth_gb_s"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_attention_flops(seq_len: int, d_model: int, batch_size: int) -> float:
        """Count FLOPs for a standard scaled dot-product attention forward pass.

        Counts: Q @ K^T (2 * seq^2 * d), softmax (~5 * seq^2 ops),
        P @ V (2 * seq^2 * d).
        """
        # Rough count per head — assume heads = 8
        n_heads = 8
        d_head = d_model // n_heads
        # QK^T: 2 * seq^2 * d_head per head
        # softmax: ~5 * seq^2 per head (exp + sum + div)
        # PV: 2 * seq^2 * d_head per head
        flops_per_head = seq_len * seq_len * (4 * d_head + 5)
        total_per_sample = n_heads * flops_per_head
        return total_per_sample * batch_size

    def _torch_dtype(self) -> torch.dtype:
        """Map precision string to torch dtype."""
        if self.precision == "fp16":
            return torch.float16
        elif self.precision == "bf16":
            return torch.bfloat16
        return torch.float32
