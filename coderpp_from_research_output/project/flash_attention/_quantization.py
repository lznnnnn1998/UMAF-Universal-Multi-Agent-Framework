"""
FP8 quantization utilities for FlashAttention v3.

Implements FP8 E4M3 format simulation:
  - 1 sign bit, 4 exponent bits, 3 mantissa bits
  - Exponent bias: 7
  - Range: max normal ~240, min subnormal ~1.95e-3

Used in FlashAttention v3 to simulate reduced-precision GEMM operations
that save memory bandwidth and enable higher throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class FP8Config:
    """Configuration for FP8 E4M3 quantization.

    Attributes:
        ebits: Number of exponent bits (4 for E4M3).
        mbits: Number of mantissa bits (3 for E4M3).
        bias: Exponent bias (7 for E4M3).
        max_normal: Maximum representable normal value.
        min_normal: Minimum positive normal value.
        min_subnormal: Minimum positive subnormal value.
        use_subnormals: Whether to support subnormal numbers.
    """

    ebits: int = 4
    mbits: int = 3
    bias: int = 7

    @property
    def total_bits(self) -> int:
        """Total bits including sign (but sign is handled separately)."""
        return 1 + self.ebits + self.mbits

    @property
    def max_normal(self) -> float:
        """Maximum representable normal value: (2 - 2^{-mbits}) * 2^{emax}."""
        emax = 2**self.ebits - 2 - self.bias  # reserve all-1s for inf/nan
        return float((2.0 - 2.0 ** (-self.mbits)) * (2.0**emax))

    @property
    def min_normal(self) -> float:
        """Minimum positive normal value: 2^{1 - bias}."""
        return float(2.0 ** (1 - self.bias))

    @property
    def min_subnormal(self) -> float:
        """Minimum positive subnormal value: 2^{1 - bias - mbits}."""
        return float(2.0 ** (1 - self.bias - self.mbits))


def _compute_fp8_levels(config: FP8Config) -> torch.Tensor:
    """Compute all representable positive FP8 E4M3 values.

    Returns a sorted tensor of all positive quantized levels.
    """
    levels = []
    # Normal numbers: exponent from 1 to 2^ebits - 2 (exclude 0 and all-1s)
    for exp_bits in range(1, 2**config.ebits - 1):
        exp_val = exp_bits - config.bias
        for mant_bits in range(2**config.mbits):
            mantissa = 1.0 + mant_bits / (2**config.mbits)
            levels.append(mantissa * (2.0**exp_val))

    # Subnormal numbers: exponent bits = 0
    exp_val = 1 - config.bias  # = -6 for E4M3
    for mant_bits in range(1, 2**config.mbits):  # skip 0
        mantissa = mant_bits / (2**config.mbits)
        levels.append(mantissa * (2.0**exp_val))

    # Include 0
    levels.append(0.0)

    return torch.tensor(sorted(set(levels)), dtype=torch.float32)


# Pre-compute FP8 levels for default config
_FP8_DEFAULT_LEVELS: torch.Tensor = _compute_fp8_levels(FP8Config())
_FP8_CUSTOM_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def _get_fp8_levels(config: FP8Config | None = None) -> torch.Tensor:
    """Get or compute FP8 representable levels (cached per config).

    Uses a pre-computed default and a dict cache for custom configs
    to avoid recomputing FP8 levels on every inner-loop call.
    """
    if config is None:
        config = FP8Config()
    key = (config.ebits, config.mbits, config.bias)
    if key == (4, 3, 7):  # default E4M3 config
        return _FP8_DEFAULT_LEVELS
    if key not in _FP8_CUSTOM_CACHE:
        _FP8_CUSTOM_CACHE[key] = _compute_fp8_levels(config)
    return _FP8_CUSTOM_CACHE[key]


def quantize_fp8_e4m3(
    x: torch.Tensor,
    config: FP8Config | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to simulated FP8 E4M3 format.

    Implements round-to-nearest-even quantization. Returns both the
    quantized representation (as float32 holding discrete FP8 values)
    and per-tensor scaling factors for dequantization.

    This simulates FP8 GEMM in FlashAttention v3 where Q, K, V are
    stored and multiplied in reduced precision to save memory bandwidth.

    Args:
        x: Input float tensor of any shape.
        config: FP8 format configuration (default: E4M3).

    Returns:
        x_q: Quantized tensor (same shape, values rounded to FP8 levels).
        scale: Per-tensor scaling factor used (always 1.0 for basic sim).
    """
    if config is None:
        config = FP8Config()

    levels = _get_fp8_levels(config)

    # Find the absolute value for quantization level lookup
    abs_x = torch.abs(x)

    # For each value, find the nearest FP8 level
    # Reshape levels for broadcasting: [1, 1, ..., n_levels]
    shape = [1] * abs_x.ndim + [-1]
    levels_flat = levels.to(x.device).reshape(shape)

    # Expand abs_x for comparison: [..., 1]
    abs_expanded = abs_x.unsqueeze(-1)  # [..., 1]

    # Find nearest level by minimum absolute difference
    diffs = torch.abs(abs_expanded - levels_flat)  # [..., n_levels]
    nearest_idx = diffs.argmin(dim=-1)  # [...]

    # Gather the nearest levels
    x_q_abs = levels.to(x.device)[nearest_idx]

    # Restore sign
    x_q = torch.where(x >= 0, x_q_abs, -x_q_abs)

    # Clamp to max representable value (handle overflow to inf/nan)
    max_val = config.max_normal
    x_q = torch.clamp(x_q, -max_val, max_val)

    return x_q, torch.tensor(1.0, device=x.device)


def dequantize_fp8_e4m3(
    x_q: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize from FP8 E4M3 representation back to float32.

    Args:
        x_q: Quantized tensor (values at FP8 representable levels).
        scale: Per-tensor or per-channel scale factor (default: 1.0).

    Returns:
        Dequantized float32 tensor.
    """
    if scale is None:
        return x_q.float()
    return (x_q * scale).float()


def simulate_fp8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    config: FP8Config | None = None,
) -> torch.Tensor:
    """Simulate an FP8 matrix multiplication.

    Quantizes both inputs to FP8, performs the matmul in higher precision
    (mimicking FP8 inputs + FP16/FP32 accumulation), and returns the result.

    This simulates the FP8 GEMM operations used in FlashAttention v3 for
    computing S = Q @ K^T with reduced-precision operands.

    Args:
        a: First matrix, shape [..., M, K].
        b: Second matrix, shape [..., K, N] (will be transposed for S computation,
            or use directly for [K, N] format).
        config: FP8 configuration.

    Returns:
        Result of a @ b with FP8-simulated inputs, shape [..., M, N].
    """
    a_q, _ = quantize_fp8_e4m3(a, config)
    b_q, _ = quantize_fp8_e4m3(b, config)

    # Multiply dequantized values in float32 (simulating FP8 inputs with
    # higher-precision accumulation, as real hardware would do)
    return torch.matmul(a_q.float(), b_q.float())
