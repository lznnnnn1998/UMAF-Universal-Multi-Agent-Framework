"""
YaRN (Yet another RoPE extensioN) — combining NTK-aware interpolation with
temperature-based attention entropy control.

YaRN extends RoPE-based models to longer contexts by:
1. NTK-aware base frequency scaling (preserves high-frequency info).
2. A ramp function that selectively interpolates frequency bands:
   - Low frequencies (small i, long wavelengths): extrapolated (kept as-is)
     for long-range dependency preservation.
   - High frequencies (large i, short wavelengths): fully interpolated
     via NTK scaling to avoid aliasing.
   - Mid frequencies: smoothly blended between the two regimes.
3. Temperature scaling on attention logits: softmax(qk^T / t) where
   t is derived from the scale factor. This prevents attention scores
   from becoming too peaked (low entropy) at extended lengths.

Reference: Peng et al., "YaRN: Efficient Context Window Extension of
           Large Language Models" https://arxiv.org/abs/2309.00071
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .rope import (
    compute_rope_frequencies,
    precompute_rope_cos_sin,
    apply_rope,
)
from .ntk import compute_ntk_base


def compute_ramp(
    dim: int,
    alpha: float = 0.1,
    beta: float = 0.9,
) -> torch.Tensor:
    """Compute the YaRN ramp function values for each frequency index.

    The ramp γ(r_i) determines the interpolation ratio for frequency index i,
    where r_i = 2i / dim is the normalized frequency index in [0, 1).

    Ramp definition:
        γ(r) = 0                    for r <= α  (extrapolation region)
        γ(r) = (r-α)/(β-α)         for α < r < β  (transition region)
        γ(r) = 1                    for r >= β  (interpolation region)

    Args:
        dim: Hidden dimension (must be even).
        alpha: Lower threshold (default 0.1). Frequencies with r <= α are
               fully extrapolated (original frequencies preserved).
        beta: Upper threshold (default 0.9). Frequencies with r >= β are
              fully interpolated (NTK-scaled frequencies used).

    Returns:
        Ramp values tensor of shape (dim//2,), dtype float32, values in [0, 1].

    Raises:
        ValueError: If alpha < 0, beta > 1, or alpha >= beta.
    """
    if not (0.0 <= alpha < beta <= 1.0):
        raise ValueError(f"Must have 0 <= alpha < beta <= 1, got alpha={alpha}, beta={beta}")

    half_dim = dim // 2
    # Normalized frequency indices: r_i = i / (half_dim) for i in [0, half_dim)
    r = torch.arange(half_dim, dtype=torch.float32) / max(half_dim - 1, 1)

    ramp = torch.zeros(half_dim, dtype=torch.float32)
    # Transition region mask
    transition = (r > alpha) & (r < beta)
    ramp[transition] = (r[transition] - alpha) / (beta - alpha)
    # Full interpolation region
    ramp[r >= beta] = 1.0

    return ramp


def compute_yarn_temperature(
    scale_factor: float,
    scale_type: str = "log",
    attention_factor: float = 1.0,
) -> float:
    """Compute YaRN attention temperature for softmax scaling.

    At extended context lengths, attention scores tend to become sharper
    (lower entropy). The temperature t > 1 spreads the softmax distribution,
    counteracting this effect.

    Two scaling formulas are supported:

    "log":    t = 1 + log(α)        (mild, suitable for moderate extensions)
    "linear": t = (0.1 · α · log(α) + 1)    (stronger, for large extensions)

    The result is further multiplied by attention_factor.

    Args:
        scale_factor: α = L_extended / L_original.
        scale_type: "log" or "linear".
        attention_factor: Additional multiplier (default 1.0).

    Returns:
        Temperature value >= 1.0.

    Raises:
        ValueError: If scale_factor < 1.0.
    """
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")

    import math
    if scale_type == "log":
        t = 1.0 + math.log(scale_factor)
    elif scale_type == "linear":
        t = 0.1 * scale_factor * math.log(max(scale_factor, 1.0 + 1e-8)) + 1.0
    else:
        raise ValueError(f"Unknown scale_type '{scale_type}'. Use 'log' or 'linear'.")

    return t * attention_factor


def compute_yarn_frequencies(
    dim: int,
    scale_factor: float,
    original_base: float = 10000.0,
    ramp_alpha: float = 0.1,
    ramp_beta: float = 0.9,
) -> torch.Tensor:
    """Compute YaRN blended frequencies combining original and NTK-scaled values.

    For each frequency index i:
        θ'_i = (1 - γ_i) · θ_i + γ_i · θ_i^{NTK}

    where γ_i is the ramp value, θ_i is the original frequency, and θ_i^{NTK}
    is the NTK-scaled frequency.

    Args:
        dim: Hidden dimension (must be even).
        scale_factor: α = L_extended / L_original.
        original_base: Original RoPE base.
        ramp_alpha: Ramp lower threshold.
        ramp_beta: Ramp upper threshold.

    Returns:
        Blended frequencies tensor of shape (dim//2,), dtype float32.
    """
    # Original and NTK-scaled frequencies
    original_freqs = compute_rope_frequencies(dim, original_base)
    ntk_base = compute_ntk_base(dim, scale_factor, original_base)
    ntk_freqs = compute_rope_frequencies(dim, ntk_base)

    # Ramp determines blending ratio per frequency
    ramp = compute_ramp(dim, ramp_alpha, ramp_beta)

    # Blend: extrapolate low freqs, interpolate high freqs
    blended_freqs = (1 - ramp) * original_freqs + ramp * ntk_freqs
    return blended_freqs


def apply_yarn_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    dim: int,
    scale_factor: float,
    original_base: float = 10000.0,
    ramp_alpha: float = 0.1,
    ramp_beta: float = 0.9,
    temperature_scale_type: str = "log",
    global_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Apply YaRN RoPE to query and key tensors.

    Returns rotated query, rotated key, AND the attention temperature
    that should be used for softmax scaling:
        attention = softmax(q_rot · k_rot^T / (sqrt(d) * temperature))

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim).
        dim: Head dimension.
        scale_factor: α = L_extended / L_original.
        original_base: Original RoPE base frequency.
        ramp_alpha: Ramp lower threshold (default 0.1).
        ramp_beta: Ramp upper threshold (default 0.9).
        temperature_scale_type: "log" or "linear".
        global_positions: Optional absolute positions.

    Returns:
        Tuple of (q_rotated, k_rotated, temperature).
    """
    head_dim = q.shape[-1]
    batch, seq_len = q.shape[:2]

    # Compute YaRN blended frequencies
    freqs = compute_yarn_frequencies(
        head_dim, scale_factor, original_base, ramp_alpha, ramp_beta,
    ).to(q.device)

    if global_positions is None:
        positions = torch.arange(seq_len, dtype=torch.float32, device=q.device)
    else:
        positions = global_positions.float()

    angles = torch.outer(positions, freqs)  # (seq_len, head_dim//2)
    cos_yarn = torch.cos(angles).to(q.dtype)
    sin_yarn = torch.sin(angles).to(q.dtype)

    # Add batch and head dims for broadcasting
    cos_yarn = cos_yarn.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim//2)
    sin_yarn = sin_yarn.unsqueeze(0).unsqueeze(2)

    q_rot = apply_rope(q, cos_yarn, sin_yarn)
    k_rot = apply_rope(k, cos_yarn, sin_yarn)

    temperature = compute_yarn_temperature(scale_factor, temperature_scale_type)

    return q_rot, k_rot, temperature


class YaRNRoPE(nn.Module):
    """YaRN (Yet another RoPE extensioN) as a reusable nn.Module.

    Combines NTK-aware frequency scaling with ramp-based selective
    interpolation and temperature-controlled attention entropy.

    Args:
        dim: Head dimension (must be even).
        scale_factor: α = L_extended / L_original.
        original_base: Original RoPE base frequency.
        ramp_alpha: Lower threshold for ramp function.
        ramp_beta: Upper threshold for ramp function.
        temperature_scale_type: "log" or "linear".

    Example:
        yarn_rope = YaRNRoPE(dim=64, scale_factor=4.0)
        q_rot, k_rot, temperature = yarn_rope(q, k)
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))
        attn = torch.softmax(scores / (math.sqrt(dim) * temperature), dim=-1)
    """

    def __init__(
        self,
        dim: int,
        scale_factor: float = 4.0,
        original_base: float = 10000.0,
        ramp_alpha: float = 0.1,
        ramp_beta: float = 0.9,
        temperature_scale_type: str = "log",
    ):
        super().__init__()
        self.dim = dim
        self.scale_factor = scale_factor
        self.original_base = original_base
        self.ramp_alpha = ramp_alpha
        self.ramp_beta = ramp_beta
        self.temperature_scale_type = temperature_scale_type

        # Precompute blended frequencies and temperature
        freqs = compute_yarn_frequencies(dim, scale_factor, original_base,
                                         ramp_alpha, ramp_beta)
        self.register_buffer("freqs", freqs, persistent=False)
        self.temperature = compute_yarn_temperature(scale_factor, temperature_scale_type)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        global_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply YaRN RoPE."""
        return apply_yarn_rope(
            q, k, self.dim, self.scale_factor,
            original_base=self.original_base,
            ramp_alpha=self.ramp_alpha,
            ramp_beta=self.ramp_beta,
            temperature_scale_type=self.temperature_scale_type,
            global_positions=global_positions,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, scale_factor={self.scale_factor}, "
            f"original_base={self.original_base}, "
            f"ramp=({self.ramp_alpha}, {self.ramp_beta}), "
            f"temperature={self.temperature:.3f} ({self.temperature_scale_type})"
        )
