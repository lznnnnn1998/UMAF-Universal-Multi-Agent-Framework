"""
NTK-aware Scaling for RoPE context extension.

NTK-aware scaling adjusts the RoPE base frequency rather than linearly
interpolating positions. By changing base → base·α^{d/(d-2)}, low-frequency
components (important for long-range dependencies) are scaled more than
high-frequency components (important for local patterns). This preserves
high-frequency information that PI would blur, while extending the effective
context window.

Reference: "NTK-Aware Scaled RoPE allows LLaMA to have extended context
           window without fine-tuning" (bloc97 / Reddit, 2023)
           https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .rope import (
    compute_rope_frequencies,
    precompute_rope_cos_sin,
    apply_rope,
)


def compute_ntk_base(
    dim: int,
    scale_factor: float,
    original_base: float = 10000.0,
) -> float:
    """Compute the NTK-aware scaled RoPE base frequency.

    Formula: base' = base · α^{d/(d-2)}

    The exponent d/(d-2) > 1 means the base increases faster than the scale
    factor. This has the effect of compressing high frequencies less than low
    frequencies, preserving local pattern recognition.

    Args:
        dim: Hidden dimension.
        scale_factor: α = L_extended / L_original.
        original_base: Original RoPE base (default 10000.0).

    Returns:
        New base frequency as a Python float.

    Raises:
        ValueError: If dim <= 2 (exponent would be undefined or infinite).
        ValueError: If scale_factor < 1.0.
    """
    if dim <= 2:
        raise ValueError(f"dim must be > 2 for NTK scaling, got {dim}")
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")

    exponent = dim / (dim - 2)
    new_base = original_base * (scale_factor ** exponent)
    return new_base


def compute_ntk_frequencies(
    dim: int,
    scale_factor: float,
    original_base: float = 10000.0,
) -> torch.Tensor:
    """Compute RoPE frequencies with NTK-aware scaled base.

    Args:
        dim: Hidden dimension (must be even).
        scale_factor: α = L_extended / L_original.
        original_base: Original RoPE base.

    Returns:
        Frequencies tensor of shape (dim//2,).
    """
    new_base = compute_ntk_base(dim, scale_factor, original_base)
    return compute_rope_frequencies(dim, new_base)


def apply_ntk_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    dim: int,
    scale_factor: float,
    original_base: float = 10000.0,
    global_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply NTK-aware scaled RoPE to query and key tensors.

    Uses a modified base frequency to extend context without explicit
    position interpolation. Unlike PI, NTK-aware scaling preserves more
    high-frequency information while extending low-frequency coverage.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim).
        dim: Head dimension.
        scale_factor: α = L_extended / L_original.
        original_base: Original RoPE base frequency.
        global_positions: Optional absolute position tensor.

    Returns:
        Tuple of (q_rotated, k_rotated).
    """
    batch, seq_len = q.shape[:2]
    head_dim = q.shape[-1]

    new_base = compute_ntk_base(head_dim, scale_factor, original_base)
    freqs = compute_rope_frequencies(head_dim, new_base).to(q.device)

    if global_positions is None:
        positions = torch.arange(seq_len, dtype=torch.float32, device=q.device)
    else:
        positions = global_positions.float()

    angles = torch.outer(positions, freqs)  # (seq_len, head_dim//2)
    cos_ntk = torch.cos(angles).to(q.dtype)
    sin_ntk = torch.sin(angles).to(q.dtype)

    # Add batch and head dims
    cos_ntk = cos_ntk.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim//2)
    sin_ntk = sin_ntk.unsqueeze(0).unsqueeze(2)

    q_rot = apply_rope(q, cos_ntk, sin_ntk)
    k_rot = apply_rope(k, cos_ntk, sin_ntk)

    return q_rot, k_rot


def frequency_analysis(
    dim: int,
    scale_factor: float,
    original_base: float = 10000.0,
) -> dict[str, torch.Tensor]:
    """Compare original vs NTK-aware frequencies for analysis.

    Returns a dict with 'original' and 'ntk' frequency tensors, plus
    the 'ratio' (ntk/original) showing how much each frequency changes.

    Args:
        dim: Hidden dimension.
        scale_factor: α = L_extended / L_original.
        original_base: Original RoPE base.

    Returns:
        Dict with keys 'original', 'ntk', 'ratio' — each (dim//2,).
    """
    original_freqs = compute_rope_frequencies(dim, original_base)
    ntk_freqs = compute_ntk_frequencies(dim, scale_factor, original_base)
    ratio = ntk_freqs / original_freqs
    return {
        "original": original_freqs,
        "ntk": ntk_freqs,
        "ratio": ratio,
    }


class NTKAwareRoPE(nn.Module):
    """NTK-aware scaled RoPE as a reusable nn.Module.

    Args:
        dim: Head dimension (must be even).
        scale_factor: α = L_extended / L_original.
        original_base: Original RoPE base frequency.

    Example:
        ntk_rope = NTKAwareRoPE(dim=64, scale_factor=4.0)
        q_rot, k_rot = ntk_rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        scale_factor: float = 4.0,
        original_base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.scale_factor = scale_factor
        self.original_base = original_base
        self.new_base = compute_ntk_base(dim, scale_factor, original_base)
        freqs = compute_rope_frequencies(dim, self.new_base)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        global_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply NTK-aware RoPE."""
        return apply_ntk_rope(
            q, k, self.dim, self.scale_factor,
            original_base=self.original_base,
            global_positions=global_positions,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, scale_factor={self.scale_factor}, "
            f"original_base={self.original_base}, new_base={self.new_base:.2f}"
        )
