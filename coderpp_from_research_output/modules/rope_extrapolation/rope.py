"""
Rotary Position Embeddings (RoPE) — core implementation.

RoPE encodes position information by rotating query and key vectors in 2D subspaces.
For position m and dimension pair (2i, 2i+1), the rotation angle is m * θ_i where
θ_i = base^{-2i/d}. This ensures the dot product q_m^T k_n depends only on the
relative position (m - n), giving the model translation-invariant positional awareness.

This module provides two APIs:
- Half-dim API (dim//2): Optimal for pre-computation; used by pi/ntk/yarn/dpe modules.
- Full-dim API (dim): Convenient for standard transformer usage; used by extrapolation.py.

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           https://arxiv.org/abs/2104.09864
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


# =============================================================================
# Half-dim API (dim//2 tensors) — used internally by pi, ntk, yarn, dpe
# =============================================================================

def compute_rope_frequencies(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Compute RoPE base frequencies θ_i = base^{-2i/dim}.

    Frequencies decrease exponentially with i, giving a geometric progression
    of wavelengths from 2π to 2π·base. Low i (low frequencies) capture long-range
    dependencies; high i (high frequencies) capture local patterns.

    Args:
        dim: Hidden dimension (must be even — rotary pairs require it).
        base: Base of the geometric progression (default 10000.0).

    Returns:
        Frequencies tensor of shape (dim//2,), dtype float32.

    Raises:
        ValueError: If dim is not even.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even for RoPE (pairwise rotation), got {dim}")
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    freqs = base ** (-i / dim)
    return freqs


def _precompute_rope_cos_sin_half(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine lookup tables (half-dim format).

    Returns cos/sin of shape (max_seq_len, dim//2) — each row stores one
    value per frequency pair. Used internally by the stacked apply_rope path.

    Args:
        dim: Hidden dimension (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        base: Base for frequency computation.
        dtype: Output data type.

    Returns:
        Tuple of (cos, sin), each of shape (max_seq_len, dim//2).
    """
    freqs = compute_rope_frequencies(dim, base)           # (dim//2,)
    positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
    angles = torch.outer(positions, freqs)                 # (max_seq_len, dim//2)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to a single tensor (half-dim path).

    For each position m and frequency index i, rotates the pair (x[2i], x[2i+1]):
        x'[2i]   = x[2i] · cos(m·θ_i) − x[2i+1] · sin(m·θ_i)
        x'[2i+1] = x[2i] · sin(m·θ_i) + x[2i+1] · cos(m·θ_i)

    cos and sin are broadcast to match x's leading dimensions.
    If cos has fewer dims than x, leading dims are prepended via unsqueeze(0).

    Args:
        x: Input tensor of shape (..., seq_len, dim) where dim is even.
        cos: Cosine values of shape (seq_len, dim//2) or (..., seq_len, dim//2).
        sin: Sine values matching cos shape.

    Returns:
        Rotated tensor of the same shape and dtype as x.
    """
    *leading, seq_len, d = x.shape
    half_d = d // 2

    # Split into even and odd dimension pairs
    x_even = x[..., 0::2]  # (..., seq_len, half_d)
    x_odd = x[..., 1::2]   # (..., seq_len, half_d)

    # Broadcast cos/sin to have same number of dims as x
    cos_bc = cos
    sin_bc = sin
    while cos_bc.dim() < x.dim():
        cos_bc = cos_bc.unsqueeze(0)
        sin_bc = sin_bc.unsqueeze(0)

    # Apply 2D rotation to each pair
    x_rot_even = x_even * cos_bc - x_odd * sin_bc
    x_rot_odd = x_even * sin_bc + x_odd * cos_bc

    # Interleave the rotated pairs back into original order
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (..., seq_len, half_d, 2)
    x_rot = x_rot.flatten(-2)                               # (..., seq_len, d)

    return x_rot.type_as(x)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors (standard attention pattern).

    This is the high-level interface used in transformer attention.
    It handles position ID lookup and applies the same rotation to both
    query and key projections.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim).
        cos: Precomputed cosine table of shape (max_seq_len, head_dim//2).
        sin: Precomputed sine table of shape (max_seq_len, head_dim//2).
        position_ids: Optional tensor of shape (batch, seq_len) with position
                      indices. If None, sequential positions 0..seq_len-1
                      are used for each batch item.

    Returns:
        Tuple of (q_rotated, k_rotated), each same shape/dtype as input.
    """
    batch, seq_len = q.shape[:2]

    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch, -1)

    # Gather cos/sin for the requested positions
    cos_selected = cos[position_ids]  # (batch, seq_len, head_dim//2)
    sin_selected = sin[position_ids]  # (batch, seq_len, head_dim//2)

    # Add head dimension for broadcasting
    cos_selected = cos_selected.unsqueeze(2)  # (batch, seq_len, 1, head_dim//2)
    sin_selected = sin_selected.unsqueeze(2)  # (batch, seq_len, 1, head_dim//2)

    q_rot = apply_rope(q, cos_selected, sin_selected)
    k_rot = apply_rope(k, cos_selected, sin_selected)

    return q_rot, k_rot


class RoPE(nn.Module):
    """Rotary Position Embedding as a reusable nn.Module.

    Precomputes cos/sin tables at construction time.  Call forward() with
    query, key, and optionally position_ids to apply the rotation.

    Example:
        rope = RoPE(dim=64, max_seq_len=2048, base=10000.0)
        q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        cos, sin = _precompute_rope_cos_sin_half(dim, max_seq_len, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE (see apply_rotary_pos_emb for argument details)."""
        return apply_rotary_pos_emb(q, k, self.cos, self.sin, position_ids)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"


# =============================================================================
# Full-dim API (dim tensors) — used by extrapolation.py and the unified interface
# =============================================================================

def compute_freqs(
    dim: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute RoPE base frequencies (alias with device/dtype support).

    Args:
        dim: Hidden dimension (must be even).
        base: Base of the geometric progression.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Frequencies tensor of shape (dim//2,).
    """
    freqs = compute_rope_frequencies(dim, base)
    if device is not None:
        freqs = freqs.to(device)
    return freqs.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input.

    For each pair (2i, 2i+1): x[2i] → -x[2i+1], x[2i+1] → x[2i].

    This implements the 90-degree rotation that is the core of the
    RoPE rotation formula: x' = x·cos(θ) + rotate_half(x)·sin(θ).

    Args:
        x: Input tensor of shape (..., dim) where dim is even.

    Returns:
        Tensor of same shape as x, with half dimensions rotated.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1)
    return x_rot.flatten(-2)


def precompute_rope_cos_sin(
    max_seq_len: int,
    dim: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine lookup tables (full-dim format).

    Returns cos/sin of shape (max_seq_len, dim) where each frequency value
    is duplicated to fill both positions of a 2D pair (cos[2i] == cos[2i+1]).
    This format is directly compatible with apply_rotary_emb's rotate_half path.

    Args:
        max_seq_len: Maximum sequence length to precompute for.
        dim: Hidden dimension (must be even).
        base: Base for frequency computation.
        device: Target device.
        dtype: Output data type.

    Returns:
        Tuple of (cos, sin), each of shape (max_seq_len, dim).

    Raises:
        ValueError: If dim is not even.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    freqs = compute_freqs(dim, base, device=device, dtype=dtype)  # (dim//2,)
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)  # (max_seq_len,)
    angles = torch.outer(positions, freqs)  # (max_seq_len, dim//2)

    cos_vals = torch.cos(angles).to(dtype)
    sin_vals = torch.sin(angles).to(dtype)

    # Repeat each value to fill a dimension pair: (max_seq_len, dim)
    cos_table = torch.repeat_interleave(cos_vals, 2, dim=-1)
    sin_table = torch.repeat_interleave(sin_vals, 2, dim=-1)

    return cos_table, sin_table


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Apply rotary embeddings using the rotate_half formula.

    Operates on position pairs: x' = x * cos(θ) + rotate_half(x) * sin(θ).

    cos and sin should have shape (full_seq_len, dim) with duplicated values
    per pair (i.e., the output format of precompute_rope_cos_sin).

    Args:
        x: Input tensor of shape (..., seq_len, dim) where dim is even.
        cos: Cosine table of shape (max_seq_len, dim).
        sin: Sine table of shape (max_seq_len, dim).
        offset: Starting position offset (used when x is a sub-sequence).

    Returns:
        Rotated tensor of same shape and dtype as x.

    Raises:
        ValueError: If dim is not even.
    """
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {d}")

    seq_len = x.shape[-2]

    # Slice cos/sin for the requested positions
    cos_slice = cos[offset:offset + seq_len]  # (seq_len, dim)
    sin_slice = sin[offset:offset + seq_len]  # (seq_len, dim)

    # Add leading dims to match x
    while cos_slice.dim() < x.dim():
        cos_slice = cos_slice.unsqueeze(0)
        sin_slice = sin_slice.unsqueeze(0)

    return (x * cos_slice) + (rotate_half(x) * sin_slice)


def apply_rotary_emb_single(
    x: torch.Tensor,
    position: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE to a single token at a given position.

    Computes cos/sin on-the-fly for the given position, making it suitable
    for autoregressive inference where you only need one position at a time.

    Args:
        x: Input tensor of shape (..., dim) where dim is even.
        position: Absolute position index.
        base: RoPE base frequency.

    Returns:
        Rotated tensor of same shape and dtype as x.

    Raises:
        ValueError: If dim is not even.
    """
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {dim}")

    freqs = compute_freqs(dim, base, device=x.device, dtype=torch.float32)  # (dim//2,)
    angles = position * freqs  # (dim//2,)
    cos_vals = torch.cos(angles).to(x.dtype)
    sin_vals = torch.sin(angles).to(x.dtype)

    # Duplicate for each pair
    cos_full = torch.repeat_interleave(cos_vals, 2)  # (dim,)
    sin_full = torch.repeat_interleave(sin_vals, 2)  # (dim,)

    return x * cos_full + rotate_half(x) * sin_full


def get_rope_embeddings(
    positions: torch.Tensor,
    dim: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get cos/sin tables for specific position indices.

    Useful when you have non-contiguous or custom position IDs.

    Args:
        positions: Position indices of arbitrary shape.
        dim: Hidden dimension (must be even).
        base: RoPE base frequency.

    Returns:
        Tuple of (cos, sin), each of shape (*positions.shape, dim).

    Raises:
        ValueError: If dim is not even.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    freqs = compute_freqs(dim, base, device=positions.device, dtype=torch.float32)  # (dim//2,)
    # Outer product: flatten positions, then reshape
    pos_flat = positions.float().reshape(-1)  # (N,)
    angles = torch.outer(pos_flat, freqs)      # (N, dim//2)

    cos_vals = torch.cos(angles)  # float32
    sin_vals = torch.sin(angles)  # float32

    # Duplicate for each pair
    cos_full = torch.repeat_interleave(cos_vals, 2, dim=-1)  # (N, dim)
    sin_full = torch.repeat_interleave(sin_vals, 2, dim=-1)  # (N, dim)

    # Reshape back to match input positions shape
    target_shape = (*positions.shape, dim)
    cos_full = cos_full.reshape(target_shape)
    sin_full = sin_full.reshape(target_shape)

    return cos_full, sin_full


def numpy_apply_rotary_emb(
    x: np.ndarray,
    pos: int,
    base: float = 10000.0,
) -> np.ndarray:
    """NumPy reference implementation of RoPE — element-wise rotation per pair.

    This is a pure NumPy, non-vectorized reference used for testing and
    validation. It applies the exact 2×2 rotation to each dimension pair.

    Args:
        x: Input vector of shape (dim,) where dim is even.
        pos: Position index.
        base: RoPE base frequency.

    Returns:
        Rotated vector of same shape as x.

    Raises:
        ValueError: If dim is not even.
    """
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"Dimension must be even, got {dim}")

    result = x.copy()
    for i in range(0, dim, 2):
        theta = 1.0 / (base ** (i / dim))
        angle = pos * theta
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # 2D rotation
        result[i] = x[i] * cos_a - x[i + 1] * sin_a
        result[i + 1] = x[i] * sin_a + x[i + 1] * cos_a
    return result
