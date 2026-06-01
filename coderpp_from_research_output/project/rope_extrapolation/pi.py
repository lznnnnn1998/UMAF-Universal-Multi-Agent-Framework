"""
Position Interpolation (PI) for RoPE-based context extension.

PI linearly rescales position indices so that a model trained on L_original
tokens can handle L_extended tokens without fine-tuning. For scale factor
α = L_extended / L_original, position m is replaced by m/α, effectively
"squeezing" all positions into the original range.

Reference: Chen et al., "Extending Context Window of Large Language Models
           via Position Interpolation" https://arxiv.org/abs/2306.15595
"""

import torch
from typing import Tuple, Optional

from .rope import (
    compute_rope_frequencies,
    apply_rotary_pos_emb,
)


def position_interpolation_scale(
    original_max_len: int,
    extended_max_len: int,
) -> float:
    """Compute the PI scale factor α = L_extended / L_original.

    Args:
        original_max_len: The maximum sequence length the model was trained on.
        extended_max_len: The target maximum sequence length.

    Returns:
        Scale factor α >= 1.0.

    Raises:
        ValueError: If extended_max_len < original_max_len.
    """
    if extended_max_len < original_max_len:
        raise ValueError(
            f"extended_max_len ({extended_max_len}) must be >= "
            f"original_max_len ({original_max_len})"
        )
    return extended_max_len / original_max_len


def apply_pi_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    dim: int,
    scale_factor: float,
    base: float = 10000.0,
    global_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Position Interpolation RoPE to query and key tensors.

    Rescales positions by 1/α before applying RoPE rotation, effectively
    mapping positions [0, α·L_original) → [0, L_original).

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim).
        dim: Head dimension (must match q.shape[-1]).
        scale_factor: α = L_extended / L_original (>= 1.0).
        base: RoPE base frequency (default 10000.0).
        global_positions: Optional absolute positions for each token.
                          If None, sequential positions [0, seq_len) are used.

    Returns:
        Tuple of (q_rotated, k_rotated).
    """
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")

    batch, seq_len = q.shape[:2]
    head_dim = q.shape[-1]

    if global_positions is None:
        positions = torch.arange(seq_len, dtype=torch.float32, device=q.device)
    else:
        positions = global_positions.float()

    # Apply PI: rescale positions
    scaled_positions = positions / scale_factor

    # Use actual float positions for sub-position precision
    freqs = compute_rope_frequencies(head_dim, base).to(q.device)  # (dim//2,)
    angles = torch.outer(scaled_positions, freqs)  # (seq_len, dim//2)
    cos_pi = torch.cos(angles).to(q.dtype)
    sin_pi = torch.sin(angles).to(q.dtype)

    # Add batch and head dims
    cos_pi = cos_pi.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim//2)
    sin_pi = sin_pi.unsqueeze(0).unsqueeze(2)

    from .rope import apply_rope
    q_rot = apply_rope(q, cos_pi, sin_pi)
    k_rot = apply_rope(k, cos_pi, sin_pi)

    return q_rot, k_rot


class PIRoPE(torch.nn.Module):
    """Position Interpolation RoPE as a reusable nn.Module.

    Precomputes frequency tables and applies PI-based position rescaling.

    Args:
        dim: Head dimension (must be even).
        original_max_len: Original training context length.
        extended_max_len: Target extended context length.
        base: RoPE base frequency.

    Example:
        pi_rope = PIRoPE(dim=64, original_max_len=2048, extended_max_len=8192)
        q_rot, k_rot = pi_rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        original_max_len: int = 2048,
        extended_max_len: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.original_max_len = original_max_len
        self.extended_max_len = extended_max_len
        self.base = base
        self.scale_factor = position_interpolation_scale(
            original_max_len, extended_max_len
        )
        # Precompute frequencies
        freqs = compute_rope_frequencies(dim, base)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        global_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply PI-RoPE (see apply_pi_rope for argument details)."""
        return apply_pi_rope(
            q, k, self.dim, self.scale_factor,
            base=self.base, global_positions=global_positions,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, original_max_len={self.original_max_len}, "
            f"extended_max_len={self.extended_max_len}, "
            f"scale_factor={self.scale_factor:.2f}, base={self.base}"
        )
