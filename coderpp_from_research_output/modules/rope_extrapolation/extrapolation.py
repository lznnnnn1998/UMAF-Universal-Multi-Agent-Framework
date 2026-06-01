"""
Length extrapolation methods for Rotary Position Embeddings.

Implements four methods for extending RoPE-based models beyond their
pre-training context length:

1. **Position Interpolation (PI)**: Linearly rescales positions so they
   fall within the pre-training range. Simple but can lose high-frequency info.

2. **NTK-aware Scaling**: Adjusts the RoPE base frequency to redistribute
   frequency bands, preserving high-frequency resolution.

3. **YaRN**: Combines NTK-aware interpolation with a temperature factor
   on attention logits to control the "sharpness" of attention at long range.

4. **DPE (Dynamic Position Encoding)**: Training-free 128K extrapolation
   using chunked position assignment and local/global attention masks.

References:
- PI: Chen et al., "Extending Context Window of Large Language Models
  via Position Interpolation" (arXiv:2306.15595)
- NTK: "NTK-Aware Scaled RoPE" (Reddit / blog posts, 2023)
- YaRN: Peng et al., "YaRN: Efficient Context Window Extension of
  Large Language Models" (arXiv:2309.00071)
- DPE: Based on chunked position assignment and attention masking patterns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .rope import compute_freqs, precompute_rope_cos_sin


# ---------------------------------------------------------------------------
# Position Interpolation (PI)
# ---------------------------------------------------------------------------

@dataclass
class PIScaler:
    """Position Interpolation scaler.

    Linearly rescales positions by a factor `scale`. For a model trained on
    L positions, PI supports up to L * scale positions by mapping each new
    position p to p/scale.

    Attributes:
        scale: Scaling factor (e.g., 2.0 for 2x context extension).
        base: RoPE base frequency (default 10000.0).
    """

    scale: float
    base: float = 10000.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    def get_scaled_position(self, position: int) -> float:
        """Return the linearly interpolated position.

        Args:
            position: Original position index (0-indexed).

        Returns:
            Scaled position, p' = p / scale.
        """
        return position / self.scale

    def get_scaled_positions(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """Return linearly interpolated positions for a tensor.

        Args:
            positions: Tensor of position indices.

        Returns:
            Scaled positions as float tensor.
        """
        return positions.float() / self.scale

    def precompute_cos_sin(
        self,
        max_seq_len: int,
        dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute cos/sin tables using PI-scaled positions.

        Args:
            max_seq_len: Maximum sequence length (original, pre-scaling).
            dim: Head dimension (must be even).
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple of (cos_table, sin_table).
        """
        positions = torch.arange(max_seq_len, device=device, dtype=dtype)
        scaled_positions = positions / self.scale  # effectively smaller angles

        freqs = compute_freqs(dim, self.base, device=device, dtype=dtype)
        angles = torch.outer(scaled_positions, freqs)

        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        cos_table = torch.repeat_interleave(cos_vals, 2, dim=-1)
        sin_table = torch.repeat_interleave(sin_vals, 2, dim=-1)

        return cos_table, sin_table


# ---------------------------------------------------------------------------
# NTK-aware Scaling
# ---------------------------------------------------------------------------

@dataclass
class NTKScaler:
    """NTK-aware RoPE scaling.

    Instead of downscaling positions (which compresses all frequencies equally),
    NTK-aware scaling adjusts the RoPE base frequency so that high frequencies
    are less affected than low frequencies. The new base is:

        base' = base * α^{d/(d-2)}

    where α is the scaling factor and d is the head dimension.

    This preserves high-frequency resolution (important for local attention)
    while extending low-frequency coverage (important for long-range attention).

    Attributes:
        scale: Scaling factor α (e.g., 2.0 for 2x context extension).
        base: Original RoPE base frequency (default 10000.0).
    """

    scale: float
    base: float = 10000.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    def get_ntk_base(self, dim: int) -> float:
        """Compute the NTK-scaled base frequency.

        Args:
            dim: Head dimension (must be even).

        Returns:
            Scaled base frequency.
        """
        if dim <= 2:
            # Degenerate case: no pairs to adjust
            return self.base * self.scale
        alpha = self.scale
        exponent = dim / (dim - 2)
        return self.base * (alpha ** exponent)

    def precompute_cos_sin(
        self,
        max_seq_len: int,
        dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute cos/sin tables using NTK-scaled base frequency.

        Args:
            max_seq_len: Maximum sequence length.
            dim: Head dimension (must be even).
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple of (cos_table, sin_table).
        """
        ntk_base = self.get_ntk_base(dim)
        return precompute_rope_cos_sin(
            max_seq_len=max_seq_len,
            dim=dim,
            base=ntk_base,
            device=device,
            dtype=dtype,
        )

    def get_effective_scale(self, dim: int) -> float:
        """Estimate the effective context length multiplier.

        The relationship between base scaling and context extension is
        approximately: effective_scale ≈ (base'/base)^{(d-2)/d}

        Args:
            dim: Head dimension.

        Returns:
            Effective context length multiplier.
        """
        ntk_base = self.get_ntk_base(dim)
        return (ntk_base / self.base) ** ((dim - 2) / dim)


# ---------------------------------------------------------------------------
# YaRN (Yet another RoPE extensioN)
# ---------------------------------------------------------------------------

@dataclass
class YaRNScaler:
    """YaRN: NTK-aware interpolation + temperature-based attention control.

    YaRN extends NTK-aware scaling with two additional components:

    1. **NTK-aware interpolation**: Same as NTKScaler, adjusting the RoPE base.
    2. **Temperature scaling**: Multiplies attention logits by 1/t before softmax,
       controlling the "sharpness" (entropy) of the attention distribution at
       longer contexts. Higher t → more uniform attention → better length extrapolation.

    The temperature is typically set as:
        t = min(scale^{0.25}, 32.0)

    For scale=2, t ≈ 1.19; for scale=4, t ≈ 1.41.
    Temperature > 1 spreads attention, counteracting the sharpening
    that occurs at extended context lengths.

    Reference: Peng et al., arXiv:2309.00071

    Attributes:
        scale: Scaling factor α.
        base: Original RoPE base frequency.
        temperature: Attention temperature. If None, computed as min(scale^{0.25}, 32.0).
    """

    scale: float
    base: float = 10000.0
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.temperature is None:
            # Default YaRN temperature formula
            self.temperature = min(self.scale ** 0.25, 32.0)
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def get_ntk_base(self, dim: int) -> float:
        """Compute the NTK-scaled base (same as NTKScaler)."""
        if dim <= 2:
            return self.base * self.scale
        alpha = self.scale
        exponent = dim / (dim - 2)
        return self.base * (alpha ** exponent)

    def precompute_cos_sin(
        self,
        max_seq_len: int,
        dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute cos/sin tables using YaRN-scaled parameters.

        Args:
            max_seq_len: Maximum sequence length.
            dim: Head dimension (must be even).
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple of (cos_table, sin_table).
        """
        ntk_base = self.get_ntk_base(dim)
        return precompute_rope_cos_sin(
            max_seq_len=max_seq_len,
            dim=dim,
            base=ntk_base,
            device=device,
            dtype=dtype,
        )

    def scale_attention_logits(
        self, attn_logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply temperature scaling to attention logits.

        Args:
            attn_logits: Raw attention logits (Q @ K^T) of shape (*, seq, seq).

        Returns:
            Temperature-scaled logits: attn_logits / temperature.
        """
        return attn_logits / self.temperature

    def get_effective_scale(self, dim: int) -> float:
        """Estimate the effective context length multiplier."""
        ntk_base = self.get_ntk_base(dim)
        return (ntk_base / self.base) ** ((dim - 2) / dim)


# ---------------------------------------------------------------------------
# DPE: Dynamic Position Encoding
# ---------------------------------------------------------------------------

@dataclass
class DPEConfig:
    """Configuration for Dynamic Position Encoding.

    DPE enables training-free extrapolation to 128K tokens by:
    1. Splitting the sequence into chunks.
    2. Assigning positions within chunks (0 to chunk_size-1), resetting
       at each chunk boundary.
    3. Using a combined local (within-chunk) + global (cross-chunk) attention mask.

    The local attention uses standard RoPE positions within the chunk.
    The global attention connects chunk representatives.

    Attributes:
        chunk_size: Size of each chunk (default 2048, matching typical pre-training length).
        max_chunks: Maximum number of chunks (default 64, for 128K context with chunk_size=2048).
        base: RoPE base frequency.
        local_attention_radius: If set, restricts local attention to this many
            neighboring tokens (sliding window).
        global_token_per_chunk: Number of tokens per chunk used for global attention
            (e.g., the first and last token of each chunk).
    """

    chunk_size: int = 2048
    max_chunks: int = 64
    base: float = 10000.0
    local_attention_radius: int | None = None
    global_token_per_chunk: int = 2


def _compute_chunk_positions(
    total_len: int,
    chunk_size: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute chunk-local position indices for a sequence.

    For each absolute position p, the chunk-local position is p % chunk_size.

    Args:
        total_len: Total sequence length.
        chunk_size: Chunk size.
        device: Target device.

    Returns:
        Tensor of shape (total_len,) with chunk-local positions 0..chunk_size-1.
    """
    positions = torch.arange(total_len, device=device)
    return positions % chunk_size


def build_dpe_mask(
    seq_len: int,
    config: DPEConfig,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a combined local+global attention mask for DPE.

    The mask has shape (seq_len, seq_len) with:
    - 0 for allowed attention pairs
    - -inf (or very negative) for disallowed pairs

    Local attention: each token attends to tokens within the same chunk
    (optionally restricted to a radius).

    Global attention: designated global tokens (first/last of each chunk)
    attend to all global tokens across chunks.

    Args:
        seq_len: Total sequence length.
        config: DPE configuration.
        device: Target device.

    Returns:
        Attention mask of shape (seq_len, seq_len), additive (0 = allow, -inf = block).
    """
    chunk_size = config.chunk_size
    radius = config.local_attention_radius
    global_per_chunk = config.global_token_per_chunk

    # Initialize mask: all zeros (all pairs allowed)
    mask = torch.zeros(seq_len, seq_len, device=device)

    # --- Local attention mask ---
    positions = torch.arange(seq_len, device=device)

    # Chunk index for each position
    chunk_ids = positions // chunk_size  # (seq_len,)

    # Distance matrix
    dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (seq_len, seq_len)

    # Same-chunk mask
    same_chunk = (chunk_ids.unsqueeze(0) == chunk_ids.unsqueeze(1))  # (seq_len, seq_len)

    # Block cross-chunk attention
    cross_chunk_mask = ~same_chunk  # positions NOT in the same chunk
    mask[cross_chunk_mask] = float("-inf")

    # Optionally apply local radius within chunk
    if radius is not None:
        beyond_radius = same_chunk & (dist > radius)
        mask[beyond_radius] = float("-inf")

    # --- Global attention ---
    # Designate first and last `global_per_chunk` tokens of each chunk as global
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    # Global token positions
    global_positions = set()
    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, seq_len)
        # First global_per_chunk tokens
        for i in range(global_per_chunk):
            if start + i < end:
                global_positions.add(start + i)
        # Last global_per_chunk tokens
        for i in range(global_per_chunk):
            if end - 1 - i >= start:
                global_positions.add(end - 1 - i)

    # Allow global-to-global attention across chunks
    if global_positions:
        global_idx = torch.tensor(
            sorted(global_positions), device=device, dtype=torch.long
        )  # (num_global,)

        # Global tokens attend to all other global tokens
        mask[global_idx.unsqueeze(1), global_idx.unsqueeze(0)] = 0

        # All tokens (local) attend to global tokens
        mask[:, global_idx] = 0

    return mask


def dpe_precompute_cos_sin(
    config: DPEConfig,
    dim: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for DPE chunk-local positions.

    Since positions cycle within [0, chunk_size), we only need tables
    for chunk_size positions.

    Args:
        config: DPE configuration.
        dim: Head dimension (must be even).
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tuple of (cos_table, sin_table), each of shape (chunk_size, dim).
    """
    return precompute_rope_cos_sin(
        max_seq_len=config.chunk_size,
        dim=dim,
        base=config.base,
        device=device,
        dtype=dtype,
    )


def apply_dpe_rotary(
    x: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    config: DPEConfig,
    offset: int = 0,
) -> torch.Tensor:
    """Apply DPE-style rotary embeddings using chunk-local positions.

    Each token's position within its chunk determines the RoPE angle.
    Position = (absolute_position + offset) % chunk_size.

    Args:
        x: Input tensor of shape (..., seq_len, dim).
        cos_table: Cosine table of shape (chunk_size, dim).
        sin_table: Sine table of shape (chunk_size, dim).
        config: DPE configuration.
        offset: Sequence-level position offset.

    Returns:
        Tensor of same shape as x with DPE rotary embeddings applied.
    """
    from .rope import apply_rotary_emb, rotate_half

    seq_len = x.shape[-2]
    dim = x.shape[-1]
    chunk_size = config.chunk_size

    # Chunk-local positions for this sequence slice
    positions = (torch.arange(seq_len, device=x.device) + offset) % chunk_size

    # Gather per-position cos/sin
    cos_slice = cos_table[positions]  # (seq_len, dim)
    sin_slice = sin_table[positions]  # (seq_len, dim)

    while cos_slice.dim() < x.dim():
        cos_slice = cos_slice.unsqueeze(0)
        sin_slice = sin_slice.unsqueeze(0)

    x_rotated = rotate_half(x)
    return x * cos_slice + x_rotated * sin_slice


# ---------------------------------------------------------------------------
# Unified helper: get the right cos/sin tables for a method
# ---------------------------------------------------------------------------

def get_cos_sin_for_method(
    method: str,
    max_seq_len: int,
    dim: int,
    scale: float = 2.0,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get precomputed cos/sin tables for a given extrapolation method.

    Args:
        method: One of "base", "pi", "ntk", "yarn".
        max_seq_len: Maximum sequence length.
        dim: Head dimension.
        scale: Scaling factor (for PI, NTK, YaRN).
        base: RoPE base frequency.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tuple of (cos_table, sin_table).
    """
    if method == "base":
        return precompute_rope_cos_sin(max_seq_len, dim, base, device, dtype)
    elif method == "pi":
        scaler = PIScaler(scale, base)
        return scaler.precompute_cos_sin(max_seq_len, dim, device, dtype)
    elif method == "ntk":
        scaler = NTKScaler(scale, base)
        return scaler.precompute_cos_sin(max_seq_len, dim, device, dtype)
    elif method == "yarn":
        scaler = YaRNScaler(scale, base)
        return scaler.precompute_cos_sin(max_seq_len, dim, device, dtype)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from 'base', 'pi', 'ntk', 'yarn'."
        )


def compare_angles(
    dim: int = 64,
    max_pos: int = 4096,
    scale: float = 2.0,
    base: float = 10000.0,
) -> dict[str, torch.Tensor]:
    """Compare rotation angles across different methods for analysis.

    Computes angles = position * θ_i for the last position across all
    frequency bands.

    Args:
        dim: Head dimension.
        max_pos: Maximum position to compare.
        scale: Scaling factor.
        base: Base frequency.

    Returns:
        Dictionary mapping method name → angle tensor of shape (dim//2,).
    """
    freqs = compute_freqs(dim, base)

    results: dict[str, torch.Tensor] = {}

    # Base: angles for position max_pos-1
    results["base"] = (max_pos - 1) * freqs

    # PI: scaled position
    results["pi"] = (max_pos - 1) / scale * freqs

    # NTK: scaled base
    ntk = NTKScaler(scale, base)
    ntk_freqs = compute_freqs(dim, ntk.get_ntk_base(dim))
    results["ntk"] = (max_pos - 1) * ntk_freqs

    # YaRN: same base as NTK
    yarn = YaRNScaler(scale, base)
    yarn_freqs = compute_freqs(dim, yarn.get_ntk_base(dim))
    results["yarn"] = (max_pos - 1) * yarn_freqs

    return results
