"""
Dynamic Position Encoding (DPE) — training-free 128K context extrapolation.

DPE extends RoPE-based models to ultra-long sequences (128K+ tokens) without
any fine-tuning by modifying how position IDs are assigned and how attention
is masked:

1. **Chunked Position Assignment**: The long sequence is split into fixed-size
   chunks. Within each chunk, position IDs reset to 0,1,...,C-1. This means
   RoPE only needs to handle relative positions within a chunk, not the full
   sequence length.

2. **Local Attention Masking**: Non-global tokens attend only within their
   local chunk (plus global tokens), preventing confusion from incorrect
   cross-chunk relative positions.

3. **Global Tokens**: A small set of tokens (typically the first few) can
   attend to and be attended by ALL tokens, preserving long-range information
   flow across chunks.

This design allows models trained on, e.g., 4K context to handle 128K
sequences, as the model only ever encounters position IDs ≤ chunk_size.

Reference: DPE is inspired by methods like StreamingLLM, LongChat, and
           various chunked-attention approaches for length extrapolation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .rope import (
    compute_rope_frequencies,
    _precompute_rope_cos_sin_half,
    apply_rope,
    precompute_rope_cos_sin,
)


def compute_dpe_position_ids(
    seq_len: int,
    chunk_size: int,
    num_global_tokens: int = 0,
    shift: int = 0,
) -> torch.Tensor:
    """Compute DPE position IDs for a sequence of given length.

    Position assignment strategy:
    - Global tokens (first num_global_tokens): use original positions [0, G-1].
    - Non-global tokens: use chunked positions, resetting each chunk.
      Position within chunk = (global_index - G) % chunk_size.

    Args:
        seq_len: Total sequence length.
        chunk_size: Size of each local chunk (e.g., 2048).
        num_global_tokens: Number of leading global tokens (default 0).
        shift: Optional shift to add to chunked positions (default 0).

    Returns:
        Position IDs tensor of shape (seq_len,), dtype int64.

    Raises:
        ValueError: If chunk_size <= 0 or num_global_tokens < 0
                   or num_global_tokens > seq_len.

    Example:
        >>> compute_dpe_position_ids(10, chunk_size=4, num_global_tokens=2)
        tensor([0, 1, 0, 1, 2, 3, 0, 1, 2, 3])
        # Tokens 0,1 are global; tokens 2-5 are chunk 0 (4 tokens);
        # tokens 6-9 are chunk 1 (4 tokens)
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if num_global_tokens < 0:
        raise ValueError(f"num_global_tokens must be >= 0, got {num_global_tokens}")
    if num_global_tokens > seq_len:
        raise ValueError(
            f"num_global_tokens ({num_global_tokens}) cannot exceed seq_len ({seq_len})"
        )

    if seq_len <= num_global_tokens:
        # All tokens are global
        return torch.arange(seq_len, dtype=torch.int64)

    # Global tokens: use original positions
    global_positions = torch.arange(num_global_tokens, dtype=torch.int64)

    # Non-global tokens: chunked positions
    non_global_len = seq_len - num_global_tokens
    local_indices = torch.arange(non_global_len, dtype=torch.int64)
    chunked_positions = (local_indices % chunk_size) + shift

    position_ids = torch.cat([global_positions, chunked_positions])
    return position_ids


def build_dpe_attention_mask(
    seq_len: int,
    chunk_size: int,
    num_global_tokens: int = 0,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """Build the DPE attention mask for local + global attention.

    The mask uses the convention:
        0.0  = allowed to attend
        -inf = blocked from attending

    This is suitable for additive masking before softmax:
        attn = softmax(scores + mask)

    Mask structure:
    - Global tokens (rows 0..G-1): attend to ALL tokens (all zeros).
    - Non-global tokens: attend to global tokens + tokens in their local chunk.

    Args:
        seq_len: Total sequence length.
        chunk_size: Size of each local chunk.
        num_global_tokens: Number of leading global tokens.
        mask_value: Value for blocked positions (default -inf).

    Returns:
        Attention mask of shape (seq_len, seq_len), dtype float32.

    Raises:
        ValueError: If chunk_size <= 0 or num_global_tokens < 0.

    Example:
        >>> mask = build_dpe_attention_mask(6, chunk_size=3, num_global_tokens=1)
        >>> mask  # Row=query, Col=key
        tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],   # global token 0 sees all
                [ 0.,  0.,  0.,  0., -inf, -inf],  # token 1 sees global + chunk[1,2,3]
                [ 0.,  0.,  0.,  0., -inf, -inf],  # token 2 sees global + chunk[1,2,3]
                [ 0.,  0.,  0.,  0., -inf, -inf],  # token 3 sees global + chunk[1,2,3]
                [ 0., -inf, -inf, -inf, 0.,  0.],  # token 4 sees global + chunk[4,5]
                [ 0., -inf, -inf, -inf, 0.,  0.]]) # token 5 sees global + chunk[4,5]
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if num_global_tokens < 0:
        raise ValueError(f"num_global_tokens must be >= 0, got {num_global_tokens}")

    # Start with all positions blocked
    mask = torch.full((seq_len, seq_len), mask_value, dtype=torch.float32)

    # Global tokens (rows) can attend to everything
    if num_global_tokens > 0:
        mask[:num_global_tokens, :] = 0.0

    # All tokens can attend to global tokens (columns)
    if num_global_tokens > 0:
        mask[:, :num_global_tokens] = 0.0

    # Non-global tokens: attend within their local chunk
    non_global_start = num_global_tokens
    for start in range(non_global_start, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        mask[start:end, start:end] = 0.0

    return mask


def build_dpe_attention_mask_bool(
    seq_len: int,
    chunk_size: int,
    num_global_tokens: int = 0,
) -> torch.Tensor:
    """Build DPE attention mask as a boolean tensor.

    True = allowed to attend, False = blocked.
    Useful with PyTorch's efficient attention implementations that accept
    boolean masks.

    Args:
        seq_len: Total sequence length.
        chunk_size: Size of each local chunk.
        num_global_tokens: Number of leading global tokens.

    Returns:
        Boolean attention mask of shape (seq_len, seq_len).
    """
    mask = build_dpe_attention_mask(seq_len, chunk_size, num_global_tokens,
                                    mask_value=float("-inf"))
    return mask == 0.0


def apply_dpe_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    dim: int,
    chunk_size: int,
    num_global_tokens: int = 0,
    base: float = 10000.0,
    shift: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply DPE-style RoPE with chunked position encoding.

    Computes chunked position IDs, builds the attention mask, and applies
    RoPE using those position IDs.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim).
        dim: Head dimension.
        chunk_size: Size of each local chunk.
        num_global_tokens: Number of leading global tokens.
        base: RoPE base frequency.
        shift: Position shift for chunked positions.

    Returns:
        Tuple of (q_rotated, k_rotated, attention_mask) where attention_mask
        is of shape (seq_len, seq_len) with 0.0 for allowed, -inf for blocked.
    """
    batch, seq_len = q.shape[:2]
    head_dim = q.shape[-1]

    # Compute DPE position IDs
    position_ids = compute_dpe_position_ids(
        seq_len, chunk_size, num_global_tokens, shift
    ).to(q.device)

    # Max position needed for cos/sin tables
    max_pos = int(position_ids.max().item()) + 1
    cos, sin = _precompute_rope_cos_sin_half(head_dim, max_pos, base, dtype=q.dtype)

    # Gather cos/sin for computed position IDs
    cos_selected = cos[position_ids].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim//2)
    sin_selected = sin[position_ids].unsqueeze(0).unsqueeze(2)

    q_rot = apply_rope(q, cos_selected, sin_selected)
    k_rot = apply_rope(k, cos_selected, sin_selected)

    # Build attention mask
    attn_mask = build_dpe_attention_mask(seq_len, chunk_size, num_global_tokens)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    attn_mask = attn_mask.to(q.device)

    return q_rot, k_rot, attn_mask


class DPERoPE(nn.Module):
    """Dynamic Position Encoding RoPE as a reusable nn.Module.

    Applies chunked position encoding with local/global attention masking
    for training-free extrapolation to ultra-long sequences.

    Args:
        dim: Head dimension (must be even).
        chunk_size: Size of each local chunk (default 2048).
        num_global_tokens: Number of leading global tokens (default 4).
        base: RoPE base frequency.
        shift: Position shift for chunked positions.

    Example:
        dpe_rope = DPERoPE(dim=64, chunk_size=2048, num_global_tokens=4)
        q_rot, k_rot, mask = dpe_rope(q, k)
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))
        scores = scores / math.sqrt(dim) + mask
        attn = torch.softmax(scores, dim=-1)
    """

    def __init__(
        self,
        dim: int,
        chunk_size: int = 2048,
        num_global_tokens: int = 4,
        base: float = 10000.0,
        shift: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.num_global_tokens = num_global_tokens
        self.base = base
        self.shift = shift

        # Precompute cos/sin for chunk_size + shift + num_global_tokens positions
        # Position IDs can be up to (chunk_size - 1 + shift) for non-global tokens
        # and up to (num_global_tokens - 1) for global tokens.
        max_pos = max(chunk_size + shift, num_global_tokens)
        cos, sin = _precompute_rope_cos_sin_half(dim, max_pos, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        global_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply DPE RoPE.

        If global_positions is provided, it overrides the chunked position
        assignment for fine-grained control.
        """
        batch, seq_len = q.shape[:2]

        if global_positions is not None:
            position_ids = global_positions.to(q.device)
        else:
            position_ids = compute_dpe_position_ids(
                seq_len, self.chunk_size, self.num_global_tokens, self.shift
            ).to(q.device)

        # Gather cos/sin for computed position IDs
        cos_sel = self.cos[position_ids].unsqueeze(0).unsqueeze(2)
        sin_sel = self.sin[position_ids].unsqueeze(0).unsqueeze(2)

        q_rot = apply_rope(q, cos_sel, sin_sel)
        k_rot = apply_rope(k, cos_sel, sin_sel)

        # Build attention mask
        attn_mask = build_dpe_attention_mask(
            seq_len, self.chunk_size, self.num_global_tokens
        )
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).to(q.device)

        return q_rot, k_rot, attn_mask

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, chunk_size={self.chunk_size}, "
            f"num_global_tokens={self.num_global_tokens}, "
            f"base={self.base}, shift={self.shift}"
        )
