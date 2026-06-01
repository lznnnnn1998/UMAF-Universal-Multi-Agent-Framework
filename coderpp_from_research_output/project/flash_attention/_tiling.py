"""
Block tiling and online softmax utilities for FlashAttention.

Provides:
  - block_partition: split tensors into SRAM-sized blocks
  - online_softmax_update: exact running statistics recurrence for online softmax
  - build_causal_mask: generate causal attention masks for block pairs
"""

from __future__ import annotations

import math
from typing import List

import torch


def block_partition(
    x: torch.Tensor, block_size: int, dim: int = -2
) -> List[torch.Tensor]:
    """Split a tensor into blocks along the specified dimension.

    This simulates SRAM partitioning: each returned block represents data
    that fits in on-chip SRAM (Br x d for Q, Bc x d for K/V).

    Args:
        x: Input tensor of any shape.
        block_size: Maximum size of each block along `dim`.
        dim: Dimension to partition along (default: -2, the sequence dim).

    Returns:
        List of tensor blocks. The last block may be smaller than block_size.

    Example:
        >>> x = torch.randn(2, 4, 100, 64)  # [batch, heads, seq, d]
        >>> blocks = block_partition(x, 32, dim=-2)
        >>> len(blocks)  # ceil(100/32) = 4
        4
        >>> blocks[0].shape  # first 3 blocks: [2, 4, 32, 64]
        torch.Size([2, 4, 32, 64])
        >>> blocks[3].shape  # last block: [2, 4, 4, 64]
        torch.Size([2, 4, 4, 64])
    """
    seq_len = x.size(dim)
    blocks: List[torch.Tensor] = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        # Use narrow for zero-copy slicing along the target dimension
        length = end - start
        blocks.append(x.narrow(dim, start, length))
    return blocks


def online_softmax_update(
    m_old: torch.Tensor,  # [..., Br]
    l_old: torch.Tensor,  # [..., Br]
    s_block: torch.Tensor,  # [..., Br, Bc]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform one step of the online softmax recurrence.

    This is the core numerical primitive of FlashAttention's forward pass.
    Given previous running max m_old and running sum l_old, and a new block
    of attention scores s_block, compute the updated statistics and the
    contribution of this block to the output.

    The recurrence (from the FlashAttention paper, Algorithm 1) is:
      m_new = max(m_old, rowmax(s_block))
      l_new = l_old * exp(m_old - m_new) + rowsum(exp(s_block - m_new))
      P_block = exp(s_block - m_new)   (partial softmax numerator)

    Args:
        m_old: Previous running max per row, shape [..., Br].
               Initialized to -inf for the first block.
        l_old: Previous running sum per row, shape [..., Br].
               Initialized to 0 for the first block.
        s_block: New attention score block, shape [..., Br, Bc].

    Returns:
        m_new: Updated running max, shape [..., Br].
        l_new: Updated running sum (of exp), shape [..., Br].
        P_block: exp(s_block - m_new), the partial softmax numerator,
                 shape [..., Br, Bc].
    """
    # Row-wise maximum of the new block
    m_block = s_block.max(dim=-1).values  # [..., Br]

    # New running max: element-wise max of old and new
    m_new = torch.maximum(m_old, m_block)  # [..., Br]

    # Rescale old accumulator: exp(m_old - m_new).
    # The exp of -inf gives 0, which is correct for initialization.
    scale_old = torch.exp(m_old - m_new)  # [..., Br]

    # Update running sum using the correct recurrence:
    #   l_new = l_old * exp(m_old - m_new) + sum(exp(s_block - m_new))
    # Note: no scale_new factor on the new block term. P_block = exp(s_block - m_new)
    # is already correctly scaled w.r.t. the new running max m_new.
    l_new = (
        scale_old * l_old
        + torch.exp(s_block - m_new.unsqueeze(-1)).sum(dim=-1)
    )  # [..., Br]

    # Partial softmax numerator for this block
    P_block = torch.exp(s_block - m_new.unsqueeze(-1))  # [..., Br, Bc]

    return m_new, l_new, P_block


def build_causal_mask(
    q_start: int,
    kv_start: int,
    q_len: int,
    kv_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a causal mask for a block pair in FlashAttention.

    For causal attention, query position p can only attend to key positions <= p.
    For Q block starting at q_start and KV block starting at kv_start,
    position (r, c) within the block is valid iff:
        kv_start + c <= q_start + r

    Args:
        q_start: Starting position of the Q block in the full sequence.
        kv_start: Starting position of the KV block in the full sequence.
        q_len: Number of query positions in this block (Br or smaller).
        kv_len: Number of key positions in this block (Bc or smaller).
        device: Torch device for the output tensor.

    Returns:
        Boolean mask of shape [q_len, kv_len], where True means "attend"
        (valid) and False means "mask out".
    """
    # Position indices within the block
    q_pos = torch.arange(q_start, q_start + q_len, device=device)  # [q_len]
    kv_pos = torch.arange(kv_start, kv_start + kv_len, device=device)  # [kv_len]

    # Causal: q_pos >= kv_pos (query can attend to keys at or before it)
    # q_pos[:, None]: [q_len, 1], kv_pos[None, :]: [1, kv_len]
    mask = q_pos.unsqueeze(1) >= kv_pos.unsqueeze(0)  # [q_len, kv_len]

    return mask


def scaled_dot_product_scores(
    q_block: torch.Tensor,  # [..., Br, d]
    k_block: torch.Tensor,  # [..., Bc, d]
    scale: float,
) -> torch.Tensor:
    """Compute scaled dot-product attention scores for a block pair.

    S = Q @ K^T / sqrt(d)

    Args:
        q_block: Query block, shape [..., Br, d].
        k_block: Key block, shape [..., Bc, d].
        scale: Scaling factor (typically 1/sqrt(d_head)).

    Returns:
        Attention scores, shape [..., Br, Bc].
    """
    return torch.matmul(q_block, k_block.transpose(-2, -1)) * scale


def compute_flops_saved(
    seq_len: int,
    d_head: int,
    Br: int = 32,
    Bc: int = 32,
) -> dict[str, float]:
    """Estimate FLOPs saved by FlashAttention vs naive attention.

    Naive attention computes the full N×N attention matrix and applies
    softmax over all elements. FlashAttention avoids materializing the
    full matrix by using block-wise tiling.

    Naive FLOPs (forward + backward):
      - S = Q @ K^T: 2 * B * H * N^2 * d  (2× for multiply-add)
      - P = softmax(S): ~5 * B * H * N^2  (exp, sum, divide per row)
      - O = P @ V: 2 * B * H * N^2 * d
      - dQ, dK, dV: ~ same as forward
      Total naive ≈ 4 * (2 * N^2 * d + 5 * N^2) * B * H

    FlashAttention FLOPs (forward + backward):
      - Same number of matmul FLOPs but better cache utilization
      - No explicit softmax over full N×N matrix
      - Backward recomputes S (extra FLOPs) but saves memory

    Args:
        seq_len: Sequence length N.
        d_head: Head dimension d.
        Br: Q block size.
        Bc: KV block size.

    Returns:
        Dict with 'naive_gflops', 'flash_gflops', 'ratio' fields.
    """
    # Constants
    B, H = 1, 1  # per-head analysis
    N = seq_len
    d = d_head

    # Naive: forward has 2 matmuls + softmax, backward is ~2× forward
    naive_matmul = 2.0 * B * H * N * N * d  # Q@K^T
    naive_softmax = 5.0 * B * H * N * N   # exp + sum + divide
    naive_attn_out = 2.0 * B * H * N * N * d  # P@V
    naive_forward = naive_matmul + naive_softmax + naive_attn_out
    naive_backward = naive_forward * 2.0  # approximate
    naive_total = naive_forward + naive_backward

    # FlashAttention: same matmul ops, but softmax is cheaper (block-wise)
    # Forward recomputation in backward adds ~1× forward FLOPs
    flash_forward = naive_matmul + naive_attn_out + (5.0 * B * H * N * N / max(Br, Bc))
    flash_backward = flash_forward * 1.5  # recomputation overhead
    flash_total = flash_forward + flash_backward

    naive_gflops = naive_total / 1e9
    flash_gflops = flash_total / 1e9
    ratio = naive_total / flash_total if flash_total > 0 else float("inf")

    return {
        "naive_gflops": naive_gflops,
        "flash_gflops": flash_gflops,
        "ratio": ratio,
    }


def compute_memory_saved(
    seq_len: int,
    d_head: int,
    bytes_per_element: int = 2,  # fp16
) -> dict[str, float]:
    """Estimate peak memory savings of FlashAttention vs naive attention.

    Naive attention stores the full N×N attention matrix (P) for backward,
    requiring O(N^2) memory. FlashAttention stores only O(N) values:
    output O (N×d), row-wise max m (N), and row-wise sum l (N).

    Args:
        seq_len: Sequence length N.
        d_head: Head dimension d.
        bytes_per_element: Bytes per element (2 for fp16, 4 for fp32).

    Returns:
        Dict with 'naive_mb', 'flash_mb', 'ratio' fields.
    """
    N = seq_len
    d = d_head
    bpe = bytes_per_element

    # Naive: stores S (N×N) + P (N×N) + O (N×d) + gradients
    naive_elements = N * N + N * N + N * d  # S, P, O
    naive_bytes = naive_elements * bpe

    # FlashAttention: stores O (N×d) + m (N) + l (N) + Q/K/V (3 × N×d)
    flash_elements = N * d + N + N + 3 * N * d
    flash_bytes = flash_elements * bpe

    naive_mb = naive_bytes / (1024 * 1024)
    flash_mb = flash_bytes / (1024 * 1024)
    ratio = naive_bytes / flash_bytes if flash_bytes > 0 else float("inf")

    return {
        "naive_mb": naive_mb,
        "flash_mb": flash_mb,
        "ratio": ratio,
    }
