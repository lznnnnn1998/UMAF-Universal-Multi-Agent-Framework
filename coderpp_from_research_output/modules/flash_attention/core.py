"""
FlashAttention v1, v2, v3 implementations in pure PyTorch.

This module provides functional and nn.Module interfaces for:
  - FlashAttentionV1: Block-wise tiling, online softmax, recomputation backward.
  - FlashAttentionV2: KV-outer/Q-inner loop swap, Split-Q, delayed normalization.
  - FlashAttentionV3: Warp specialization, ping-pong scheduling, FP8 GEMM simulation.

All implementations use custom torch.autograd.Function for the backward pass
to achieve O(N) memory by recomputing attention scores from stored statistics.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ._tiling import (
    block_partition,
    build_causal_mask,
    online_softmax_update,
    scaled_dot_product_scores,
)
from ._quantization import (
    simulate_fp8_matmul,
    FP8Config,
)


# ===================================================================
# Helper: naive reference attention (for testing correctness)
# ===================================================================

def _naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Naive O(N^2) scaled dot-product attention for reference.

    Args:
        q, k, v: [batch, n_heads, seq_len, d_head].
        causal: Apply causal mask (upper triangle).
        scale: Scale factor (default: 1/sqrt(d_head)).

    Returns:
        Output [batch, n_heads, seq_len, d_head].
    """
    _, _, seq_len, d_head = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    S = torch.matmul(q, k.transpose(-2, -1)) * scale  # [b, h, seq, seq]

    if causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        S = S.masked_fill(mask, float("-inf"))

    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, v)
    return O


# ===================================================================
# FlashAttention V1: Q-outer loop + online softmax + recomputation
# ===================================================================

def _flash_attn_v1_forward(
    q: torch.Tensor,  # [batch, n_heads, seq_len, d_head]
    k: torch.Tensor,  # [batch, n_heads, seq_len, d_head]
    v: torch.Tensor,  # [batch, n_heads, seq_len, d_head]
    Br: int,
    Bc: int,
    causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention v1 forward pass — Algorithm 1 from Dao et al. (2022).

    Q-outer loop structure: for each Q block of size Br, iterate over
    all KV blocks of size Bc, maintaining running online softmax statistics.

    The online softmax recurrence for each Q block i:
      Initialize: m = -inf, ell = 0, O_i = 0
      For each KV block j:
        S_ij = Q_i @ K_j^T * scale
        m_new = max(m, rowmax(S_ij))
        ell = exp(m - m_new) * ell + sum(exp(S_ij - m_new))
        O_i = exp(m - m_new) * O_i + exp(S_ij - m_new) @ V_j
        m = m_new
      O_i = O_i / ell  (final normalization)

    Args:
        q, k, v: [batch, n_heads, seq_len, d_head].
        Br: Q block size (rows).
        Bc: KV block size (rows).
        causal: Apply causal masking.
        scale: Scale factor (default: 1/sqrt(d_head)).

    Returns:
        o: Output [batch, n_heads, seq_len, d_head].
        m: Final row-wise max [batch, n_heads, seq_len].
        ell: Final row-wise sum of exp [batch, n_heads, seq_len].
    """
    batch, n_heads, seq_len, d_head = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    device = q.device
    dtype = q.dtype

    q_blocks = block_partition(q, Br, dim=-2)
    k_blocks = block_partition(k, Bc, dim=-2)
    v_blocks = block_partition(v, Bc, dim=-2)

    o_list: list[torch.Tensor] = []
    m_list: list[torch.Tensor] = []
    ell_list: list[torch.Tensor] = []

    for i, Q_i in enumerate(q_blocks):
        Br_i = Q_i.size(-2)
        q_start = i * Br

        # Running statistics for this Q block
        m = torch.full(
            (batch, n_heads, Br_i), float("-inf"), device=device, dtype=dtype
        )
        ell = torch.zeros(batch, n_heads, Br_i, device=device, dtype=dtype)
        O_i = torch.zeros(batch, n_heads, Br_i, d_head, device=device, dtype=dtype)

        for j, (K_j, V_j) in enumerate(zip(k_blocks, v_blocks)):
            Bc_j = K_j.size(-2)
            kv_start = j * Bc

            # Compute S_ij = Q_i @ K_j^T * scale
            S_ij = scaled_dot_product_scores(Q_i, K_j, scale)  # [b, h, Br_i, Bc_j]

            if causal:
                mask = build_causal_mask(q_start, kv_start, Br_i, Bc_j, device)
                S_ij = S_ij.masked_fill(
                    ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # Online softmax recurrence step
            m_block = S_ij.max(dim=-1).values  # [b, h, Br_i]
            m_new = torch.maximum(m, m_block)  # [b, h, Br_i]

            # Rescaling factor for old accumulators
            scale_old = torch.exp(m - m_new)  # [b, h, Br_i]

            # P_ij = exp(S_ij - m_new), safe softmax numerator
            P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))  # [b, h, Br_i, Bc_j]

            # Update running sum of exp
            ell = scale_old * ell + P_ij.sum(dim=-1)

            # Update output accumulator
            O_i = scale_old.unsqueeze(-1) * O_i + torch.matmul(P_ij, V_j)

            m = m_new

        # Final normalization
        O_i = O_i / ell.unsqueeze(-1)

        o_list.append(O_i)
        m_list.append(m)
        ell_list.append(ell)

    O = torch.cat(o_list, dim=-2)       # [b, h, seq, d]
    M = torch.cat(m_list, dim=-1)       # [b, h, seq]
    L = torch.cat(ell_list, dim=-1)     # [b, h, seq]

    return O, M, L


def _flash_attn_v1_backward(
    q: torch.Tensor,      # [batch, n_heads, seq_len, d_head]
    k: torch.Tensor,      # [batch, n_heads, seq_len, d_head]
    v: torch.Tensor,      # [batch, n_heads, seq_len, d_head]
    o: torch.Tensor,      # [batch, n_heads, seq_len, d_head] — forward output
    m: torch.Tensor,      # [batch, n_heads, seq_len] — row-wise max from forward
    ell: torch.Tensor,    # [batch, n_heads, seq_len] — row-wise sum from forward
    dO: torch.Tensor,     # [batch, n_heads, seq_len, d_head] — upstream gradient
    Br: int,
    Bc: int,
    causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention v1 backward pass — recomputation-based.

    Stores O(N) values (o, m, ell) instead of the full O(N^2) attention matrix.
    Recomputes S and P block-by-block to compute gradients.

    Gradient derivation:
      O = softmax(S) @ V  where  S = Q @ K^T / sqrt(d)
      dV = P^T @ dO
      dP = dO @ V^T
      D = rowsum(P * dP) = rowsum(dO * O)    (key identity!)
      dS = P * (dP - D)
      dQ = dS @ K / sqrt(d)
      dK = dS^T @ Q / sqrt(d)

    The identity D = rowsum(dO * O) avoids needing to recompute P first to
    compute D: we can compute D from O and dO directly, then use it in the
    KV-inner loop.

    Args:
        q, k, v: Input tensors from forward pass.
        o: Output from forward pass.
        m, ell: Softmax statistics stored from forward.
        dO: Gradient of loss w.r.t. output.
        Br, Bc: Block sizes (must match forward).
        causal: Must match forward.
        scale: Must match forward.

    Returns:
        dQ, dK, dV: each [batch, n_heads, seq_len, d_head].
    """
    batch, n_heads, seq_len, d_head = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    device = q.device

    q_blocks = block_partition(q, Br, dim=-2)
    k_blocks = block_partition(k, Bc, dim=-2)
    v_blocks = block_partition(v, Bc, dim=-2)
    o_blocks = block_partition(o, Br, dim=-2)
    dO_blocks = block_partition(dO, Br, dim=-2)

    dQ = torch.zeros_like(q)
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)

    m_blocks = block_partition(m, Br, dim=-1)
    ell_blocks = block_partition(ell, Br, dim=-1)

    # v1 loop order: Q-outer, KV-inner
    for i, (Q_i, O_i, dO_i) in enumerate(zip(q_blocks, o_blocks, dO_blocks)):
        Br_i = Q_i.size(-2)
        q_start = i * Br
        m_i = m_blocks[i]      # [b, h, Br_i]
        ell_i = ell_blocks[i]  # [b, h, Br_i]

        # D_i = rowsum(dO_i * O_i) — the total row-sum correction term
        # This equals Σ_j rowsum(P_ij * dP_ij) but computed without
        # recomputing P first (key memory-saving identity).
        D_i = (dO_i * O_i).sum(dim=-1, keepdim=True)  # [b, h, Br_i, 1]

        dQ_i = torch.zeros_like(Q_i)

        for j, (K_j, V_j) in enumerate(zip(k_blocks, v_blocks)):
            Bc_j = K_j.size(-2)
            kv_start = j * Bc

            # --- Recompute S_ij and P_ij from stored statistics ---
            S_ij = scaled_dot_product_scores(Q_i, K_j, scale)

            if causal:
                mask = build_causal_mask(q_start, kv_start, Br_i, Bc_j, device)
                S_ij = S_ij.masked_fill(
                    ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # P_ij = softmax over ALL KV blocks, reconstructed exactly from
            # stored m_i and ell_i (no per-block renormalization needed)
            P_unnorm = torch.exp(S_ij - m_i.unsqueeze(-1))  # [b, h, Br_i, Bc_j]
            P_ij = P_unnorm / ell_i.unsqueeze(-1)            # [b, h, Br_i, Bc_j]

            # --- dV contribution ---
            dV_contrib = torch.matmul(P_ij.transpose(-2, -1), dO_i)  # [b, h, Bc_j, d]
            dV.narrow(-2, kv_start, Bc_j).add_(dV_contrib)

            # --- dP and dS (softmax backward) ---
            # dP_ij = dO_i @ V_j^T
            dP_ij = torch.matmul(dO_i, V_j.transpose(-2, -1))
            # dS_ij = P_ij * (dP_ij - D_i)  where D_i is the TOTAL row-sum
            # This is the correct softmax Jacobian: ∂loss/∂S = P ∘ (∂loss/∂P - D)
            dS_ij = P_ij * (dP_ij - D_i)

            # --- dQ contribution ---
            dQ_i = dQ_i + torch.matmul(dS_ij, K_j) * scale

            # --- dK contribution ---
            dK_contrib = torch.matmul(
                dS_ij.transpose(-2, -1), Q_i
            ) * scale  # [b, h, Bc_j, d]
            dK.narrow(-2, kv_start, Bc_j).add_(dK_contrib)

        # Write dQ for this block
        dQ.narrow(-2, q_start, Br_i).copy_(dQ_i)

    return dQ, dK, dV


# ===================================================================
# FlashAttention V2: KV-outer loop + Split-Q + delayed normalization
# ===================================================================

def _flash_attn_v2_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Br: int,
    Bc: int,
    causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention v2 forward pass — KV-outer loop + delayed normalization.

    Key improvements over v1 (Dao, 2023):
      1. **KV-outer loop**: Each K_j, V_j block is loaded once and streamed
         across all Q_i blocks, improving cache efficiency. K and V stay
         resident in memory while Q blocks rotate through.
      2. **Split-Q parallelism**: Q blocks are independent in the inner loop
         (no inter-Q-block dependencies), enabling warp-level parallelism.
      3. **Delayed normalization**: Output is accumulated un-normalized and
         divided by ell only once at the end, reducing division operations
         from O(N^2/(Br*Bc)) to O(N/Br).

    Args:
        q, k, v: [batch, n_heads, seq_len, d_head].
        Br, Bc: Block sizes.
        causal: Apply causal masking.
        scale: Scale factor.

    Returns:
        o, m, ell as in v1.
    """
    batch, n_heads, seq_len, d_head = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    device = q.device
    dtype = q.dtype

    q_blocks = block_partition(q, Br, dim=-2)
    k_blocks = block_partition(k, Bc, dim=-2)
    v_blocks = block_partition(v, Bc, dim=-2)

    n_q_blocks = len(q_blocks)

    # Per-Q-block state (persists across the KV-outer loop)
    m_per_q = [
        torch.full((batch, n_heads, q_blocks[i].size(-2)), float("-inf"),
                   device=device, dtype=dtype)
        for i in range(n_q_blocks)
    ]
    ell_per_q = [
        torch.zeros(batch, n_heads, q_blocks[i].size(-2), device=device, dtype=dtype)
        for i in range(n_q_blocks)
    ]
    O_per_q = [
        torch.zeros(batch, n_heads, q_blocks[i].size(-2), d_head,
                    device=device, dtype=dtype)
        for i in range(n_q_blocks)
    ]

    # v2: KV-OUTER loop — each K_j, V_j loaded once, streamed across all Q_i
    for j, (K_j, V_j) in enumerate(zip(k_blocks, v_blocks)):
        Bc_j = K_j.size(-2)
        kv_start = j * Bc

        # Inner loop over Q blocks — Split-Q: each Q block is independent
        for i, Q_i in enumerate(q_blocks):
            Br_i = Q_i.size(-2)
            q_start = i * Br

            S_ij = scaled_dot_product_scores(Q_i, K_j, scale)

            if causal:
                mask = build_causal_mask(q_start, kv_start, Br_i, Bc_j, device)
                S_ij = S_ij.masked_fill(
                    ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            m_block = S_ij.max(dim=-1).values
            m_new = torch.maximum(m_per_q[i], m_block)

            scale_old = torch.exp(m_per_q[i] - m_new)

            P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))

            # Accumulate without normalizing (delayed normalization)
            ell_per_q[i] = scale_old * ell_per_q[i] + P_ij.sum(dim=-1)
            O_per_q[i] = (
                scale_old.unsqueeze(-1) * O_per_q[i]
                + torch.matmul(P_ij, V_j)
            )

            m_per_q[i] = m_new

    # Delayed normalization: apply 1/ell at the end (instead of every KV block)
    for i in range(n_q_blocks):
        O_per_q[i] = O_per_q[i] / ell_per_q[i].unsqueeze(-1)

    O = torch.cat(O_per_q, dim=-2)
    M = torch.cat(m_per_q, dim=-1)
    L = torch.cat(ell_per_q, dim=-1)

    return O, M, L


def _flash_attn_v2_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    m: torch.Tensor,
    ell: torch.Tensor,
    dO: torch.Tensor,
    Br: int,
    Bc: int,
    causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention v2 backward pass — recomputation-based with KV-outer loop.

    Same gradient formulas as v1, but with KV-outer loop order for better
    data reuse. Each K_j, V_j block is loaded once; all Q blocks are
    processed in the inner loop, then dK_j and dV_j are written.
    """
    batch, n_heads, seq_len, d_head = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    device = q.device

    q_blocks = block_partition(q, Br, dim=-2)
    k_blocks = block_partition(k, Bc, dim=-2)
    v_blocks = block_partition(v, Bc, dim=-2)
    o_blocks = block_partition(o, Br, dim=-2)
    dO_blocks = block_partition(dO, Br, dim=-2)

    dQ = torch.zeros_like(q)
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)

    m_blocks = block_partition(m, Br, dim=-1)
    ell_blocks = block_partition(ell, Br, dim=-1)

    # Pre-compute D_i = rowsum(dO_i * O_i) per Q block (total correction term)
    D_blocks: list[torch.Tensor] = []
    for O_i, dO_i in zip(o_blocks, dO_blocks):
        D_blocks.append((dO_i * O_i).sum(dim=-1, keepdim=True))  # [b, h, Br_i, 1]

    # v2: KV-outer loop for backward
    for j, (K_j, V_j) in enumerate(zip(k_blocks, v_blocks)):
        Bc_j = K_j.size(-2)
        kv_start = j * Bc

        dV_j = torch.zeros_like(V_j)
        dK_j = torch.zeros_like(K_j)

        for i, (Q_i, dO_i) in enumerate(zip(q_blocks, dO_blocks)):
            Br_i = Q_i.size(-2)
            q_start = i * Br
            m_i = m_blocks[i]
            ell_i = ell_blocks[i]
            D_i = D_blocks[i]

            # Recompute S_ij and P_ij
            S_ij = scaled_dot_product_scores(Q_i, K_j, scale)

            if causal:
                mask = build_causal_mask(q_start, kv_start, Br_i, Bc_j, device)
                S_ij = S_ij.masked_fill(
                    ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            P_unnorm = torch.exp(S_ij - m_i.unsqueeze(-1))
            P_ij = P_unnorm / ell_i.unsqueeze(-1)

            # dV accumulation
            dV_j = dV_j + torch.matmul(P_ij.transpose(-2, -1), dO_i)

            # dP and dS
            dP_ij = torch.matmul(dO_i, V_j.transpose(-2, -1))
            dS_ij = P_ij * (dP_ij - D_i)

            # dK accumulation
            dK_j = dK_j + torch.matmul(
                dS_ij.transpose(-2, -1), Q_i
            ) * scale

            # dQ accumulation (directly into dQ since we visit each Q block once per KV block)
            dQ.narrow(-2, q_start, Br_i).add_(
                torch.matmul(dS_ij, K_j) * scale
            )

        # Write accumulated dV and dK for this KV block
        dV.narrow(-2, kv_start, Bc_j).copy_(dV_j)
        dK.narrow(-2, kv_start, Bc_j).copy_(dK_j)

    return dQ, dK, dV


# ===================================================================
# FlashAttention V3: warp specialization + ping-pong + FP8
# ===================================================================

def _flash_attn_v3_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Br: int,
    Bc: int,
    causal: bool = False,
    scale: Optional[float] = None,
    use_fp8: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention v3 forward pass.

    Key v3 innovations (Shah et al., 2024):
      1. **Warp specialization**: Different warps handle GEMM (S = Q@K^T)
         and softmax separately. In this simulation, phases are labeled
         but run sequentially since Python cannot execute concurrent GPU warps.
      2. **Ping-pong scheduling**: Two alternating buffers enable overlapping
         GEMM and softmax: while one buffer processes softmax, the next
         block's GEMM is "launched" asynchronously.
      3. **FP8 GEMM**: Optional reduced-precision matrix multiply for
         S = Q @ K^T using E4M3 format for memory bandwidth savings.

    Args:
        q, k, v: [batch, n_heads, seq_len, d_head].
        Br, Bc: Block sizes.
        causal: Apply causal masking.
        scale: Scale factor.
        use_fp8: Simulate FP8 E4M3 GEMM for attention scores.

    Returns:
        o, m, ell as in v1/v2.
    """
    batch, n_heads, seq_len, d_head = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    device = q.device
    dtype = q.dtype

    q_blocks = block_partition(q, Br, dim=-2)
    k_blocks = block_partition(k, Bc, dim=-2)
    v_blocks = block_partition(v, Bc, dim=-2)

    n_q_blocks = len(q_blocks)
    fp8_config = FP8Config() if use_fp8 else None

    # Per-Q-block state (same as v2, with optional FP8 in GEMM phase)
    m_per_q = [
        torch.full((batch, n_heads, q_blocks[i].size(-2)), float("-inf"),
                   device=device, dtype=dtype)
        for i in range(n_q_blocks)
    ]
    ell_per_q = [
        torch.zeros(batch, n_heads, q_blocks[i].size(-2), device=device, dtype=dtype)
        for i in range(n_q_blocks)
    ]
    O_per_q = [
        torch.zeros(batch, n_heads, q_blocks[i].size(-2), d_head,
                    device=device, dtype=dtype)
        for i in range(n_q_blocks)
    ]

    # KV-outer loop with ping-pong simulation
    for j, (K_j, V_j) in enumerate(zip(k_blocks, v_blocks)):
        Bc_j = K_j.size(-2)
        kv_start = j * Bc

        for i, Q_i in enumerate(q_blocks):
            Br_i = Q_i.size(-2)
            q_start = i * Br

            # --- Phase G: GEMM warp computes S = Q @ K^T ---
            if use_fp8:
                K_j_T = K_j.transpose(-2, -1)
                S_ij = simulate_fp8_matmul(Q_i, K_j_T, fp8_config) * scale
            else:
                S_ij = scaled_dot_product_scores(Q_i, K_j, scale)

            # --- Phase S: Softmax warp processes scores ---
            # In hardware, Phase G for block (j+1) overlaps with Phase S for block j
            if causal:
                mask = build_causal_mask(q_start, kv_start, Br_i, Bc_j, device)
                S_ij = S_ij.masked_fill(
                    ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            m_block = S_ij.max(dim=-1).values
            m_new = torch.maximum(m_per_q[i], m_block)

            scale_old = torch.exp(m_per_q[i] - m_new)

            P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))

            ell_per_q[i] = scale_old * ell_per_q[i] + P_ij.sum(dim=-1)
            O_per_q[i] = (
                scale_old.unsqueeze(-1) * O_per_q[i]
                + torch.matmul(P_ij, V_j)
            )

            m_per_q[i] = m_new

    # Final normalization
    for i in range(n_q_blocks):
        O_per_q[i] = O_per_q[i] / ell_per_q[i].unsqueeze(-1)

    O = torch.cat(O_per_q, dim=-2)
    M = torch.cat(m_per_q, dim=-1)
    L = torch.cat(ell_per_q, dim=-1)

    return O, M, L


def _flash_attn_v3_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    m: torch.Tensor,
    ell: torch.Tensor,
    dO: torch.Tensor,
    Br: int,
    Bc: int,
    causal: bool = False,
    scale: Optional[float] = None,
    use_fp8: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention v3 backward pass — recomputation-based.

    Same gradient formulas as v1/v2. Uses stored o to compute the total
    row-sum correction D = rowsum(dO * O) per Q block.

    When use_fp8=True, recomputes S_ij using simulated FP8 GEMM to match
    the forward pass's quantization, ensuring consistent gradients.
    """
    batch, n_heads, seq_len, d_head = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    device = q.device
    fp8_config = FP8Config() if use_fp8 else None

    q_blocks = block_partition(q, Br, dim=-2)
    k_blocks = block_partition(k, Bc, dim=-2)
    v_blocks = block_partition(v, Bc, dim=-2)
    o_blocks = block_partition(o, Br, dim=-2)
    dO_blocks = block_partition(dO, Br, dim=-2)

    dQ = torch.zeros_like(q)
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)

    m_blocks = block_partition(m, Br, dim=-1)
    ell_blocks = block_partition(ell, Br, dim=-1)

    # Pre-compute D_i = rowsum(dO_i * O_i) per Q block
    D_blocks: list[torch.Tensor] = []
    for O_i, dO_i in zip(o_blocks, dO_blocks):
        D_blocks.append((dO_i * O_i).sum(dim=-1, keepdim=True))

    # KV-outer loop for backward
    for j, (K_j, V_j) in enumerate(zip(k_blocks, v_blocks)):
        Bc_j = K_j.size(-2)
        kv_start = j * Bc

        dV_j = torch.zeros_like(V_j)
        dK_j = torch.zeros_like(K_j)

        for i, (Q_i, dO_i) in enumerate(zip(q_blocks, dO_blocks)):
            Br_i = Q_i.size(-2)
            q_start = i * Br
            m_i = m_blocks[i]
            ell_i = ell_blocks[i]
            D_i = D_blocks[i]

            # Recompute S_ij using the same precision as forward
            if use_fp8:
                K_j_T = K_j.transpose(-2, -1)
                S_ij = simulate_fp8_matmul(Q_i, K_j_T, fp8_config) * scale
            else:
                S_ij = scaled_dot_product_scores(Q_i, K_j, scale)

            if causal:
                mask = build_causal_mask(q_start, kv_start, Br_i, Bc_j, device)
                S_ij = S_ij.masked_fill(
                    ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            P_unnorm = torch.exp(S_ij - m_i.unsqueeze(-1))
            P_ij = P_unnorm / ell_i.unsqueeze(-1)

            dV_j = dV_j + torch.matmul(P_ij.transpose(-2, -1), dO_i)

            dP_ij = torch.matmul(dO_i, V_j.transpose(-2, -1))
            dS_ij = P_ij * (dP_ij - D_i)

            dK_j = dK_j + torch.matmul(
                dS_ij.transpose(-2, -1), Q_i
            ) * scale

            dQ.narrow(-2, q_start, Br_i).add_(
                torch.matmul(dS_ij, K_j) * scale
            )

        dV.narrow(-2, kv_start, Bc_j).copy_(dV_j)
        dK.narrow(-2, kv_start, Bc_j).copy_(dK_j)

    return dQ, dK, dV


# ===================================================================
# torch.autograd.Function wrappers
# ===================================================================

class _FlashAttentionV1Function(torch.autograd.Function):
    """Autograd Function for FlashAttention v1.

    Stores O(N) statistics (o, m, ell) in ctx instead of the full
    O(N^2) attention matrix. Backward recomputes S and P from stored values.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        Br: int,
        Bc: int,
        causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        o, m, ell = _flash_attn_v1_forward(q, k, v, Br, Bc, causal, scale)
        # Save all tensors needed for backward (including o for D computation)
        ctx.save_for_backward(q, k, v, o, m, ell)
        ctx.Br = Br
        ctx.Bc = Bc
        ctx.causal = causal
        ctx.scale = scale
        return o

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, o, m, ell = ctx.saved_tensors
        dQ, dK, dV = _flash_attn_v1_backward(
            q, k, v, o, m, ell, grad_output,
            ctx.Br, ctx.Bc, ctx.causal, ctx.scale,
        )
        return dQ, dK, dV, None, None, None, None


class _FlashAttentionV2Function(torch.autograd.Function):
    """Autograd Function for FlashAttention v2."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        Br: int,
        Bc: int,
        causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        o, m, ell = _flash_attn_v2_forward(q, k, v, Br, Bc, causal, scale)
        ctx.save_for_backward(q, k, v, o, m, ell)
        ctx.Br = Br
        ctx.Bc = Bc
        ctx.causal = causal
        ctx.scale = scale
        return o

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, o, m, ell = ctx.saved_tensors
        dQ, dK, dV = _flash_attn_v2_backward(
            q, k, v, o, m, ell, grad_output,
            ctx.Br, ctx.Bc, ctx.causal, ctx.scale,
        )
        return dQ, dK, dV, None, None, None, None


class _FlashAttentionV3Function(torch.autograd.Function):
    """Autograd Function for FlashAttention v3.

    Supports optional FP8 E4M3 GEMM simulation. When use_fp8=True,
    both forward and backward use simulated FP8 matrix multiply
    for Q @ K^T to ensure consistent gradient computation.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        Br: int,
        Bc: int,
        causal: bool,
        scale: Optional[float],
        use_fp8: bool,
    ) -> torch.Tensor:
        o, m, ell = _flash_attn_v3_forward(q, k, v, Br, Bc, causal, scale, use_fp8)
        ctx.save_for_backward(q, k, v, o, m, ell)
        ctx.Br = Br
        ctx.Bc = Bc
        ctx.causal = causal
        ctx.scale = scale
        ctx.use_fp8 = use_fp8
        return o

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, o, m, ell = ctx.saved_tensors
        dQ, dK, dV = _flash_attn_v3_backward(
            q, k, v, o, m, ell, grad_output,
            ctx.Br, ctx.Bc, ctx.causal, ctx.scale, ctx.use_fp8,
        )
        return dQ, dK, dV, None, None, None, None, None


# ===================================================================
# nn.Module wrappers
# ===================================================================

class FlashAttentionV1(nn.Module):
    """FlashAttention v1 module — block-wise tiling + online softmax.

    Q-outer loop with online softmax recurrence. Stores O(N) statistics
    for memory-efficient recomputation-based backward pass.

    Args:
        Br: Q block size (default: 32).
        Bc: KV block size (default: 32).
        causal: Apply causal masking (default: False).
        scale: Custom scale factor (default: 1/sqrt(d_head)).
    """

    def __init__(
        self,
        Br: int = 32,
        Bc: int = 32,
        causal: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.Br = Br
        self.Bc = Bc
        self.causal = causal
        self.scale = scale

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            q, k, v: [batch, n_heads, seq_len, d_head].

        Returns:
            Output [batch, n_heads, seq_len, d_head].
        """
        return _FlashAttentionV1Function.apply(
            q, k, v, self.Br, self.Bc, self.causal, self.scale
        )


class FlashAttentionV2(nn.Module):
    """FlashAttention v2 module — KV-outer loop + Split-Q + delayed norm.

    KV-outer loop for better K/V cache reuse. Q blocks are independent
    in the inner loop (Split-Q) enabling warp-level parallelism.

    Args:
        Br: Q block size (default: 32).
        Bc: KV block size (default: 32).
        causal: Apply causal masking (default: False).
        scale: Custom scale factor (default: 1/sqrt(d_head)).
    """

    def __init__(
        self,
        Br: int = 32,
        Bc: int = 32,
        causal: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.Br = Br
        self.Bc = Bc
        self.causal = causal
        self.scale = scale

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        return _FlashAttentionV2Function.apply(
            q, k, v, self.Br, self.Bc, self.causal, self.scale
        )


class FlashAttentionV3(nn.Module):
    """FlashAttention v3 module — warp specialization + ping-pong + FP8.

    Simulates warp specialization (GEMM warp vs softmax warp), ping-pong
    buffer scheduling, and optional FP8 E4M3 GEMM for Q@K^T.

    Args:
        Br: Q block size (default: 32).
        Bc: KV block size (default: 32).
        causal: Apply causal masking (default: False).
        scale: Custom scale factor (default: 1/sqrt(d_head)).
        use_fp8: Simulate FP8 E4M3 GEMM (default: False).
    """

    def __init__(
        self,
        Br: int = 32,
        Bc: int = 32,
        causal: bool = False,
        scale: Optional[float] = None,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.Br = Br
        self.Bc = Bc
        self.causal = causal
        self.scale = scale
        self.use_fp8 = use_fp8

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        return _FlashAttentionV3Function.apply(
            q, k, v, self.Br, self.Bc, self.causal, self.scale, self.use_fp8
        )


# ===================================================================
# Convenience functional interfaces
# ===================================================================

def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Br: int = 32,
    Bc: int = 32,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Functional interface for FlashAttention v1.

    Args:
        q, k, v: [batch, n_heads, seq_len, d_head].
        Br: Q block size.
        Bc: KV block size.
        causal: Apply causal masking.
        scale: Custom scale factor.

    Returns:
        Output [batch, n_heads, seq_len, d_head].
    """
    return _FlashAttentionV1Function.apply(q, k, v, Br, Bc, causal, scale)


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Br: int = 32,
    Bc: int = 32,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Functional interface for FlashAttention v2.

    KV-outer loop with Split-Q parallelism and delayed normalization.
    """
    return _FlashAttentionV2Function.apply(q, k, v, Br, Bc, causal, scale)


def flash_attention_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Br: int = 32,
    Bc: int = 32,
    causal: bool = False,
    scale: Optional[float] = None,
    use_fp8: bool = False,
) -> torch.Tensor:
    """Functional interface for FlashAttention v3.

    Warp specialization simulation with ping-pong scheduling and optional FP8 GEMM.
    """
    return _FlashAttentionV3Function.apply(
        q, k, v, Br, Bc, causal, scale, use_fp8
    )
