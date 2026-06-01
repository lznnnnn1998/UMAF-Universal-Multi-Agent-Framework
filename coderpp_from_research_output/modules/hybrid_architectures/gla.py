"""Gated Linear Attention (GLA) with chunkwise parallel forms.

Reference: Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024).
"Gated Linear Attention Transformers with Hardware-Efficient Training."
arXiv:2312.06635.

GLA extends standard linear attention with a data-dependent **forget gate**
that controls how much past information is retained at each position.
Like a gated RNN but parallelisable through chunkwise computation and
associative scans.

Key formulas (recurrent)
------------------------

::

    S_t   = G_t ⊙ S_{t−1} + K_t^T ⊗ V_t     (state update)
    N_t   = G_t ⊙ N_{t−1} + K_t^T            (normaliser)
    O_t   = Q_t ⊗ S_t / (Q_t ⊗ N_t)          (output)

where *G_t = sigmoid(γ_t)* ∈ (0, 1) is the forget gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Standalone functional API
# ---------------------------------------------------------------------------

def gla_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    initial_normalizer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gated linear attention — purely recurrent (O(L) sequential).

    Suitable for autoregressive inference or as a correctness reference
    for the chunkwise form.

    Args:
        q: ``(B, L, d_k)`` queries.
        k: ``(B, L, d_k)`` keys.
        v: ``(B, L, d_v)`` values.
        gate: ``(B, L, d_k)`` forget gate values in (0, 1).
        initial_state: ``(B, d_k, d_v)`` initial KV state.
        initial_normalizer: ``(B, d_k)`` initial normaliser.

    Returns:
        ``(output, final_state, final_normalizer)`` where
        *output* is ``(B, L, d_v)``.
    """
    B, L, d_k = q.shape
    d_v = v.shape[-1]
    device = q.device

    if initial_state is None:
        S = torch.zeros(B, d_k, d_v, device=device, dtype=q.dtype)
    else:
        S = initial_state

    if initial_normalizer is None:
        N = torch.zeros(B, d_k, device=device, dtype=q.dtype)
    else:
        N = initial_normalizer

    outputs: list[torch.Tensor] = []
    for t in range(L):
        g_t = gate[:, t, :]                                           # (B, d_k)
        k_t = k[:, t, :]                                              # (B, d_k)
        v_t = v[:, t, :]                                              # (B, d_v)
        q_t = q[:, t, :]                                              # (B, d_k)

        # S_t = G_t ⊙ S_{t−1} + K_t^T ⊗ V_t
        S = g_t.unsqueeze(-1) * S + torch.einsum("bk,bv->bkv", k_t, v_t)

        # N_t = G_t ⊙ N_{t−1} + K_t^T
        N = g_t * N + k_t

        # O_t = Q_t @ S_t / (Q_t @ N_t)
        num = torch.einsum("bk,bkv->bv", q_t, S)                     # (B, d_v)
        den = (q_t * N).sum(dim=-1, keepdim=True) + 1e-8             # (B, 1)
        o_t = num / den
        outputs.append(o_t)

    out = torch.stack(outputs, dim=1)                                 # (B, L, d_v)
    return out, S, N


def gla_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    initial_normalizer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gated linear attention — chunkwise parallel form.

    Splits the sequence into chunks of size *C*.  Within each chunk the
    computation is fully parallel (cumulative-product + matmul); across
    chunks we pass a recurrent state ``(S, N)``.

    This matches the *hardware-efficient tiling* strategy from the paper:
    intra-chunk parallelism with inter-chunk recurrence.

    Args:
        q: ``(B, L, d_k)`` queries.
        k: ``(B, L, d_k)`` keys.
        v: ``(B, L, d_v)`` values.
        gate: ``(B, L, d_k)`` forget gate values in (0, 1).
        chunk_size: Number of tokens per chunk (default 64).
        initial_state: ``(B, d_k, d_v)`` initial KV state.
        initial_normalizer: ``(B, d_k)`` initial normaliser.

    Returns:
        ``(output, final_state, final_normalizer)``.
    """
    B, L, d_k = q.shape
    d_v = v.shape[-1]
    device = q.device

    # Save originals before padding — needed to recompute correct final
    # states when padding occurs (zero-padded positions contaminate the
    # final S, N otherwise).
    _q_orig = q
    _k_orig = k
    _v_orig = v
    _gate_orig = gate
    _init_S = initial_state
    _init_N = initial_normalizer

    # Pad to multiple of chunk_size
    if L % chunk_size != 0:
        pad_len = chunk_size - (L % chunk_size)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        gate = F.pad(gate, (0, 0, 0, pad_len))
        L_padded = L + pad_len
    else:
        L_padded = L
        pad_len = 0

    if initial_state is None:
        S = torch.zeros(B, d_k, d_v, device=device, dtype=q.dtype)
    else:
        S = initial_state

    if initial_normalizer is None:
        N = torch.zeros(B, d_k, device=device, dtype=q.dtype)
    else:
        N = initial_normalizer

    num_chunks = L_padded // chunk_size
    all_outputs: list[torch.Tensor] = []

    for c in range(num_chunks):
        start = c * chunk_size
        end = start + chunk_size

        Q_c = q[:, start:end]       # (B, C, d_k)
        K_c = k[:, start:end]
        V_c = v[:, start:end]
        G_c = gate[:, start:end]

        C = chunk_size

        # --- Cumulative gate product (fp64 for numerical stability) ---
        # cum_gate[t] = prod_{i=0}^{t} G_c[i]   (along time, per channel)
        # Using float64 prevents underflow when gates << 1 over many steps.
        G_c_fp64 = G_c.to(torch.float64)
        cum_gate_fp64 = torch.cumprod(G_c_fp64, dim=1)                 # (B, C, d_k)
        # Clamp to avoid extreme values that cause division instability.
        cum_gate_fp64 = cum_gate_fp64.clamp(min=1e-30)

        # Weighted keys: K̃_t = K_t / cum_gate[t]
        # Reason: the recurrence coefficient of KV_j on S_t is
        #   prod_{i=j+1}^{t} G_i = cum_gate[t] / cum_gate[j]
        # So S_intra[t] = cum_gate[t] * sum_{j=0}^{t} KV_j / cum_gate[j]
        K_tilde_fp64 = K_c.to(torch.float64) / cum_gate_fp64            # (B, C, d_k)
        V_c_fp64 = V_c.to(torch.float64)

        # --- Intra-chunk state via cumulative sum (associative scan) ---
        # S_intra[t] = cum_gate[t] * sum_{j=0}^{t} K̃_j @ v_j^T
        KV_contrib_fp64 = torch.einsum("bck,bcv->bckv", K_tilde_fp64, V_c_fp64)
        KV_cumsum_fp64 = torch.cumsum(KV_contrib_fp64, dim=1)

        S_intra_fp64 = cum_gate_fp64.unsqueeze(-1) * KV_cumsum_fp64

        # Add contribution from previous-chunk state S
        S_fp64 = S.to(torch.float64)
        S_prev_expanded_fp64 = cum_gate_fp64.unsqueeze(-1) * S_fp64.unsqueeze(1)
        S_total_fp64 = S_prev_expanded_fp64 + S_intra_fp64

        # --- Normaliser (same pattern) ---
        N_contrib_fp64 = K_tilde_fp64
        N_cumsum_fp64 = torch.cumsum(N_contrib_fp64, dim=1)
        N_intra_fp64 = cum_gate_fp64 * N_cumsum_fp64
        N_fp64 = N.to(torch.float64)
        N_prev_expanded_fp64 = cum_gate_fp64 * N_fp64.unsqueeze(1)
        N_total_fp64 = N_prev_expanded_fp64 + N_intra_fp64

        # --- Output (in fp64, cast back at end) ---
        # O[t] = Q[t] @ S_total[t] / (Q[t] @ N_total[t])
        Q_c_fp64 = Q_c.to(torch.float64)
        num_fp64 = torch.einsum("bck,bckv->bcv", Q_c_fp64, S_total_fp64)
        den_fp64 = (Q_c_fp64 * N_total_fp64).sum(dim=-1, keepdim=True) + 1e-8
        O_c = (num_fp64 / den_fp64).to(dtype=q.dtype)
        all_outputs.append(O_c)

        # --- Update recurrent state for next chunk (cast back) ---
        S = S_total_fp64[:, -1].to(dtype=q.dtype)                       # (B, d_k, d_v)
        N = N_total_fp64[:, -1].to(dtype=q.dtype)                       # (B, d_k)

    out = torch.cat(all_outputs, dim=1)                                # (B, L_padded, d_v)
    if pad_len > 0:
        out = out[:, :L]                                               # (B, L, d_v)
        # Recompute final states from the recurrent form to avoid
        # contamination from zero-padded positions (which zero out the
        # cumulative gate product and corrupt S, N).
        _, S, N = gla_recurrent(
            _q_orig, _k_orig, _v_orig, _gate_orig,
            initial_state=_init_S,
            initial_normalizer=_init_N,
        )

    return out, S, N


# ---------------------------------------------------------------------------
# GLA Module (multi-head wrapper)
# ---------------------------------------------------------------------------

class GatedLinearAttention(nn.Module):
    """Multi-head Gated Linear Attention module.

    Projects input → Q/K/V/Gate, applies GLA (recurrent or chunkwise),
    then projects back to ``d_model``.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        chunk_size: Chunk size for chunkwise mode (0 = use recurrent).
        dropout: Dropout probability.
        bias: Whether Q/K/V projections include bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Gate projection (outputs gate logits per head-channel)
        self.gate_proj = nn.Linear(d_model, d_model, bias=bias)

        # Swish activation for gating (following the GLA paper)
        self.gate_activation = nn.SiLU()

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "chunkwise",
        initial_state: torch.Tensor | None = None,
        initial_normalizer: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, L, d_model)`` input.
            mode: ``"chunkwise"`` or ``"recurrent"``.
            initial_state: ``(B, n_heads, head_dim, head_dim)`` optional initial state.
            initial_normalizer: ``(B, n_heads, head_dim)`` optional initial normaliser.

        Returns:
            ``(B, L, d_model)`` output, or ``(output, state, normalizer)``
            if *return_state* is true.
        """
        B, L, D = x.shape
        H = self.n_heads

        Q = rearrange(self.q_proj(x), "b l (h d) -> b l h d", h=H)
        K = rearrange(self.k_proj(x), "b l (h d) -> b l h d", h=H)
        V = rearrange(self.v_proj(x), "b l (h d) -> b l h d", h=H)

        # Gate: project → activate → sigmoid for (0,1) range
        gate_logits = rearrange(self.gate_proj(x), "b l (h d) -> b l h d", h=H)
        gate = torch.sigmoid(self.gate_activation(gate_logits) + 1.0)

        # Process each head independently, then concatenate
        head_outputs: list[torch.Tensor] = []
        final_states: list[torch.Tensor] = []
        final_norms: list[torch.Tensor] = []

        for h_idx in range(H):
            q_h = Q[:, :, h_idx, :]                                       # (B, L, d)
            k_h = K[:, :, h_idx, :]
            v_h = V[:, :, h_idx, :]
            g_h = gate[:, :, h_idx, :]

            S0 = initial_state[:, h_idx] if initial_state is not None else None
            N0 = initial_normalizer[:, h_idx] if initial_normalizer is not None else None

            if mode == "recurrent":
                o_h, S_final, N_final = gla_recurrent(q_h, k_h, v_h, g_h, S0, N0)
            elif mode == "chunkwise":
                o_h, S_final, N_final = gla_chunkwise(q_h, k_h, v_h, g_h, self.chunk_size, S0, N0)
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'recurrent' or 'chunkwise'.")

            head_outputs.append(o_h)                                      # (B, L, d)
            final_states.append(S_final)                                  # (B, d, d)
            final_norms.append(N_final)                                   # (B, d)

        out = torch.stack(head_outputs, dim=2)                             # (B, L, H, d)
        out = rearrange(out, "b l h d -> b l (h d)")                       # (B, L, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        final_state = torch.stack(final_states, dim=1)                     # (B, H, d, d)
        final_normalizer = torch.stack(final_norms, dim=1)                 # (B, H, d)

        return out, final_state, final_normalizer
