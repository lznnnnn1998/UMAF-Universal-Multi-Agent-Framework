"""RetNet: Retentive Network — dual-form retention mechanism.

Implements multi-scale retention with three compute modes:

* **Parallel** — O(T²) attention-style, for training with full sequence.
* **Recurrent** — O(T) with constant state, for autoregressive inference.
* **Chunkwise** — Hybrid that splits the sequence into chunks, using
  parallel form within chunks and recurrent state between chunks.
  Provides a practical trade-off between training speed and memory.

The three forms are mathematically equivalent (up to floating-point
precision), so the model can switch between them transparently.

Reference
---------
- Retentive Network: A Successor to Transformer for Large Language Models
  (Sun et al., Microsoft, 2023)  https://arxiv.org/abs/2307.08621
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm


# ---------------------------------------------------------------------------
# Decay matrix helper
# ---------------------------------------------------------------------------

def _build_decay_matrix(gamma: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Build the causal decay mask  D_{ij} = γ^{i-j}  for i ≥ j.

    Args:
        gamma: Scalar or (heads,) decay factor(s), each in (0, 1).
        seq_len: Sequence length.

    Returns:
        (1 or heads, 1, seq_len, seq_len) lower-triangular decay matrix.
    """
    # indices: (seq_len, seq_len)
    idx = torch.arange(seq_len, device=gamma.device)
    i, j = idx.unsqueeze(1), idx.unsqueeze(0)         # (T, 1), (1, T)
    dist = i - j                                        # (T, T)

    # γ^{dist} for i ≥ j, 0 otherwise
    # gamma shape:  (heads,)  or  ()
    g = gamma.view(-1, 1, 1, 1)                        # (H, 1, 1, 1)
    decay = g ** dist.clamp(min=0).float()              # (H, 1, T, T)
    # Return (1, H, T, T) for broadcasting with scores (B, H, T, T)
    decay = decay.squeeze(1)                            # (H, T, T)
    decay = decay.unsqueeze(0)                          # (1, H, T, T)
    mask = (dist >= 0).float()                          # (T, T)
    mask = mask.unsqueeze(0).unsqueeze(0)               # (1, 1, T, T)
    return decay * mask                                 # (1, H, T, T)


# ---------------------------------------------------------------------------
# Retention — parallel form (training)
# ---------------------------------------------------------------------------

def retention_parallel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Parallel (training-mode) retention.

    Computes retention for the full sequence in one go using matrix
    multiplication::

        D_{ij} = γ^{i-j}  for i ≥ j, else 0
        Retention(Q, K, V) = (Q K^T ⊙ D) V

    Args:
        q: (B, H, T, D)  queries.
        k: (B, H, T, D)  keys.
        v: (B, H, T, D)  values.
        gamma: (H,)  per-head decay in (0, 1).

    Returns:
        (B, H, T, D)  retained output.
    """
    B, H, T, D = q.shape
    scale = D ** -0.5

    # QK^T  —  (B, H, T, T)
    scores = torch.einsum("bhtd,bhsd->bhts", q * scale, k)

    # Decay mask:  D_{ij} = γ^{i-j}, shape (1, H, T, T)
    D = _build_decay_matrix(gamma, T).to(q.device).to(q.dtype)  # (1, H, T, T)
    scores = scores * D  # (B, H, T, T) — broadcast over batch dim 0

    # (QK^T ⊙ D) V  —  (B, H, T, D)
    output = torch.einsum("bhts,bhsd->bhtd", scores, v)
    return output


# ---------------------------------------------------------------------------
# Retention — recurrent form (inference)
# ---------------------------------------------------------------------------

def retention_recurrent(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    gamma: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent (inference-mode) retention.

    Updates a state matrix  S ∈ R^{H, D, D}  autoregressively::

        S_t = γ S_{t-1} + K_t^T V_t
        o_t = Q_t S_t

    This is O(1) memory per step and O(T·D²) total, avoiding the O(T²)
    cost of the parallel form.

    Args:
        q: (B, H, T, D)  queries.
        k: (B, H, T, D)  keys.
        v: (B, H, T, D)  values.
        gamma: (H,)  per-head decay.
        state: Optional initial state  (B, H, D, D).  Zeros if None.

    Returns:
        output: (B, H, T, D)
        state:  Final (B, H, D, D)  state matrix.
    """
    B, H, T, D = q.shape
    device = q.device
    dtype = q.dtype
    scale = D ** -0.5

    if state is None:
        S = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
    else:
        S = state.float()

    g = gamma.view(H, 1, 1).float()  # (H, 1, 1)

    outputs: list[torch.Tensor] = []
    k_f32 = k.float() * scale
    v_f32 = v.float()
    q_f32 = q.float()

    for t in range(T):
        # K_t^T V_t  outer product:  (B, H, D, 1) × (B, H, 1, D) → (B, H, D, D)
        kt = k_f32[:, :, t].unsqueeze(-1)    # (B, H, D, 1)
        vt = v_f32[:, :, t].unsqueeze(-2)    # (B, H, 1, D)
        outer = kt @ vt                       # (B, H, D, D)

        # State update:  S_t = γ S_{t-1} + K_t^T V_t
        S = g * S + outer

        # Output:  Q_t S_t
        qt = q_f32[:, :, t].unsqueeze(-2)    # (B, H, 1, D)
        ot = qt @ S                           # (B, H, 1, D)
        outputs.append(ot.squeeze(-2).unsqueeze(2))  # (B, H, 1, D)

    output = torch.cat(outputs, dim=2).to(dtype)  # (B, H, T, D)
    return output, S.to(dtype)


# ---------------------------------------------------------------------------
# Retention — chunkwise form (efficient training)
# ---------------------------------------------------------------------------

def retention_chunkwise(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    gamma: torch.Tensor, chunk_size: int = 64,
) -> torch.Tensor:
    """Chunkwise retention — parallel within chunks, recurrent across chunks.

    Splits the sequence into chunks of size *chunk_size*:
    1. Within each chunk: use parallel retention (matrix multiply).
    2. Between chunks: pass the recurrent state  S.

    This gives a favourable trade-off: smaller memory than full parallel
    (O(T·C) vs O(T²)) and better GPU utilisation than pure recurrent.

    Args:
        q, k, v: (B, H, T, D).
        gamma:   (H,) per-head decay.
        chunk_size: Number of timesteps per chunk.

    Returns:
        (B, H, T, D).
    """
    B, H, T, D = q.shape
    device = q.device
    dtype = q.dtype
    scale = D ** -0.5

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - T % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
    T_pad = T + pad_len
    num_chunks = T_pad // chunk_size

    # Reshape into chunks:  (B, H, num_chunks, chunk_size, D)
    q_c = q.view(B, H, num_chunks, chunk_size, D)
    k_c = k.view(B, H, num_chunks, chunk_size, D)
    v_c = v.view(B, H, num_chunks, chunk_size, D)

    g = gamma.view(H, 1, 1).float()   # (H, 1, 1)
    S = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
    outputs: list[torch.Tensor] = []

    for c_idx in range(num_chunks):
        q_chunk = q_c[:, :, c_idx].float() * scale  # (B, H, C, D)
        k_chunk = k_c[:, :, c_idx].float()          # (B, H, C, D)
        v_chunk = v_c[:, :, c_idx].float()          # (B, H, C, D)
        C = chunk_size

        # ---------------------------------------------------------------
        # Within-chunk parallel retention (same as parallel form)
        # ---------------------------------------------------------------
        # Q K^T  (B, H, C, C)
        scores = torch.einsum("bhcd,bhld->bhcl", q_chunk, k_chunk)

        # Within-chunk decay mask
        idx_c = torch.arange(C, device=device)
        i_c, j_c = idx_c.unsqueeze(1), idx_c.unsqueeze(0)
        dist_c = i_c - j_c
        D_c = (g ** dist_c.clamp(min=0).float().unsqueeze(0))  # (H, C, C)
        D_c = D_c * (dist_c >= 0).float().unsqueeze(0)        # (H, C, C)
        # D_c now (H, C, C) → broadcast to (B, H, C, C)
        scores = scores * D_c.unsqueeze(0)

        # Within-chunk output
        out_inner = torch.einsum("bhcl,bhld->bhcd", scores, v_chunk)

        # ---------------------------------------------------------------
        # Cross-chunk contribution (from recurrent state S)
        # ---------------------------------------------------------------
        # The contribution of all past chunks to current positions:
        # For position j in chunk c, past state contributes:
        #   Q_{c,j} · (γ^{c*C + j} · S_{c-1})
        # where S_{c-1} is the accumulated outer-product state.

        # Decay factor from start of chunk to each position
        pos_in_chunk = torch.arange(C, device=device).float()     # (C,)
        chunk_start = c_idx * chunk_size
        # For position j:  decay from the last token of previous chunk
        # The recurrence accumulates as:  γ^j  S  after j steps
        # Actually:  (γ^C)^c · S forms the state at chunk start.
        # After j steps within chunk:  γ^j · (state from previous chunks)
        # pos_in_chunk gives 0..C-1, but state S represents the last token
        # of the previous chunk, so the first token of this chunk needs γ^1
        decay_per_pos = g.squeeze(-1) ** (pos_in_chunk + 1)  # (H, C)

        # Cross contribution:  Q @ (S * decay)
        # S is (B, H, D, D), q is (B, H, C, D)
        cross = torch.einsum("bhcd,bhde->bhce", q_chunk, S)          # (B, H, C, D)
        cross = cross * decay_per_pos.unsqueeze(0).unsqueeze(-1)    # (B, H, C, D)

        # Total chunk output
        out_chunk = out_inner + cross

        # ---------------------------------------------------------------
        # Update recurrent state for next chunk
        # ---------------------------------------------------------------
        # Accumulate the entire chunk's contribution to S:
        # S_new = γ^C · S_old + Σ_{j=0}^{C-1} γ^{C-1-j} K_j^T V_j
        # This is exactly the recursive state update applied to the chunk.

        # Decay the existing state by chunk_size
        S = (g.squeeze(-1) ** C).unsqueeze(-1) * S           # (B, H, D, D)

        # Add new contributions from this chunk
        for j in range(C):
            decay_j = g ** (C - 1 - j)                       # (H, 1, 1)
            kj = k_chunk[:, :, j].unsqueeze(-1)              # (B, H, D, 1)
            vj = v_chunk[:, :, j].unsqueeze(-2)              # (B, H, 1, D)
            S = S + decay_j.unsqueeze(0) * (kj @ vj)

        outputs.append(out_chunk)

    out = torch.cat(outputs, dim=2).to(dtype)                # (B, H, T_pad, D)
    if pad_len > 0:
        out = out[:, :, :T]                                   # trim padding
    return out


# ---------------------------------------------------------------------------
# Multi-Scale Retention layer
# ---------------------------------------------------------------------------

class MultiScaleRetention(nn.Module):
    """Multi-head, multi-scale retention with parallel and recurrent modes.

    Each head has its own decay rate γ_h, initialised uniformly between
    γ_min and γ_max.  In **parallel** mode the full O(T²) computation is
    used; in **recurrent** mode the O(T·D²) state-passing form is used.

    Args:
        dim: Model dimension.
        num_heads: Number of retention heads.
        head_dim: Dimension per head (default: dim // num_heads).
        gamma_min: Minimum decay (shortest effective context).
        gamma_max: Maximum decay (longest effective context).
        double_v_dim: If True, double the value dimension and split
            into two parts, using one for gating (as in the paper).
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        head_dim: int | None = None,
        gamma_min: float = 0.84,
        gamma_max: float = 0.98,
        double_v_dim: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.double_v_dim = double_v_dim
        v_dim = self.head_dim * (2 if double_v_dim else 1)
        inner_dim = num_heads * self.head_dim

        self.w_q = nn.Linear(dim, inner_dim, bias=False)
        self.w_k = nn.Linear(dim, inner_dim, bias=False)
        self.w_v = nn.Linear(dim, num_heads * v_dim, bias=False)

        # Per-head decay rates (stored as logit to ensure 0 < γ < 1)
        gamma_init = torch.linspace(
            self._gamma_to_logit(gamma_min),
            self._gamma_to_logit(gamma_max),
            num_heads,
        )
        self.gamma_logit = nn.Parameter(gamma_init)

        self.out_proj = nn.Linear(num_heads * v_dim, dim, bias=False)
        self.norm = RMSNorm(dim)

        # Swish gate (optional; from the paper)
        self.swish_gate = nn.Linear(dim, num_heads * v_dim, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, num_heads * v_dim)

    @staticmethod
    def _gamma_to_logit(gamma: float) -> float:
        import math
        return math.log(gamma / (1.0 - gamma))

    def _get_gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_logit)  # (H,)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "parallel",
        chunk_size: int = 64,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Multi-scale retention forward pass.

        Args:
            x:          (B, T, D) input sequence.
            mode:       'parallel', 'recurrent', or 'chunkwise'.
            chunk_size: Chunk size for chunkwise mode.
            state:      Initial recurrent state (B, H, D, D).
                        Only used in 'recurrent' mode.

        Returns:
            If mode='recurrent': (output, final_state).
            Otherwise: output tensor (B, T, D).
        """
        B, T, D = x.shape
        H = self.num_heads
        d = self.head_dim

        x_norm = self.norm(x)

        # Per-head value dimension (may be doubled for richer representation)
        vd = d * (2 if self.double_v_dim else 1)

        # Project to Q, K, V
        q = self.w_q(x_norm).view(B, T, H, d).transpose(1, 2)        # (B, H, T, d)
        k = self.w_k(x_norm).view(B, T, H, d).transpose(1, 2)
        v = self.w_v(x_norm).view(B, T, H, vd).transpose(1, 2)       # (B, H, T, vd)

        gamma = self._get_gamma()  # (H,)

        # Split v for retention: the retention functions require q, k, v to
        # share the same head dimension.  When double_v_dim is True the extra
        # channels are passed through and concatenated after retention.
        v_ret = v if vd == d else v[..., :d]

        # Dispatch by mode
        if mode == "parallel":
            out = retention_parallel(q, k, v_ret, gamma)
        elif mode == "recurrent":
            out, new_state = retention_recurrent(q, k, v_ret, gamma, state)
        elif mode == "chunkwise":
            out = retention_chunkwise(q, k, v_ret, gamma, chunk_size)
        else:
            raise ValueError(f"Unknown mode: {mode}. "
                             f"Use 'parallel', 'recurrent', or 'chunkwise'.")

        # Concatenate extra value channels when double_v_dim is enabled
        if vd != d:
            v_extra = v[..., d:]                              # (B, H, T, vd-d)
            out = torch.cat([out, v_extra], dim=-1)           # (B, H, T, vd)

        out = out.transpose(1, 2).reshape(B, T, H * vd)       # (B, T, H*vd)

        # Group norm + swish gate (from the paper)
        # GroupNorm expects (N, C, *) — transpose to (N, C, T), then back
        gate = self.swish_gate(x_norm)
        out = out.transpose(1, 2)                              # (B, C, T)
        out = self.group_norm(out)
        out = out.transpose(1, 2)                              # (B, T, C)
        out = out * gate

        out = self.out_proj(out)                               # (B, T, D)

        if mode == "recurrent":
            return out, new_state
        return out

    def reset_parameters(self) -> None:
        for module in [self.w_q, self.w_k, self.w_v, self.out_proj, self.swish_gate]:
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
        self.norm.reset_parameters()
        # Re-init gamma and group_norm
        nn.init.constant_(self.gamma_logit, 0.0)
        self.group_norm.reset_parameters()


# ---------------------------------------------------------------------------
# RetNet Block
# ---------------------------------------------------------------------------

class RetNetBlock(nn.Module):
    """A complete RetNet layer:  retention → FFN.

    Follows the pre-norm structure::

        x = x + retention(norm1(x))
        x = x + ffn(norm2(x))

    Args:
        dim: Model dimension.
        num_heads: Number of retention heads.
        head_dim: Dimension per head.
        gamma_min, gamma_max: Decay range.
        ffn_mult: FFN hidden = ffn_mult * dim.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        head_dim: int | None = None,
        gamma_min: float = 0.84,
        gamma_max: float = 0.98,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.retention = MultiScaleRetention(
            dim=dim, num_heads=num_heads, head_dim=head_dim,
            gamma_min=gamma_min, gamma_max=gamma_max,
        )

        ffn_dim = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "parallel",
        state: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x:     (B, T, D).
            mode:  'parallel', 'recurrent', or 'chunkwise'.
            state: Initial recurrent state (only for 'recurrent').

        Returns:
            If mode='recurrent': (output, final_state), else output.
        """
        # Retention sub-block
        ret_out = self.retention(self.norm1(x), mode=mode, state=state)
        if mode == "recurrent":
            ret_out, new_state = ret_out
            x = x + self.dropout(ret_out)
        else:
            x = x + self.dropout(ret_out)

        # FFN sub-block
        x = x + self.dropout(self.ffn(self.norm2(x)))

        if mode == "recurrent":
            return x, new_state
        return x

    def reset_parameters(self) -> None:
        self.retention.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
