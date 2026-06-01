"""RWKV-style token-mixing and channel-mixing blocks.

Implements the core WKV (Weighted Key Value) operator — an attention
alternative that uses exponential decay over time instead of softmax
over the full sequence.  Includes TimeMixBlock (causal token mixing
with WKV), ChannelMixBlock (per-token FFN with squared-ReLU gating),
and a full RWKVBlock that stacks them.

Reference
---------
- RWKV-v4 / RWKV: Reinventing RNNs for the Transformer Era
  (Peng et al., 2023)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm, SquaredReLU


# ---------------------------------------------------------------------------
# Token shift
# ---------------------------------------------------------------------------

def token_shift(x: torch.Tensor) -> torch.Tensor:
    """Shift tokens one step along the time axis for causal mixing.

    Returns a tensor of shape (B, T, 2*D) where the first half is x
    shifted right by one position (zero-padded at the front) and the
    second half is the original x.  This gives each position access to
    both the current token and the previous token's representation.

    Args:
        x: (B, T, D)

    Returns:
        (B, T, 2*D)
    """
    # Shift right: [0, x0, x1, ..., x_{T-2}]
    shifted = torch.roll(x, shifts=1, dims=1)
    shifted[:, 0] = 0.0
    return torch.cat([shifted, x], dim=-1)


# ---------------------------------------------------------------------------
# WKV operator
# ---------------------------------------------------------------------------

class WKVOperator(nn.Module):
    """Weighted Key Value operator — a causal, recurrent attention alternative.

    For each channel *c* and position *t* the operator computes::

        a_t[c] = e^{-w[c]} · a_{t-1}[c]  +  e^{k_t[c]} · v_t[c]
        b_t[c] = e^{-w[c]} · b_{t-1}[c]  +  e^{k_t[c]}
        wkv_t[c] = (a_{t-1}[c] + e^{u[c] + k_t[c]} · v_t[c])
                 / (b_{t-1}[c] + e^{u[c] + k_t[c]})

    where *w* controls the per-channel exponential decay rate and *u*
    gives the current token a bonus (``time_first``).

    The recurrence runs in O(T·D) time with O(D) state, making it far
    more memory-efficient than self-attention for long sequences.

    Args:
        dim: Number of channels (feature dimension).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Learn in log-space; map to (0, 1) via  exp(-exp(w)) for stability.
        self.time_decay = nn.Parameter(torch.linspace(-2.0, 2.0, dim))
        self.time_first = nn.Parameter(torch.zeros(dim))

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute WKV for a full sequence.

        Args:
            k: key tensor,  shape (B, T, D) — raw logits (will be exp'd).
            v: value tensor, shape (B, T, D).

        Returns:
            wkv: (B, T, D) — weighted combination of past values.
        """
        B, T, D = k.shape
        device = k.device

        # Channel-wise decay  ∈ (0, 1)
        decay = torch.exp(-torch.exp(self.time_decay)).to(device)  # (D,)
        bonus = self.time_first.to(device)  # (D,)

        # Accumulators:  a = numerator,  b = denominator
        a = torch.zeros(B, D, device=device, dtype=torch.float32)
        b = torch.zeros(B, D, device=device, dtype=torch.float32)

        k_f32 = k.float()
        v_f32 = v.float()
        decay_f32 = decay.float()
        bonus_f32 = bonus.float()

        outputs: list[torch.Tensor] = []
        for t in range(T):
            ek = torch.exp(k_f32[:, t])  # (B, D)
            ev = v_f32[:, t]             # (B, D)
            ekv = ek * ev                # (B, D)

            # Current-position term with "first-token" bonus
            bonus_ek = torch.exp(bonus_f32) * ek  # (B, D)

            num = a + bonus_ek * ev      # (B, D)
            den = b + bonus_ek           # (B, D)
            wkv_t = num / (den + 1e-8)
            outputs.append(wkv_t.unsqueeze(1))

            # Advance state
            a = decay_f32 * a + ekv
            b = decay_f32 * b + ek

        return torch.cat(outputs, dim=1).to(k.dtype)  # (B, T, D)

    def reset_parameters(self) -> None:
        nn.init.constant_(self.time_decay, 0.0)
        nn.init.constant_(self.time_first, 0.0)


# ---------------------------------------------------------------------------
# Time-mix block
# ---------------------------------------------------------------------------

class TimeMixBlock(nn.Module):
    """RWKV time-mixing block — causal token interaction via WKV.

    Flow::

        shifted = token_shift(norm(x))          # (B, T, 2*D)
        r  = σ(W_r @ shifted)                   # receptance
        k  =     W_k @ shifted                   # key
        v  =     W_v @ shifted                   # value
        wkv = WKVOperator(k, v)                 # (B, T, D)
        out = W_o @ (r ⊙ wkv)                   # output projection

    Args:
        dim: Feature dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        # Each linear takes the concatenated [shifted_x, x] → 2*dim inputs
        self.w_r = nn.Linear(dim * 2, dim, bias=False)
        self.w_k = nn.Linear(dim * 2, dim, bias=False)
        self.w_v = nn.Linear(dim * 2, dim, bias=False)
        self.w_out = nn.Linear(dim, dim, bias=False)
        self.wkv = WKVOperator(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        shifted = token_shift(x_norm)             # (B, T, 2*D)

        r = torch.sigmoid(self.w_r(shifted))      # (B, T, D)
        k = self.w_k(shifted)                     # (B, T, D)
        v = self.w_v(shifted)                     # (B, T, D)

        wkv_out = self.wkv(k, v)                  # (B, T, D)
        out = self.w_out(r * wkv_out)             # (B, T, D)
        return residual + out

    def reset_parameters(self) -> None:
        for linear in [self.w_r, self.w_k, self.w_v, self.w_out]:
            nn.init.xavier_uniform_(linear.weight, gain=0.5)
        self.wkv.reset_parameters()


# ---------------------------------------------------------------------------
# Channel-mix block
# ---------------------------------------------------------------------------

class ChannelMixBlock(nn.Module):
    """RWKV channel-mixing block — per-token FFN with squared-ReLU gating.

    Equivalent to a gated feed-forward network applied independently at
    each position::

        shifted = token_shift(norm(x))
        r  = σ(W_r @ shifted - μ_r)              # receptance
        k  =    W_k @ shifted - μ_k              # key
        out = r ⊙ (W_v @ max(k, 0)^2)           # gated squared-ReLU

    Args:
        dim: Feature dimension.
        hidden_dim: Intermediate FFN dimension (default: 4*dim).
    """

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.norm = RMSNorm(dim)
        self.w_r = nn.Linear(dim * 2, dim, bias=False)
        self.w_k = nn.Linear(dim * 2, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, dim, bias=False)
        self.squared_relu = SquaredReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        shifted = token_shift(x_norm)

        r = torch.sigmoid(self.w_r(shifted))       # (B, T, D)
        k = self.w_k(shifted)                      # (B, T, H)
        v = self.squared_relu(k)                   # (B, T, H) — max(0, k)^2
        out = r * self.w_v(v)                      # (B, T, D)
        return residual + out

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_r.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w_k.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w_v.weight, gain=0.5)


# ---------------------------------------------------------------------------
# Full RWKV block
# ---------------------------------------------------------------------------

class RWKVBlock(nn.Module):
    """A complete RWKV layer: time-mix → channel-mix.

    Args:
        dim: Feature dimension.
        hidden_dim: Intermediate dimension for the channel-mix FFN.
        dropout: Dropout applied after each sub-block (default 0.0).
    """

    def __init__(self, dim: int, hidden_dim: int | None = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.time_mix = TimeMixBlock(dim)
        self.channel_mix = ChannelMixBlock(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_mix(x)
        x = self.dropout(x)
        x = self.channel_mix(x)
        return x
