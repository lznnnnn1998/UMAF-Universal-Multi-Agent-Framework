"""Mega-style moving average gated attention.

Reference: Ma, X., Zhou, C., Kong, X., He, J., Gui, L., Neubig, G.,
May, J., & Zettlemoyer, L. (2023). "Mega: Moving Average Equipped
Gated Attention."  arXiv:2209.10655.

Mega combines:
- **Exponential Moving Average (EMA)** for local smoothing / positional encoding.
- **Single-head gated attention** for token mixing, applied to the EMA outputs.
- **Gating mechanisms** (input gate, forget gate, output gate) inspired by
  gated RNNs but operating in parallel across the sequence.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ExponentialMovingAverage(nn.Module):
    """Multi-dimensional Exponential Moving Average (EMA).

    For an input sequence *x* of length *L*, computes::

        h_t = α ⊙ x_t  +  (1 − α_damp) ⊙ h_{t−1}

    where *α = σ(α_logit)* is the smoothing factor and
    *α_damp = exp(−Δ · δ)* introduces a per-channel damping controlled
    by a learned time-constant *δ*.

    The EMA is applied causally (each *h_t* only depends on *x_{≤t}*).

    Args:
        d_model: Input / output dimension.
        bidirectional: If ``True``, apply EMA in both forward and backward
                       directions and average (not causal).
    """

    def __init__(self, d_model: int, bidirectional: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional

        # Smoothing factor α (learned per channel)
        self.alpha_logit = nn.Parameter(torch.zeros(d_model))

        # Damping factor δ (learned per channel)
        self.delta = nn.Parameter(torch.zeros(d_model))

        # Step size Δ (used as a multiplier on delta)
        self.register_buffer("_dt_scale", torch.tensor(1.0))

    def _compute_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute EMA coefficients for each position.

        Returns:
            alpha: ``(d_model,)`` smoothing factor in (0, 1).
            alpha_damp: ``(d_model,)`` damped smoothing factor.
        """
        alpha = torch.sigmoid(self.alpha_logit)                         # (D,)
        delta_dt = torch.exp(self.delta) * self._dt_scale              # (D,)
        alpha_damp = torch.exp(-delta_dt)                               # (D,) — in (0,1)
        return alpha, alpha_damp

    def _ema_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal forward EMA."""
        B, L, D = x.shape
        alpha, alpha_damp = self._compute_coefficients()

        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs: list[torch.Tensor] = []
        for t in range(L):
            x_t = x[:, t, :]                                            # (B, D)
            h = alpha * x_t + (1.0 - alpha_damp) * h                    # (B, D)
            outputs.append(h)
        return torch.stack(outputs, dim=1)                               # (B, L, D)

    def _ema_backward(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse-direction EMA (anti-causal)."""
        return torch.flip(
            self._ema_forward(torch.flip(x, dims=[1])),
            dims=[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply EMA to input.

        Args:
            x: ``(B, L, d_model)`` input.

        Returns:
            ``(B, L, d_model)`` EMA-smoothed output.
        """
        if self.bidirectional:
            fwd = self._ema_forward(x)
            bwd = self._ema_backward(x)
            return (fwd + bwd) / 2.0
        else:
            return self._ema_forward(x)


class MegaGatedAttention(nn.Module):
    """Single-head gated attention used inside Mega.

    Unlike standard multi-head attention, Mega uses a single attention head
    with a **gating mechanism**:

    - *Input gate* η controls how much the attention output contributes.
    - *Forget gate* φ controls the recurrent component (optional, controlled
      via the EMA sub-layer).
    - Final output is gated: ``y = η ⊙ Attention(x) + (1−η) ⊙ x``.

    Attention uses EMA to compute a relative-position-like bias.

    Args:
        d_model: Model dimension.
        dropout: Attention dropout probability.
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = d_model ** -0.5

        # Q, K, V for single-head attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Input gate η (controls contribution of attention branch)
        self.input_gate = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass — single-head gated attention.

        Args:
            x: ``(B, L, d_model)`` input.
            mask: Optional ``(L, L)`` or ``(B, L, L)`` attention mask.

        Returns:
            ``(B, L, d_model)`` output.
        """
        B, L, D = x.shape

        Q = self.q_proj(x)                                             # (B, L, D)
        K = self.k_proj(x)                                             # (B, L, D)
        V = self.v_proj(x)                                             # (B, L, D)

        # Scaled dot-product attention (single head, D-dim)
        attn = torch.einsum("bld,bmd->blm", Q, K) * self.scale         # (B, L, L)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        additive_mask = torch.where(causal_mask, float("-inf"), 0.0)
        additive_mask = additive_mask.view(1, L, L)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.view(1, L, L)
            additive_mask = additive_mask + mask

        attn = attn + additive_mask
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.bmm(attn_weights, V)                          # (B, L, D)

        # Input gating
        eta = torch.sigmoid(self.input_gate(x))                         # (B, L, D)
        gated = eta * attn_out + (1.0 - eta) * x                       # (B, L, D)

        return self.out_proj(gated)


class MegaLayer(nn.Module):
    """Mega layer: EMA sub-layer → Gated Attention sub-layer.

    Follows the pre-norm Transformer pattern with the EMA acting as a
    positional encoder substitute:

    ::

        x'   = EMA(Norm(x)) + x           (EMA sub-layer)
        y    = GatedAttn(Norm(x')) + x'    (Attention sub-layer)

    Args:
        d_model: Model dimension.
        dropout: Dropout probability.
        bidirectional_ema: If ``True``, use bidirectional EMA.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        bidirectional_ema: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.ema = ExponentialMovingAverage(d_model, bidirectional=bidirectional_ema)
        self.gated_attn = MegaGatedAttention(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, L, d_model)`` input.

        Returns:
            ``(B, L, d_model)`` output.
        """
        # EMA sub-layer (pre-norm)
        x = self.ema(self.norm1(x)) + x
        x = self.dropout(x)

        # Gated attention sub-layer (pre-norm)
        x = self.gated_attn(self.norm2(x)) + x
        x = self.dropout(x)

        return x
