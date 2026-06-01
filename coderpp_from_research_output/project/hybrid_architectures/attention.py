"""Attention variants including sliding-window and linear attention.

Implements:
- SlidingWindowAttention: multi-head attention restricted to a local window.
- CausalMask helper for efficient causal masking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _causal_sliding_mask(
    seq_len: int, window_size: int, device: torch.device | None = None
) -> torch.Tensor:
    """Build a causal sliding-window boolean mask.

    Position *i* may attend to positions *j* ∈ [max(0, i−window_size+1), i].

    Args:
        seq_len: Sequence length.
        window_size: Number of recent positions to attend to.
        device: Target device.

    Returns:
        ``(seq_len, seq_len)`` boolean tensor; ``True`` = attend.
    """
    row = torch.arange(seq_len, device=device).unsqueeze(1)   # (L, 1)
    col = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, L)
    return (col <= row) & (col >= row - window_size + 1)


class SlidingWindowAttention(nn.Module):
    """Multi-head sliding window attention.

    Each query token attends to at most ``window_size`` previous key/value tokens
    (including itself).  Beyond the window the attention weight is zero.

    This is the standard building block used in architectures such as
    Mistral, Longformer, and the hybrid architectures in this package.

    Args:
        d_model: Model dimension (must be divisible by *n_heads*).
        n_heads: Number of attention heads.
        window_size: Number of tokens in the attention window.
        dropout: Attention dropout probability.
        bias: Whether to use bias in Q/K/V projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, L, d_model)`` input sequence.
            mask: ``(L, L)`` or ``(B, 1, L, L)`` optional additional mask
                  (combined with the causal window mask via AND).

        Returns:
            ``(B, L, d_model)`` output sequence.
        """
        B, L, D = x.shape
        H = self.n_heads

        # Linear projection → split into Q, K, V
        qkv = self.qkv(x)                                          # (B, L, 3D)
        q, k, v = rearrange(
            qkv, "b l (three h d) -> three b h l d", three=3, h=H
        )

        # Scaled dot-product attention
        attn_weights = torch.einsum("bhld,bhmd->bhlm", q, k) * self.scale  # (B, H, L, L)

        # Causal sliding-window mask
        window_mask = _causal_sliding_mask(L, self.window_size, device=x.device)  # (L, L)
        # Convert boolean → additive mask (0 for attend, −inf for ignore)
        additive_mask = torch.where(window_mask, 0.0, float("-inf"))
        additive_mask = additive_mask.view(1, 1, L, L)             # (1, 1, L, L)

        if mask is not None:
            # Merge with user-supplied mask
            if mask.dim() == 2:
                mask = mask.view(1, 1, L, L)
            additive_mask = additive_mask + mask

        attn_weights = attn_weights + additive_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum("bhlm,bhmd->bhld", attn_weights, v)     # (B, H, L, d)
        out = rearrange(out, "b h l d -> b l (h d)")                # (B, L, D)
        return self.out_proj(out)


class LinearAttention(nn.Module):
    """Linearised attention using the kernel trick (Katharopoulos et al. 2020).

    Instead of softmax(QK^T)V — O(L²) — this computes
    φ(Q) (φ(K)^T V) / φ(Q) (φ(K)^T 1) — O(L d²).

    Used here primarily as a comparison point for the gated variants (GLA).

    Args:
        d_model: Model dimension.
        n_heads: Number of heads.
        feature_map: ``"elu"`` (default, φ(x)=1+elu(x)) or ``"relu"``.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        feature_map: str = "elu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.feature_map = feature_map

    @staticmethod
    def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    @staticmethod
    def _relu_feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

    def _apply_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu":
            return self._elu_feature_map(x)
        elif self.feature_map == "relu":
            return self._relu_feature_map(x)
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — linear attention.

        Args:
            x: ``(B, L, d_model)`` input.

        Returns:
            ``(B, L, d_model)`` output.
        """
        B, L, D = x.shape
        H = self.n_heads

        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b l (three h d) -> three b h l d", three=3, h=H)

        # Apply feature map for positive similarities
        q = self._apply_feature_map(q)
        k = self._apply_feature_map(k)

        # Causal linear attention via cumulative sum
        # KV state: (B, H, d, d)
        # Normalizer: (B, H, d)
        outputs: list[torch.Tensor] = []
        kv_state = torch.zeros(B, H, self.head_dim, self.head_dim, device=x.device)
        k_sum = torch.zeros(B, H, self.head_dim, device=x.device)

        for t in range(L):
            k_t = k[:, :, t, :]      # (B, H, d)
            v_t = v[:, :, t, :]      # (B, H, d)
            q_t = q[:, :, t, :]      # (B, H, d)

            kv_state = kv_state + torch.einsum("bhk,bhv->bhkv", k_t, v_t)
            k_sum = k_sum + k_t

            num = torch.einsum("bhk,bhkv->bhv", q_t, kv_state)
            den = torch.einsum("bhk,bhk->bh", q_t, k_sum).unsqueeze(-1) + 1e-8
            out_t = num / den
            outputs.append(out_t)

        out = torch.stack(outputs, dim=2)                # (B, H, L, d)
        out = rearrange(out, "b h l d -> b l (h d)")     # (B, L, D)
        return self.out_proj(out)
