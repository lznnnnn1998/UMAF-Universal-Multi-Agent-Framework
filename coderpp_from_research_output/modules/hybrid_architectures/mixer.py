"""Unified sequence mixer abstraction.

Provides a common interface for both attention-based and recurrent
sequence mixing operators, plus a configurable hybrid mixer that can
combine multiple mixing paths with kernel fusion support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SlidingWindowAttention
from .ssm import DiagonalSSM


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MixerMode(Enum):
    """Operating mode for a sequence mixer."""

    ATTENTION = "attention"   # Softmax attention (quadratic or windowed)
    RECURRENT = "recurrent"   # SSM / RNN-style (sub-quadratic)
    HYBRID = "hybrid"         # Both pathways active simultaneously


class KernelFusionPath(Enum):
    """Kernel fusion strategy for the hybrid mixer.

    - **SERIAL**: Run attention then SSM sequentially (higher accuracy,
      higher latency).
    - **PARALLEL**: Run attention and SSM in parallel, sum outputs.
    - **INTERLEAVED**: Alternate attention and SSM blocks.
    """

    SERIAL = "serial"
    PARALLEL = "parallel"
    INTERLEAVED = "interleaved"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SequenceMixer(nn.Module, ABC):
    """Abstract base class for sequence mixing operators.

    A sequence mixer transforms an input ``(B, L, D)`` tensor into an
    output of the same shape.  Subclasses implement one or more of:

    - ``forward_attention()``   — softmax / linear attention mode
    - ``forward_recurrent()``   — SSM / recurrent mode
    - ``forward_hybrid()``      — combined mode

    The public ``forward()`` dispatches to the appropriate method based
    on *mode*.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Attention-mode forward."""
        ...

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """Recurrent-mode forward.  Default: raise NotImplementedError."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support recurrent mode."
        )

    def forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        """Hybrid-mode forward.  Default: sum of attention + recurrent."""
        return self.forward_attention(x) + self.forward_recurrent(x)

    def forward(self, x: torch.Tensor, mode: MixerMode | str = MixerMode.HYBRID) -> torch.Tensor:
        """Dispatch to the appropriate mode.

        Args:
            x: ``(B, L, d_model)`` input.
            mode: ``"attention"``, ``"recurrent"``, or ``"hybrid"``.

        Returns:
            ``(B, L, d_model)`` output.
        """
        if isinstance(mode, str):
            mode = MixerMode(mode)

        if mode == MixerMode.ATTENTION:
            return self.forward_attention(x)
        elif mode == MixerMode.RECURRENT:
            return self.forward_recurrent(x)
        elif mode == MixerMode.HYBRID:
            return self.forward_hybrid(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Concrete mixers
# ---------------------------------------------------------------------------

class SSMMixer(SequenceMixer):
    """Pure SSM-based sequence mixer."""

    def __init__(self, d_model: int, d_state: int = 64) -> None:
        super().__init__(d_model)
        self.ssm = DiagonalSSM(d_state, d_model)

    def forward_attention(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SSMMixer does not support attention mode.")

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssm(x, mode="recurrent")[0]

    def forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssm(x, mode="convolution")  # conv = parallel


class AttentionMixer(SequenceMixer):
    """Pure attention-based sequence mixer."""

    def __init__(
        self, d_model: int, n_heads: int = 8, window_size: int | None = None
    ) -> None:
        super().__init__(d_model)
        self.attn = SlidingWindowAttention(
            d_model, n_heads, window_size=window_size or 128
        )

    def forward_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AttentionMixer does not support recurrent mode.")

    def forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_attention(x)


# ---------------------------------------------------------------------------
# Hybrid mixer
# ---------------------------------------------------------------------------

class HybridMixer(SequenceMixer):
    """Configurable hybrid mixer combining attention and SSM pathways.

    Supports three kernel fusion strategies (see :class:`KernelFusionPath`):

    *SERIAL*
        ``x → Attn → Norm → SSM → Output``
        Each stage refines the previous one.  Provides the highest
        expressive power at the cost of sequential latency.

    *PARALLEL*
        ``x → Attn ─┐``
        ``x → SSM  ─┴→ Sum → Output``
        Both pathways operate on the same input and their outputs are
        summed.  Lower latency, suitable when attention and SSM capture
        complementary features.

    *INTERLEAVED*
        ``x → [Attn → SSM] × num_blocks → Output``
        Repeats a block of (attention, SSM) multiple times, similar to
        architectures like MambaFormer.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_state: SSM state dimension.
        window_size: Attention window size (None → full attention).
        num_blocks: Number of interleaved blocks (only for INTERLEAVED).
        fusion_path: Kernel fusion strategy.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_state: int = 64,
        window_size: int | None = 128,
        num_blocks: int = 2,
        fusion_path: KernelFusionPath | str = KernelFusionPath.INTERLEAVED,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(d_model)

        if isinstance(fusion_path, str):
            fusion_path = KernelFusionPath(fusion_path)
        self.fusion_path = fusion_path
        self.num_blocks = num_blocks

        # --- Parallel / Interleaved pathways ---
        if fusion_path in (KernelFusionPath.PARALLEL, KernelFusionPath.INTERLEAVED):
            self.attention_blocks = nn.ModuleList([
                SlidingWindowAttention(
                    d_model, n_heads,
                    window_size=window_size or 128,
                    dropout=dropout,
                )
                for _ in range(num_blocks if fusion_path == KernelFusionPath.INTERLEAVED else 1)
            ])
            self.ssm_blocks = nn.ModuleList([
                DiagonalSSM(d_state, d_model)
                for _ in range(num_blocks if fusion_path == KernelFusionPath.INTERLEAVED else 1)
            ])
            self.norms_attn = nn.ModuleList([
                nn.LayerNorm(d_model)
                for _ in range(num_blocks if fusion_path == KernelFusionPath.INTERLEAVED else 1)
            ])
            self.norms_ssm = nn.ModuleList([
                nn.LayerNorm(d_model)
                for _ in range(num_blocks if fusion_path == KernelFusionPath.INTERLEAVED else 1)
            ])
            self.dropout = nn.Dropout(dropout)

        # --- Alpha blending for parallel mode ---
        if fusion_path == KernelFusionPath.PARALLEL:
            self.alpha_attn = nn.Parameter(torch.ones(1))
            self.alpha_ssm = nn.Parameter(torch.ones(1))

        # --- Serial pathway ---
        if fusion_path == KernelFusionPath.SERIAL:
            self.attn = SlidingWindowAttention(
                d_model, n_heads,
                window_size=window_size or 128,
                dropout=dropout,
            )
            self.ssm = DiagonalSSM(d_state, d_model)
            self.norm_attn = nn.LayerNorm(d_model)
            self.norm_ssm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

    def forward_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Attention-only forward (windowed or full)."""
        if self.fusion_path == KernelFusionPath.SERIAL:
            return self.attn(x)
        elif self.fusion_path in (KernelFusionPath.PARALLEL, KernelFusionPath.INTERLEAVED):
            return self.attention_blocks[0](x)
        return x

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """Recurrent (SSM) forward."""
        if self.fusion_path == KernelFusionPath.SERIAL:
            return self.ssm(x, mode="convolution")
        elif self.fusion_path in (KernelFusionPath.PARALLEL, KernelFusionPath.INTERLEAVED):
            return self.ssm_blocks[0](x, mode="convolution")
        return x

    def forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        """Hybrid forward dispatching on the configured fusion path."""
        if self.fusion_path == KernelFusionPath.SERIAL:
            return self._forward_serial(x)
        elif self.fusion_path == KernelFusionPath.PARALLEL:
            return self._forward_parallel(x)
        elif self.fusion_path == KernelFusionPath.INTERLEAVED:
            return self._forward_interleaved(x)
        return x

    def _forward_serial(self, x: torch.Tensor) -> torch.Tensor:
        """Serial: Attn → SSM."""
        h = self.attn(self.norm_attn(x)) + x
        h = self.dropout(h)
        h = self.ssm(self.norm_ssm(h), mode="convolution") + h
        return self.dropout(h)

    def _forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel: α·Attn(x) + β·SSM(x)."""
        a = self.attention_blocks[0](self.norms_attn[0](x)) + x
        s = self.ssm_blocks[0](self.norms_ssm[0](x), mode="convolution") + x
        return self.alpha_attn * a + self.alpha_ssm * s

    def _forward_interleaved(self, x: torch.Tensor) -> torch.Tensor:
        """Interleaved: [Attn → SSM] × num_blocks."""
        h = x
        for i in range(self.num_blocks):
            h = self.attention_blocks[i](self.norms_attn[i](h)) + h
            h = self.dropout(h)
            h = self.ssm_blocks[i](self.norms_ssm[i](h), mode="convolution") + h
            h = self.dropout(h)
        return h

    @property
    def supported_modes(self) -> list[MixerMode]:
        """Return the modes this mixer supports."""
        return [MixerMode.ATTENTION, MixerMode.RECURRENT, MixerMode.HYBRID]
