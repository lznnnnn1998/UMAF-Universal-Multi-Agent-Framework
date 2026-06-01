"""Hybrid Architectures: attention + sub-quadratic sequence models.

This package implements architectures that strategically combine attention
mechanisms with efficient sub-quadratic sequence models for long-range
sequence processing.

Modules:
    ssm: Diagonal State Space Models (S4D-style)
    attention: Sliding window attention and variants
    h3: H3 (Hungry Hungry Hippos) gated SSM layers
    gla: Gated Linear Attention with chunkwise parallel forms
    mega: Mega-style EMA-gated attention
    mixer: Unified sequence mixer abstraction
"""

from .ssm import DiagonalSSM
from .attention import SlidingWindowAttention
from .h3 import H3Layer
from .gla import GatedLinearAttention, gla_chunkwise, gla_recurrent
from .mega import MegaLayer, ExponentialMovingAverage
from .mixer import (
    MixerMode,
    SequenceMixer,
    HybridMixer,
    KernelFusionPath,
)

__all__ = [
    # SSM
    "DiagonalSSM",
    # Attention
    "SlidingWindowAttention",
    # H3
    "H3Layer",
    # GLA
    "GatedLinearAttention",
    "gla_chunkwise",
    "gla_recurrent",
    # Mega
    "MegaLayer",
    "ExponentialMovingAverage",
    # Mixer
    "MixerMode",
    "SequenceMixer",
    "HybridMixer",
    "KernelFusionPath",
]
