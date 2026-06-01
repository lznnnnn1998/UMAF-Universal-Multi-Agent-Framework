"""
Attention Entropy Analyzer — Diagnoses attention quality via entropy distributions.

Measures per-layer and per-head attention entropy to detect attention dilution
(attending to too many tokens), attention collapse (attending to too few tokens),
and healthy focused attention patterns.

Usage:
    from evaluation.entropy import (
        AttentionEntropyAnalyzer, LayerEntropyProfile,
        HeadEntropyDistribution, EntropyDiagnosis,
    )

    analyzer = AttentionEntropyAnalyzer()
    results = analyzer.analyze(attention_matrices)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch


# ─────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class HeadEntropyDistribution:
    """Per-head attention entropy statistics.

    Attributes:
        head_index: Index of the attention head within its layer.
        layer_index: Index of the transformer layer.
        mean_entropy: Mean entropy across all positions.
        std_entropy: Standard deviation of entropy across positions.
        max_entropy: Maximum entropy observed.
        min_entropy: Minimum entropy observed.
        max_possible_entropy: log(seq_len) — the theoretical maximum.
        normalized_entropy: mean_entropy / max_possible_entropy (0-1).
        dilution_flag: True if entropy is suspiciously high (dilution).
        collapse_flag: True if entropy is suspiciously low (collapse).
    """

    head_index: int
    layer_index: int
    mean_entropy: float = 0.0
    std_entropy: float = 0.0
    max_entropy: float = 0.0
    min_entropy: float = 0.0
    max_possible_entropy: float = 0.0
    normalized_entropy: float = 0.0
    dilution_flag: bool = False
    collapse_flag: bool = False

    def __repr__(self) -> str:
        status = "OK"
        if self.dilution_flag:
            status = "DILUTED"
        elif self.collapse_flag:
            status = "COLLAPSED"
        return (
            f"Head({self.layer_index},{self.head_index}) "
            f"entropy={self.mean_entropy:.3f} "
            f"norm={self.normalized_entropy:.3f} [{status}]"
        )


@dataclass
class LayerEntropyProfile:
    """Per-layer attention entropy profile.

    Attributes:
        layer_index: Index of the transformer layer.
        num_heads: Number of attention heads in this layer.
        head_distributions: List of HeadEntropyDistribution for each head.
        layer_mean_entropy: Mean entropy across all heads in this layer.
        layer_std_entropy: Standard deviation of per-head mean entropies.
        heads_collapsed: Number of heads flagged as collapsed.
        heads_diluted: Number of heads flagged as diluted.
        healthy_heads: Number of heads neither collapsed nor diluted.
    """

    layer_index: int
    num_heads: int
    head_distributions: list[HeadEntropyDistribution] = field(default_factory=list)
    layer_mean_entropy: float = 0.0
    layer_std_entropy: float = 0.0
    heads_collapsed: int = 0
    heads_diluted: int = 0
    healthy_heads: int = 0

    @property
    def health_ratio(self) -> float:
        """Fraction of heads that are healthy."""
        if self.num_heads == 0:
            return 0.0
        return self.healthy_heads / self.num_heads

    def __repr__(self) -> str:
        return (
            f"Layer({self.layer_index}) "
            f"entropy={self.layer_mean_entropy:.3f}±{self.layer_std_entropy:.3f} "
            f"healthy={self.healthy_heads}/{self.num_heads} "
            f"collapsed={self.heads_collapsed} diluted={self.heads_diluted}"
        )


@dataclass
class EntropyDiagnosis:
    """Overall attention entropy diagnosis across all layers.

    Attributes:
        num_layers: Total layers analyzed.
        layer_profiles: Per-layer profiles.
        total_heads: Total heads across all layers.
        heads_collapsed: Total collapsed heads.
        heads_diluted: Total diluted heads.
        healthy_heads: Total healthy heads.
        global_mean_entropy: Mean entropy across all heads/layers.
        health_verdict: "HEALTHY", "WARNING", or "UNHEALTHY".
    """

    num_layers: int
    layer_profiles: list[LayerEntropyProfile] = field(default_factory=list)
    total_heads: int = 0
    heads_collapsed: int = 0
    heads_diluted: int = 0
    healthy_heads: int = 0
    global_mean_entropy: float = 0.0
    health_verdict: str = "HEALTHY"

    def __repr__(self) -> str:
        return (
            f"EntropyDiagnosis(layers={self.num_layers}, heads={self.total_heads}, "
            f"mean_entropy={self.global_mean_entropy:.3f}, "
            f"healthy={self.healthy_heads}/{self.total_heads} "
            f"[{self.health_verdict}])"
        )


# ─────────────────────────────────────────────────────────────────────────
# Attention Entropy Analyzer
# ─────────────────────────────────────────────────────────────────────────


class AttentionEntropyAnalyzer:
    """Analyze attention entropy to diagnose attention quality.

    Computes per-head, per-layer Shannon entropy of attention distributions
    and flags heads exhibiting dilution (too uniform) or collapse (too peaked).

    Args:
        dilution_threshold: Normalized entropy above this value is
            flagged as dilution (default: 0.85).
        collapse_threshold: Normalized entropy below this value is
            flagged as collapse (default: 0.15).
        min_head_size: Minimum sequence length for meaningful analysis
            (default: 4).
    """

    def __init__(
        self,
        dilution_threshold: float = 0.85,
        collapse_threshold: float = 0.15,
        min_head_size: int = 4,
    ):
        if not 0.0 < collapse_threshold < dilution_threshold < 1.0:
            raise ValueError(
                "Must have 0 < collapse_threshold < dilution_threshold < 1"
            )
        self.dilution_threshold = dilution_threshold
        self.collapse_threshold = collapse_threshold
        self.min_head_size = min_head_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        attention_matrices: list[torch.Tensor] | torch.Tensor,
        layer_indices: list[int] | None = None,
    ) -> EntropyDiagnosis:
        """Analyze attention entropy across layers and heads.

        Args:
            attention_matrices: Either:
                - A list of tensors, each [batch, n_heads, seq_q, seq_k],
                  one per layer.
                - A single tensor [n_layers, batch, n_heads, seq_q, seq_k].
            layer_indices: Optional layer index labels (default: 0, 1, 2, ...).

        Returns:
            EntropyDiagnosis with full per-layer and per-head analysis.
        """
        if isinstance(attention_matrices, torch.Tensor):
            # Convert [L, B, H, Q, K] → list of [B, H, Q, K]
            attention_matrices = [attention_matrices[i] for i in range(attention_matrices.shape[0])]

        num_layers = len(attention_matrices)
        if layer_indices is None:
            layer_indices = list(range(num_layers))

        layer_profiles: list[LayerEntropyProfile] = []
        total_collapsed = 0
        total_diluted = 0
        total_healthy = 0
        total_heads = 0
        all_mean_entropies: list[float] = []

        for layer_idx, attn in enumerate(attention_matrices):
            profile = self._analyze_layer(
                attn, layer_index=layer_indices[layer_idx]
            )
            layer_profiles.append(profile)
            total_collapsed += profile.heads_collapsed
            total_diluted += profile.heads_diluted
            total_healthy += profile.healthy_heads
            total_heads += profile.num_heads
            all_mean_entropies.append(profile.layer_mean_entropy)

        global_mean = (
            sum(all_mean_entropies) / len(all_mean_entropies)
            if all_mean_entropies
            else 0.0
        )

        # Health verdict
        if total_heads == 0:
            verdict = "EMPTY"
        else:
            health_ratio = total_healthy / total_heads
            if health_ratio >= 0.8:
                verdict = "HEALTHY"
            elif health_ratio >= 0.5:
                verdict = "WARNING"
            else:
                verdict = "UNHEALTHY"

        return EntropyDiagnosis(
            num_layers=num_layers,
            layer_profiles=layer_profiles,
            total_heads=total_heads,
            heads_collapsed=total_collapsed,
            heads_diluted=total_diluted,
            healthy_heads=total_healthy,
            global_mean_entropy=global_mean,
            health_verdict=verdict,
        )

    def analyze_head(
        self, attention_weights: torch.Tensor, head_index: int = 0, layer_index: int = 0
    ) -> HeadEntropyDistribution:
        """Analyze entropy for a single attention head.

        Args:
            attention_weights: Tensor [batch, n_heads, seq_q, seq_k] or
                [seq_q, seq_k] for a single head.
            head_index: Index of the head.
            layer_index: Index of the layer.

        Returns:
            HeadEntropyDistribution.
        """
        if attention_weights.dim() == 4:
            # [B, H, Q, K] → extract head
            attention_weights = attention_weights[0, head_index]
        elif attention_weights.dim() == 3:
            attention_weights = attention_weights[head_index]

        return self._analyze_single_head(attention_weights, head_index, layer_index)

    def compute_entropy(self, probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Compute Shannon entropy: H = -Σ p * log(p).

        Args:
            probs: Probability distribution tensor (must sum to ~1 along dim).
            dim: Dimension to compute entropy over.

        Returns:
            Entropy tensor with ``dim`` removed.
        """
        # Clamp to avoid log(0)
        eps = 1e-12
        probs_safe = probs.clamp(min=eps)
        entropy = -(probs_safe * torch.log(probs_safe)).sum(dim=dim)
        return entropy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_single_head(
        self,
        attn: torch.Tensor,
        head_index: int,
        layer_index: int,
    ) -> HeadEntropyDistribution:
        """Analyze a single head's attention distribution."""
        seq_len = attn.shape[-1]
        if seq_len < self.min_head_size:
            return HeadEntropyDistribution(
                head_index=head_index,
                layer_index=layer_index,
                max_possible_entropy=math.log(max(seq_len, 1)),
            )

        max_ent = math.log(seq_len)
        per_position_entropy = self.compute_entropy(attn, dim=-1)

        mean_ent = per_position_entropy.mean().item()
        std_ent = per_position_entropy.std().item()
        max_ent_obs = per_position_entropy.max().item()
        min_ent_obs = per_position_entropy.min().item()
        norm_ent = mean_ent / max_ent if max_ent > 0 else 0.0

        dilution = norm_ent > self.dilution_threshold
        collapse = norm_ent < self.collapse_threshold

        return HeadEntropyDistribution(
            head_index=head_index,
            layer_index=layer_index,
            mean_entropy=mean_ent,
            std_entropy=std_ent,
            max_entropy=max_ent_obs,
            min_entropy=min_ent_obs,
            max_possible_entropy=max_ent,
            normalized_entropy=norm_ent,
            dilution_flag=dilution,
            collapse_flag=collapse,
        )

    def _analyze_layer(
        self,
        attn: torch.Tensor,
        layer_index: int,
    ) -> LayerEntropyProfile:
        """Analyze all heads in a layer."""
        # attn: [batch, n_heads, seq_q, seq_k]
        if attn.dim() != 4:
            raise ValueError(
                f"Expected 4D attention tensor [batch, n_heads, seq_q, seq_k], "
                f"got {attn.dim()}D tensor with shape {tuple(attn.shape)}. "
                f"Use analyze_head() for single-head tensors."
            )
        batch, n_heads, seq_q, seq_k = attn.shape
        _ = batch  # unused

        head_dists: list[HeadEntropyDistribution] = []
        collapsed = 0
        diluted = 0

        for h in range(n_heads):
            head_dist = self._analyze_single_head(
                attn[0, h], head_index=h, layer_index=layer_index
            )
            head_dists.append(head_dist)
            if head_dist.collapse_flag:
                collapsed += 1
            elif head_dist.dilution_flag:
                diluted += 1

        healthy = n_heads - collapsed - diluted
        mean_entropies = [h.mean_entropy for h in head_dists]
        layer_mean = sum(mean_entropies) / len(mean_entropies) if mean_entropies else 0.0

        # Std of per-head means
        if len(mean_entropies) > 1:
            m = layer_mean
            layer_std = math.sqrt(
                sum((x - m) ** 2 for x in mean_entropies) / len(mean_entropies)
            )
        else:
            layer_std = 0.0

        return LayerEntropyProfile(
            layer_index=layer_index,
            num_heads=n_heads,
            head_distributions=head_dists,
            layer_mean_entropy=layer_mean,
            layer_std_entropy=layer_std,
            heads_collapsed=collapsed,
            heads_diluted=diluted,
            healthy_heads=healthy,
        )
