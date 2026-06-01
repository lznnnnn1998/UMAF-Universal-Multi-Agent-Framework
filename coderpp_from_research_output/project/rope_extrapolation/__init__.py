"""
rope_extrapolation — RoPE-based position encoding and length extrapolation.

This package implements five position encoding strategies:

- **rope**: Vanilla Rotary Position Embeddings (Su et al., 2021)
- **pi**: Position Interpolation (Chen et al., 2023)
- **ntk**: NTK-aware scaling (bloc97, 2023)
- **yarn**: YaRN — NTK-aware + ramp + temperature (Peng et al., 2023)
- **dpe**: Dynamic Position Encoding — chunked positions + local/global masks

Quickstart:
    from rope_extrapolation import RoPE, PIRoPE, NTKAwareRoPE, YaRNRoPE, DPERoPE

    rope = RoPE(dim=64, max_seq_len=2048)
    q_rot, k_rot = rope(q, k)
"""

from .rope import (
    # Core RoPE — half-dim API
    compute_rope_frequencies,
    apply_rope,
    apply_rotary_pos_emb,
    RoPE,
    # Core RoPE — full-dim API (standard interface)
    compute_freqs,
    precompute_rope_cos_sin,
    rotate_half,
    apply_rotary_emb,
    apply_rotary_emb_single,
    get_rope_embeddings,
    numpy_apply_rotary_emb,
)

from .pi import (
    position_interpolation_scale,
    apply_pi_rope,
    PIRoPE,
)

from .ntk import (
    compute_ntk_base,
    compute_ntk_frequencies,
    apply_ntk_rope,
    frequency_analysis,
    NTKAwareRoPE,
)

from .yarn import (
    compute_ramp,
    compute_yarn_temperature,
    compute_yarn_frequencies,
    apply_yarn_rope,
    YaRNRoPE,
)

from .dpe import (
    compute_dpe_position_ids,
    build_dpe_attention_mask,
    build_dpe_attention_mask_bool,
    apply_dpe_rope,
    DPERoPE,
)

# Extrapolation module — higher-level scaler objects and utilities
from .extrapolation import (
    PIScaler,
    NTKScaler,
    YaRNScaler,
    DPEConfig,
    build_dpe_mask,
    dpe_precompute_cos_sin,
    apply_dpe_rotary,
    get_cos_sin_for_method,
    compare_angles,
)

__all__ = [
    # RoPE core (half-dim)
    "compute_rope_frequencies",
    "apply_rope",
    "apply_rotary_pos_emb",
    "RoPE",
    # RoPE core (full-dim)
    "compute_freqs",
    "precompute_rope_cos_sin",
    "rotate_half",
    "apply_rotary_emb",
    "apply_rotary_emb_single",
    "get_rope_embeddings",
    "numpy_apply_rotary_emb",
    # PI
    "position_interpolation_scale",
    "apply_pi_rope",
    "PIRoPE",
    # NTK
    "compute_ntk_base",
    "compute_ntk_frequencies",
    "apply_ntk_rope",
    "frequency_analysis",
    "NTKAwareRoPE",
    # YaRN
    "compute_ramp",
    "compute_yarn_temperature",
    "compute_yarn_frequencies",
    "apply_yarn_rope",
    "YaRNRoPE",
    # DPE
    "compute_dpe_position_ids",
    "build_dpe_attention_mask",
    "build_dpe_attention_mask_bool",
    "apply_dpe_rope",
    "DPERoPE",
    # Extrapolation
    "PIScaler",
    "NTKScaler",
    "YaRNScaler",
    "DPEConfig",
    "build_dpe_mask",
    "dpe_precompute_cos_sin",
    "apply_dpe_rotary",
    "get_cos_sin_for_method",
    "compare_angles",
]
