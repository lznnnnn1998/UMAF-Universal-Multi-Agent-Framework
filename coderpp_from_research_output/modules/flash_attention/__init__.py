"""
FlashAttention module — implements the FlashAttention algorithm lineage (v1–v3)
in pure PyTorch.

Core algorithms:
  - Block-wise tiling with SRAM-aware partitioning (Br x d, Bc x d blocks)
  - Online softmax with exact running statistics recurrence
  - Recomputation-based backward pass (O(N) memory)
  - v2: Q-outer/KV-inner loop order swap, Split-Q, delayed normalization
  - v3: Warp specialization, ping-pong scheduling, FP8 quantization

Usage:
    from flash_attention import flash_attention_v1, FlashAttentionV1
    attn = FlashAttentionV1(Br=32, Bc=32, causal=True)
    output = attn(q, k, v)
"""

from .core import (
    FlashAttentionV1,
    FlashAttentionV2,
    FlashAttentionV3,
    flash_attention_v1,
    flash_attention_v2,
    flash_attention_v3,
)
from ._tiling import (
    block_partition,
    online_softmax_update,
    build_causal_mask,
    scaled_dot_product_scores,
    compute_flops_saved,
    compute_memory_saved,
)
from ._quantization import (
    FP8Config,
    quantize_fp8_e4m3,
    dequantize_fp8_e4m3,
    simulate_fp8_matmul,
)

__all__ = [
    # Core attention modules
    "FlashAttentionV1",
    "FlashAttentionV2",
    "FlashAttentionV3",
    "flash_attention_v1",
    "flash_attention_v2",
    "flash_attention_v3",
    # Tiling utilities
    "block_partition",
    "online_softmax_update",
    "build_causal_mask",
    "scaled_dot_product_scores",
    "compute_flops_saved",
    "compute_memory_saved",
    # Quantization
    "FP8Config",
    "quantize_fp8_e4m3",
    "dequantize_fp8_e4m3",
    "simulate_fp8_matmul",
]
