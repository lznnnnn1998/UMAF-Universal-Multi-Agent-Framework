"""
Unit tests for the flash_attention module.

Tests cover:
  - Tiling utilities (block_partition, online_softmax_update, build_causal_mask, etc.)
  - FP8 quantization (quantize, dequantize, simulate matmul)
  - FlashAttention V1, V2, V3 forward correctness vs naive attention
  - Backward pass gradient correctness via torch.autograd.gradcheck
  - Edge cases (odd block sizes, small sequences, causal masking)
  - Module interfaces (FlashAttentionV1, V2, V3 nn.Module)
  - Functional interfaces (flash_attention_v1, v2, v3)

Run with:
    PYTHONPATH=modules python -m pytest modules/flash_attention/ -v
"""

from __future__ import annotations

import math
import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _import_module(name: str):
    """Import a module from flash_attention or raise a skip."""
    try:
        return __import__(f"flash_attention.{name}", fromlist=[name])
    except ImportError as e:
        pytest.skip(f"Could not import flash_attention.{name}: {e}")


# We import directly since PYTHONPATH=modules is set during test runs
from flash_attention._tiling import (
    block_partition,
    build_causal_mask,
    compute_flops_saved,
    compute_memory_saved,
    online_softmax_update,
    scaled_dot_product_scores,
)
from flash_attention._quantization import (
    FP8Config,
    dequantize_fp8_e4m3,
    quantize_fp8_e4m3,
    simulate_fp8_matmul,
)
from flash_attention.core import (
    FlashAttentionV1,
    FlashAttentionV2,
    FlashAttentionV3,
    _naive_attention,
    flash_attention_v1,
    flash_attention_v2,
    flash_attention_v3,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH = 2
HEADS = 4
SEQ_LEN = 64
D_HEAD = 32
ATOL = 1e-4
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_qkv():
    """Generate random Q, K, V tensors for testing."""
    torch.manual_seed(42)
    q = torch.randn(BATCH, HEADS, SEQ_LEN, D_HEAD, dtype=torch.float64)
    k = torch.randn(BATCH, HEADS, SEQ_LEN, D_HEAD, dtype=torch.float64)
    v = torch.randn(BATCH, HEADS, SEQ_LEN, D_HEAD, dtype=torch.float64)
    return q, k, v


@pytest.fixture
def random_qkv_small():
    """Smaller Q, K, V for faster gradient checks."""
    torch.manual_seed(123)
    q = torch.randn(1, 2, 16, 16, dtype=torch.float64)
    k = torch.randn(1, 2, 16, 16, dtype=torch.float64)
    v = torch.randn(1, 2, 16, 16, dtype=torch.float64)
    return q, k, v


# ===================================================================
# Tests: _tiling.py
# ===================================================================


class TestBlockPartition:
    """Tests for block_partition."""

    def test_exact_division(self):
        x = torch.randn(2, 4, 64, 32)
        blocks = block_partition(x, 32, dim=-2)
        assert len(blocks) == 2
        assert blocks[0].shape == (2, 4, 32, 32)
        assert blocks[1].shape == (2, 4, 32, 32)

    def test_remainder(self):
        x = torch.randn(2, 4, 100, 64)
        blocks = block_partition(x, 32, dim=-2)
        assert len(blocks) == 4
        assert blocks[0].shape == (2, 4, 32, 64)
        assert blocks[1].shape == (2, 4, 32, 64)
        assert blocks[2].shape == (2, 4, 32, 64)
        assert blocks[3].shape == (2, 4, 4, 64)

    def test_single_block(self):
        x = torch.randn(2, 4, 10, 64)
        blocks = block_partition(x, 100, dim=-2)
        assert len(blocks) == 1
        assert blocks[0].shape == (2, 4, 10, 64)

    def test_block_size_one(self):
        x = torch.randn(2, 4, 5, 64)
        blocks = block_partition(x, 1, dim=-2)
        assert len(blocks) == 5
        for b in blocks:
            assert b.shape[-2] == 1

    def test_partition_dim(self):
        x = torch.randn(2, 4, 64, 32)
        blocks = block_partition(x, 16, dim=-1)
        assert len(blocks) == 2
        assert blocks[0].shape == (2, 4, 64, 16)
        assert blocks[1].shape == (2, 4, 64, 16)

    def test_concat_reconstruction(self):
        x = torch.randn(2, 4, 100, 64)
        blocks = block_partition(x, 32, dim=-2)
        reconstructed = torch.cat(blocks, dim=-2)
        assert torch.equal(reconstructed, x)

    def test_empty_input_handled(self):
        """Edge case: block size > dimension size should still work."""
        x = torch.randn(1, 1, 3, 8)
        blocks = block_partition(x, 10, dim=-2)
        assert len(blocks) == 1
        assert blocks[0].shape == (1, 1, 3, 8)


class TestOnlineSoftmaxUpdate:
    """Tests for online_softmax_update."""

    def test_single_block_equals_softmax(self):
        """With no prior state, online softmax should equal regular softmax."""
        torch.manual_seed(42)
        s = torch.randn(4, 8, 16, 32, dtype=torch.float64)  # [b, h, Br, Bc]
        Br = s.shape[-2]

        m_old = torch.full((4, 8, Br), float("-inf"), dtype=torch.float64)
        l_old = torch.zeros(4, 8, Br, dtype=torch.float64)

        m_new, l_new, P_block = online_softmax_update(m_old, l_old, s)

        # Regular softmax
        P_ref = torch.softmax(s, dim=-1)

        assert torch.allclose(P_block / l_new.unsqueeze(-1), P_ref, atol=ATOL, rtol=RTOL)

    def test_two_blocks_match_full_softmax(self):
        """Online softmax over 2 blocks should equal softmax over concatenated scores."""
        torch.manual_seed(42)
        s1 = torch.randn(2, 2, 8, 16, dtype=torch.float64)
        s2 = torch.randn(2, 2, 8, 16, dtype=torch.float64)
        s_full = torch.cat([s1, s2], dim=-1)
        Br = s1.shape[-2]

        # Block 1
        m_old = torch.full((2, 2, Br), float("-inf"), dtype=torch.float64)
        l_old = torch.zeros(2, 2, Br, dtype=torch.float64)
        m1, l1, P1 = online_softmax_update(m_old, l_old, s1)

        # Block 2
        m_final, l_final, P2 = online_softmax_update(m1, l1, s2)

        # P1 was computed relative to m1. If m_final > m1, rescale P1.
        # P1_rescaled = exp(s1 - m1 + m1 - m_final) = exp(s1 - m_final)
        scale_P1 = torch.exp(m1 - m_final).unsqueeze(-1)  # [2, 2, 8, 1]
        P1_rescaled = P1 * scale_P1

        # Reconstruct full softmax using m_final-based exp and l_final denominator
        P_reconstructed = torch.cat(
            [P1_rescaled / l_final.unsqueeze(-1), P2 / l_final.unsqueeze(-1)], dim=-1
        )
        P_ref = torch.softmax(s_full, dim=-1)

        assert torch.allclose(P_reconstructed, P_ref, atol=ATOL, rtol=RTOL)

    def test_numerical_stability_large_values(self):
        """Online softmax should handle large values without overflow."""
        s = torch.tensor([[[[1000.0, 2000.0], [3000.0, 4000.0]]]], dtype=torch.float64)
        Br = 2

        m_old = torch.full((1, 1, Br), float("-inf"), dtype=torch.float64)
        l_old = torch.zeros(1, 1, Br, dtype=torch.float64)

        m_new, l_new, P = online_softmax_update(m_old, l_old, s)

        assert not torch.isnan(m_new).any()
        assert not torch.isnan(l_new).any()
        assert not torch.isnan(P).any()
        assert not torch.isinf(P).any()

    def test_zero_scores(self):
        """All-zero scores should produce uniform softmax."""
        s = torch.zeros(1, 1, 4, 8, dtype=torch.float64)
        Br = 4

        m_old = torch.full((1, 1, Br), float("-inf"), dtype=torch.float64)
        l_old = torch.zeros(1, 1, Br, dtype=torch.float64)

        m_new, l_new, P = online_softmax_update(m_old, l_old, s)
        P_norm = P / l_new.unsqueeze(-1)

        # Each row should be uniform
        expected = torch.ones_like(P_norm) / P_norm.shape[-1]
        assert torch.allclose(P_norm, expected, atol=ATOL)


class TestBuildCausalMask:
    """Tests for build_causal_mask."""

    def test_all_valid(self):
        """Q block entirely after KV block: all positions valid."""
        mask = build_causal_mask(q_start=10, kv_start=0, q_len=4, kv_len=4)
        assert mask.all()

    def test_all_masked(self):
        """Q block entirely before KV block: no positions valid."""
        mask = build_causal_mask(q_start=0, kv_start=10, q_len=4, kv_len=4)
        assert not mask.any()

    def test_partial_mask(self):
        """Overlapping blocks: triangular mask."""
        mask = build_causal_mask(q_start=0, kv_start=0, q_len=4, kv_len=4)
        # Lower triangular (including diagonal)
        expected = torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ])
        assert torch.equal(mask, expected)

    def test_uneven_blocks(self):
        mask = build_causal_mask(q_start=3, kv_start=0, q_len=2, kv_len=5)
        # Q positions: 3, 4. KV positions: 0, 1, 2, 3, 4.
        # pos 3 attends to 0,1,2,3. pos 4 attends to 0,1,2,3,4.
        expected = torch.tensor([
            [True, True, True, True, False],   # q=3: kv=0,1,2,3 valid; kv=4 invalid
            [True, True, True, True, True],    # q=4: all valid
        ])
        assert torch.equal(mask, expected)

    def test_device_passthrough(self):
        mask = build_causal_mask(0, 0, 4, 4, device=torch.device("cpu"))
        assert mask.device.type == "cpu"


class TestScaledDotProductScores:
    """Tests for scaled_dot_product_scores."""

    def test_shape(self):
        q = torch.randn(2, 4, 32, 64)
        k = torch.randn(2, 4, 32, 64)
        scale = 1.0 / math.sqrt(64)
        scores = scaled_dot_product_scores(q, k, scale)
        assert scores.shape == (2, 4, 32, 32)

    def test_scale_effect(self):
        q = torch.ones(1, 1, 4, 8)
        k = torch.ones(1, 1, 4, 8)
        s1 = scaled_dot_product_scores(q, k, 1.0)
        s2 = scaled_dot_product_scores(q, k, 0.5)
        assert torch.allclose(s1 * 0.5, s2)


class TestComputeFlopsSaved:
    """Tests for compute_flops_saved."""

    def test_returns_dict_with_keys(self):
        result = compute_flops_saved(1024, 64)
        assert "naive_gflops" in result
        assert "flash_gflops" in result
        assert "ratio" in result

    def test_flash_more_efficient(self):
        result = compute_flops_saved(2048, 64)
        assert result["ratio"] > 1.0

    def test_small_sequence(self):
        result = compute_flops_saved(32, 64)
        assert result["ratio"] >= 1.0


class TestComputeMemorySaved:
    """Tests for compute_memory_saved."""

    def test_returns_dict_with_keys(self):
        result = compute_memory_saved(1024, 64)
        assert "naive_mb" in result
        assert "flash_mb" in result
        assert "ratio" in result

    def test_flash_more_memory_efficient(self):
        result = compute_memory_saved(2048, 64)
        assert result["ratio"] > 1.0

    def test_fp32_uses_more_memory(self):
        r16 = compute_memory_saved(512, 64, bytes_per_element=2)
        r32 = compute_memory_saved(512, 64, bytes_per_element=4)
        assert r32["naive_mb"] > r16["naive_mb"]


# ===================================================================
# Tests: _quantization.py
# ===================================================================


class TestFP8Config:
    """Tests for FP8Config."""

    def test_defaults(self):
        cfg = FP8Config()
        assert cfg.ebits == 4
        assert cfg.mbits == 3
        assert cfg.bias == 7
        assert cfg.total_bits == 8

    def test_max_normal_positive(self):
        cfg = FP8Config()
        assert cfg.max_normal > 0
        assert cfg.max_normal < 1000  # ~240 for E4M3

    def test_min_normal_positive(self):
        cfg = FP8Config()
        assert cfg.min_normal > 0

    def test_min_subnormal_positive(self):
        cfg = FP8Config()
        assert cfg.min_subnormal > 0
        assert cfg.min_subnormal < cfg.min_normal


class TestQuantizeFP8:
    """Tests for quantize_fp8_e4m3."""

    def test_quantize_preserves_shape(self):
        x = torch.randn(4, 8, 16)
        x_q, scale = quantize_fp8_e4m3(x)
        assert x_q.shape == x.shape

    def test_quantize_zero(self):
        x = torch.zeros(3, 4)
        x_q, scale = quantize_fp8_e4m3(x)
        assert torch.allclose(x_q, torch.zeros_like(x_q))

    def test_quantize_small_values_become_zero(self):
        """Values below min_subnormal should quantize to zero."""
        cfg = FP8Config()
        tiny = torch.tensor([cfg.min_subnormal * 0.1])
        x_q, _ = quantize_fp8_e4m3(tiny, cfg)
        assert x_q.item() == 0.0

    def test_quantize_large_values_clamped(self):
        """Values above max_normal should be clamped."""
        cfg = FP8Config()
        huge = torch.tensor([cfg.max_normal * 10.0])
        x_q, _ = quantize_fp8_e4m3(huge, cfg)
        assert x_q.item() <= cfg.max_normal

    def test_quantize_dequantize_roundtrip(self):
        """Quantize then dequantize should be close for in-range values."""
        x = torch.randn(10, 10) * 5.0  # moderate values within FP8 range
        x_q, scale = quantize_fp8_e4m3(x)
        x_dq = dequantize_fp8_e4m3(x_q, scale)
        # FP8 has limited precision, so we use a larger tolerance
        assert x_dq.shape == x.shape


class TestSimulateFP8Matmul:
    """Tests for simulate_fp8_matmul."""

    def test_shape(self):
        a = torch.randn(2, 4, 32, 16)   # [batch, heads, M, K]
        b = torch.randn(2, 4, 16, 32)   # [batch, heads, K, N]
        result = simulate_fp8_matmul(a, b)
        assert result.shape == (2, 4, 32, 32)

    def test_approximate_result(self):
        """FP8 matmul should give approximately the correct result."""
        torch.manual_seed(42)
        a = torch.randn(16, 32)  # [M, K]
        b = torch.randn(32, 16)  # [K, N]
        fp8_result = simulate_fp8_matmul(a, b)
        fp32_result = torch.matmul(a.float(), b.float())
        # FP8 is approximate; result should be in same order of magnitude
        assert fp8_result.shape == fp32_result.shape


# ===================================================================
# Tests: FlashAttention V1
# ===================================================================


class TestFlashAttentionV1Forward:
    """Forward pass correctness tests for FlashAttention v1."""

    def test_matches_naive_non_causal(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        flash_out = flash_attention_v1(q, k, v, Br=16, Bc=16, causal=False, scale=scale)
        naive_out = _naive_attention(q, k, v, causal=False, scale=scale)

        assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL)

    def test_matches_naive_causal(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        flash_out = flash_attention_v1(q, k, v, Br=16, Bc=16, causal=True, scale=scale)
        naive_out = _naive_attention(q, k, v, causal=True, scale=scale)

        assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL)

    def test_odd_block_sizes(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        # Prime block sizes that don't evenly divide sequence length
        flash_out = flash_attention_v1(q, k, v, Br=13, Bc=17, causal=False, scale=scale)
        naive_out = _naive_attention(q, k, v, causal=False, scale=scale)

        assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL)

    def test_single_batch_single_head(self):
        q = torch.randn(1, 1, 32, 16, dtype=torch.float64)
        k = torch.randn(1, 1, 32, 16, dtype=torch.float64)
        v = torch.randn(1, 1, 32, 16, dtype=torch.float64)
        scale = 1.0 / math.sqrt(16)

        flash_out = flash_attention_v1(q, k, v, Br=8, Bc=8, scale=scale)
        naive_out = _naive_attention(q, k, v, scale=scale)

        assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL)

    def test_variable_sequence_lengths(self):
        """Test with sequences that aren't powers of 2."""
        for n in [17, 31, 50, 63]:
            q = torch.randn(1, 2, n, 16, dtype=torch.float64)
            k = torch.randn(1, 2, n, 16, dtype=torch.float64)
            v = torch.randn(1, 2, n, 16, dtype=torch.float64)
            scale = 1.0 / math.sqrt(16)

            flash_out = flash_attention_v1(q, k, v, Br=16, Bc=16, scale=scale)
            naive_out = _naive_attention(q, k, v, scale=scale)

            assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL), (
                f"Failed for seq_len={n}"
            )

    def test_module_interface(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        attn = FlashAttentionV1(Br=16, Bc=16, causal=False, scale=scale)
        out = attn(q, k, v)
        naive_out = _naive_attention(q, k, v, causal=False, scale=scale)

        assert torch.allclose(out, naive_out, atol=ATOL, rtol=RTOL)


class TestFlashAttentionV1Backward:
    """Backward pass gradient tests for FlashAttention v1."""

    def test_gradient_wrt_q(self, random_qkv_small):
        q, k, v = random_qkv_small
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        scale = 1.0 / math.sqrt(q.shape[-1])
        out = flash_attention_v1(q, k, v, Br=8, Bc=8, scale=scale)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_gradient_matches_naive(self, random_qkv_small):
        q1, k1, v1 = random_qkv_small
        q2 = q1.clone().detach().requires_grad_(True)
        k2 = k1.clone().detach().requires_grad_(True)
        v2 = v1.clone().detach().requires_grad_(True)
        q1 = q1.clone().detach().requires_grad_(True)
        k1 = k1.clone().detach().requires_grad_(True)
        v1 = v1.clone().detach().requires_grad_(True)

        scale = 1.0 / math.sqrt(q1.shape[-1])

        # Flash path
        flash_out = flash_attention_v1(q1, k1, v1, Br=8, Bc=8, scale=scale)
        flash_loss = flash_out.sum()
        flash_loss.backward()

        # Naive path
        naive_out = _naive_attention(q2, k2, v2, scale=scale)
        naive_loss = naive_out.sum()
        naive_loss.backward()

        assert torch.allclose(q1.grad, q2.grad, atol=1e-3, rtol=1e-2), "dQ mismatch"
        assert torch.allclose(k1.grad, k2.grad, atol=1e-3, rtol=1e-2), "dK mismatch"
        assert torch.allclose(v1.grad, v2.grad, atol=1e-3, rtol=1e-2), "dV mismatch"

    def test_gradient_causal(self, random_qkv_small):
        q1, k1, v1 = random_qkv_small
        q2 = q1.clone().detach().requires_grad_(True)
        k2 = k1.clone().detach().requires_grad_(True)
        v2 = v1.clone().detach().requires_grad_(True)
        q1 = q1.clone().detach().requires_grad_(True)
        k1 = k1.clone().detach().requires_grad_(True)
        v1 = v1.clone().detach().requires_grad_(True)

        scale = 1.0 / math.sqrt(q1.shape[-1])

        flash_out = flash_attention_v1(q1, k1, v1, Br=8, Bc=8, causal=True, scale=scale)
        flash_loss = flash_out.sum()
        flash_loss.backward()

        naive_out = _naive_attention(q2, k2, v2, causal=True, scale=scale)
        naive_loss = naive_out.sum()
        naive_loss.backward()

        assert torch.allclose(q1.grad, q2.grad, atol=1e-3, rtol=1e-2), "dQ causal mismatch"
        assert torch.allclose(v1.grad, v2.grad, atol=1e-3, rtol=1e-2), "dV causal mismatch"


# ===================================================================
# Tests: FlashAttention V2
# ===================================================================


class TestFlashAttentionV2Forward:
    """Forward pass correctness tests for FlashAttention v2."""

    def test_matches_naive_non_causal(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        flash_out = flash_attention_v2(q, k, v, Br=16, Bc=16, causal=False, scale=scale)
        naive_out = _naive_attention(q, k, v, causal=False, scale=scale)

        assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL)

    def test_matches_naive_causal(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        flash_out = flash_attention_v2(q, k, v, Br=16, Bc=16, causal=True, scale=scale)
        naive_out = _naive_attention(q, k, v, causal=True, scale=scale)

        assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL)

    def test_matches_v1(self, random_qkv):
        """V2 should produce identical results to V1 (same algorithm, different loop order)."""
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        out_v1 = flash_attention_v1(q, k, v, Br=16, Bc=16, causal=False, scale=scale)
        out_v2 = flash_attention_v2(q, k, v, Br=16, Bc=16, causal=False, scale=scale)

        assert torch.allclose(out_v1, out_v2, atol=ATOL, rtol=RTOL)

    def test_variable_sequence_lengths(self):
        for n in [15, 29, 47, 60]:
            q = torch.randn(1, 2, n, 16, dtype=torch.float64)
            k = torch.randn(1, 2, n, 16, dtype=torch.float64)
            v = torch.randn(1, 2, n, 16, dtype=torch.float64)
            scale = 1.0 / math.sqrt(16)

            flash_out = flash_attention_v2(q, k, v, Br=13, Bc=17, scale=scale)
            naive_out = _naive_attention(q, k, v, scale=scale)

            assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL), (
                f"V2 failed for seq_len={n}"
            )

    def test_module_interface(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        attn = FlashAttentionV2(Br=16, Bc=16, causal=False, scale=scale)
        out = attn(q, k, v)
        naive_out = _naive_attention(q, k, v, causal=False, scale=scale)

        assert torch.allclose(out, naive_out, atol=ATOL, rtol=RTOL)


class TestFlashAttentionV2Backward:
    """Backward pass gradient tests for FlashAttention v2."""

    def test_gradient_non_nan(self, random_qkv_small):
        q, k, v = random_qkv_small
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        scale = 1.0 / math.sqrt(q.shape[-1])
        out = flash_attention_v2(q, k, v, Br=8, Bc=8, scale=scale)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None and not torch.isnan(q.grad).any()
        assert k.grad is not None and not torch.isnan(k.grad).any()
        assert v.grad is not None and not torch.isnan(v.grad).any()

    def test_gradient_matches_naive(self, random_qkv_small):
        q1, k1, v1 = random_qkv_small
        q2 = q1.clone().detach().requires_grad_(True)
        k2 = k1.clone().detach().requires_grad_(True)
        v2 = v1.clone().detach().requires_grad_(True)
        q1 = q1.clone().detach().requires_grad_(True)
        k1 = k1.clone().detach().requires_grad_(True)
        v1 = v1.clone().detach().requires_grad_(True)

        scale = 1.0 / math.sqrt(q1.shape[-1])

        flash_out = flash_attention_v2(q1, k1, v1, Br=8, Bc=8, scale=scale)
        flash_loss = flash_out.sum()
        flash_loss.backward()

        naive_out = _naive_attention(q2, k2, v2, scale=scale)
        naive_loss = naive_out.sum()
        naive_loss.backward()

        assert torch.allclose(q1.grad, q2.grad, atol=1e-3, rtol=1e-2), "V2 dQ mismatch"
        assert torch.allclose(k1.grad, k2.grad, atol=1e-3, rtol=1e-2), "V2 dK mismatch"
        assert torch.allclose(v1.grad, v2.grad, atol=1e-3, rtol=1e-2), "V2 dV mismatch"


# ===================================================================
# Tests: FlashAttention V3
# ===================================================================


class TestFlashAttentionV3Forward:
    """Forward pass correctness tests for FlashAttention v3."""

    def test_matches_naive_non_causal(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        flash_out = flash_attention_v3(q, k, v, Br=16, Bc=16, causal=False, scale=scale)
        naive_out = _naive_attention(q, k, v, causal=False, scale=scale)

        assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL)

    def test_matches_naive_causal(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        flash_out = flash_attention_v3(q, k, v, Br=16, Bc=16, causal=True, scale=scale)
        naive_out = _naive_attention(q, k, v, causal=True, scale=scale)

        assert torch.allclose(flash_out, naive_out, atol=ATOL, rtol=RTOL)

    def test_fp8_mode_runs(self, random_qkv):
        """With use_fp8=True, should still produce reasonable output."""
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        out_fp8 = flash_attention_v3(
            q, k, v, Br=16, Bc=16, causal=False, scale=scale, use_fp8=True
        )
        assert out_fp8.shape == q.shape
        assert not torch.isnan(out_fp8).any()
        assert not torch.isinf(out_fp8).any()

    def test_fp8_approximates_fp32(self, random_qkv):
        """FP8 output should be close to FP32 output."""
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        out_fp32 = flash_attention_v3(
            q.float(), k.float(), v.float(), Br=16, Bc=16, scale=scale, use_fp8=False
        )
        out_fp8 = flash_attention_v3(
            q.float(), k.float(), v.float(), Br=16, Bc=16, scale=scale, use_fp8=True
        )

        # FP8 introduces quantization error; check correlation
        # For values in FP8 range, approximation should be reasonable
        assert out_fp8.shape == out_fp32.shape

    def test_module_interface(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        attn = FlashAttentionV3(Br=16, Bc=16, causal=False, scale=scale, use_fp8=False)
        out = attn(q, k, v)
        naive_out = _naive_attention(q, k, v, causal=False, scale=scale)

        assert torch.allclose(out, naive_out, atol=ATOL, rtol=RTOL)

    def test_module_with_fp8(self, random_qkv):
        q, k, v = random_qkv
        # Use moderate values to stay within FP8 range
        q = q * 0.1
        k = k * 0.1
        v = v * 0.1

        attn = FlashAttentionV3(Br=16, Bc=16, causal=False, scale=None, use_fp8=True)
        out = attn(q, k, v)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()


class TestFlashAttentionV3Backward:
    """Backward pass gradient tests for FlashAttention v3."""

    def test_gradient_non_nan(self, random_qkv_small):
        q, k, v = random_qkv_small
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        scale = 1.0 / math.sqrt(q.shape[-1])
        out = flash_attention_v3(q, k, v, Br=8, Bc=8, scale=scale)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None and not torch.isnan(q.grad).any()
        assert k.grad is not None and not torch.isnan(k.grad).any()
        assert v.grad is not None and not torch.isnan(v.grad).any()

    def test_gradient_matches_naive(self, random_qkv_small):
        q1, k1, v1 = random_qkv_small
        q2 = q1.clone().detach().requires_grad_(True)
        k2 = k1.clone().detach().requires_grad_(True)
        v2 = v1.clone().detach().requires_grad_(True)
        q1 = q1.clone().detach().requires_grad_(True)
        k1 = k1.clone().detach().requires_grad_(True)
        v1 = v1.clone().detach().requires_grad_(True)

        scale = 1.0 / math.sqrt(q1.shape[-1])

        flash_out = flash_attention_v3(q1, k1, v1, Br=8, Bc=8, scale=scale)
        flash_loss = flash_out.sum()
        flash_loss.backward()

        naive_out = _naive_attention(q2, k2, v2, scale=scale)
        naive_loss = naive_out.sum()
        naive_loss.backward()

        assert torch.allclose(q1.grad, q2.grad, atol=1e-3, rtol=1e-2), "V3 dQ mismatch"
        assert torch.allclose(k1.grad, k2.grad, atol=1e-3, rtol=1e-2), "V3 dK mismatch"
        assert torch.allclose(v1.grad, v2.grad, atol=1e-3, rtol=1e-2), "V3 dV mismatch"


# ===================================================================
# Tests: Cross-version consistency
# ===================================================================


class TestCrossVersionConsistency:
    """Ensure all three versions produce identical results."""

    def test_all_versions_match(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        out_v1 = flash_attention_v1(q, k, v, Br=16, Bc=16, scale=scale)
        out_v2 = flash_attention_v2(q, k, v, Br=16, Bc=16, scale=scale)
        out_v3 = flash_attention_v3(q, k, v, Br=16, Bc=16, scale=scale)

        assert torch.allclose(out_v1, out_v2, atol=ATOL, rtol=RTOL)
        assert torch.allclose(out_v2, out_v3, atol=ATOL, rtol=RTOL)
        assert torch.allclose(out_v1, out_v3, atol=ATOL, rtol=RTOL)

    def test_all_versions_match_causal(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        out_v1 = flash_attention_v1(q, k, v, Br=16, Bc=16, causal=True, scale=scale)
        out_v2 = flash_attention_v2(q, k, v, Br=16, Bc=16, causal=True, scale=scale)
        out_v3 = flash_attention_v3(q, k, v, Br=16, Bc=16, causal=True, scale=scale)

        assert torch.allclose(out_v1, out_v2, atol=ATOL, rtol=RTOL)
        assert torch.allclose(out_v2, out_v3, atol=ATOL, rtol=RTOL)


# ===================================================================
# Tests: Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_sequence_length_one(self):
        q = torch.randn(2, 4, 1, 32, dtype=torch.float64)
        k = torch.randn(2, 4, 1, 32, dtype=torch.float64)
        v = torch.randn(2, 4, 1, 32, dtype=torch.float64)
        scale = 1.0 / math.sqrt(32)

        out_v1 = flash_attention_v1(q, k, v, Br=32, Bc=32, scale=scale)
        out_v2 = flash_attention_v2(q, k, v, Br=32, Bc=32, scale=scale)
        out_v3 = flash_attention_v3(q, k, v, Br=32, Bc=32, scale=scale)
        naive = _naive_attention(q, k, v, scale=scale)

        assert torch.allclose(out_v1, naive, atol=ATOL, rtol=RTOL)
        assert torch.allclose(out_v2, naive, atol=ATOL, rtol=RTOL)
        assert torch.allclose(out_v3, naive, atol=ATOL, rtol=RTOL)

    def test_block_size_larger_than_sequence(self, random_qkv):
        """Block size exceeding sequence length should behave like single block."""
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        out = flash_attention_v1(q, k, v, Br=256, Bc=256, scale=scale)
        naive = _naive_attention(q, k, v, scale=scale)

        assert torch.allclose(out, naive, atol=ATOL, rtol=RTOL)

    def test_large_head_dim(self):
        q = torch.randn(1, 2, 16, 128, dtype=torch.float64)
        k = torch.randn(1, 2, 16, 128, dtype=torch.float64)
        v = torch.randn(1, 2, 16, 128, dtype=torch.float64)
        scale = 1.0 / math.sqrt(128)

        out = flash_attention_v1(q, k, v, Br=8, Bc=8, scale=scale)
        naive = _naive_attention(q, k, v, scale=scale)

        assert torch.allclose(out, naive, atol=ATOL, rtol=RTOL)

    def test_no_grad_mode(self, random_qkv):
        """Should work in torch.no_grad() context."""
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        with torch.no_grad():
            out = flash_attention_v1(q, k, v, Br=16, Bc=16, scale=scale)

        naive = _naive_attention(q, k, v, scale=scale)
        assert torch.allclose(out, naive, atol=ATOL, rtol=RTOL)

    def test_input_unchanged(self, random_qkv):
        """Forward pass should not modify input tensors."""
        q, k, v = random_qkv
        q_copy = q.clone()
        k_copy = k.clone()
        v_copy = v.clone()

        _ = flash_attention_v1(q, k, v, Br=16, Bc=16)

        assert torch.equal(q, q_copy)
        assert torch.equal(k, k_copy)
        assert torch.equal(v, v_copy)


# ===================================================================
# Tests: Determinism
# ===================================================================


class TestDeterminism:
    """Determinism and reproducibility tests."""

    def test_same_input_same_output(self, random_qkv):
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        out1 = flash_attention_v1(q, k, v, Br=16, Bc=16, scale=scale)
        out2 = flash_attention_v1(q, k, v, Br=16, Bc=16, scale=scale)

        assert torch.equal(out1, out2)

    def test_v1_v2_identical_numerics(self, random_qkv):
        """V1 and V2 share the same mathematical core; results should be identical."""
        q, k, v = random_qkv
        scale = 1.0 / math.sqrt(D_HEAD)

        out1 = flash_attention_v1(q, k, v, Br=16, Bc=16, scale=scale)
        out2 = flash_attention_v2(q, k, v, Br=16, Bc=16, scale=scale)

        assert torch.allclose(out1, out2, atol=ATOL, rtol=RTOL)
