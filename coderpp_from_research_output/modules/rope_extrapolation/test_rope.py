"""Tests for rope_extrapolation module."""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest
import torch

# Ensure the modules directory is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rope_extrapolation import (
    # Core RoPE
    compute_freqs,
    precompute_rope_cos_sin,
    apply_rotary_emb,
    rotate_half,
    apply_rotary_emb_single,
    get_rope_embeddings,
    numpy_apply_rotary_emb,
    # Extrapolation
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


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def dim() -> int:
    return 64


@pytest.fixture
def base() -> float:
    return 10000.0


# =========================================================================
# Core RoPE tests
# =========================================================================

class TestComputeFreqs:
    """Tests for compute_freqs()."""

    def test_basic(self, dim, base):
        freqs = compute_freqs(dim, base)
        assert freqs.shape == (dim // 2,)
        assert freqs.dtype == torch.float32
        # Frequencies should be decreasing
        assert freqs[0] > freqs[-1]

    def test_decreasing(self):
        """θ_i should decrease monotonically with i."""
        freqs = compute_freqs(64, 10000.0)
        for i in range(len(freqs) - 1):
            assert freqs[i] > freqs[i + 1], f"freqs[{i}] > freqs[{i+1}] expected"

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            compute_freqs(63, 10000.0)

    def test_custom_base(self):
        """Higher base = lower frequencies (base in denominator)."""
        freqs_10k = compute_freqs(64, 10000.0)
        freqs_100k = compute_freqs(64, 100000.0)
        # freqs[0] = base^0 = 1.0 for all bases, so check later indices
        # 1/100000^x < 1/10000^x for x>0, so freqs_100k should be smaller
        assert freqs_100k[1] < freqs_10k[1]

    def test_formula_exact(self):
        """Verify θ_i = base^{-2i/d} exactly for a few values."""
        dim = 64
        base = 10000.0
        freqs = compute_freqs(dim, base)
        for i_idx, i_val in enumerate(range(0, dim, 2)):
            expected = 1.0 / (base ** (i_val / dim))
            assert torch.allclose(freqs[i_idx], torch.tensor(expected, dtype=torch.float32))

    def test_device(self):
        if torch.cuda.is_available():
            freqs = compute_freqs(64, 10000.0, device="cuda")
            assert freqs.device.type == "cuda"

    def test_dtype(self):
        freqs = compute_freqs(64, 10000.0, dtype=torch.float64)
        assert freqs.dtype == torch.float64


class TestRotateHalf:
    """Tests for rotate_half()."""

    def test_1d(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = rotate_half(x)
        expected = torch.tensor([-2.0, 1.0, -4.0, 3.0])
        assert torch.allclose(result, expected)

    def test_2d(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0]])
        result = rotate_half(x)
        expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0],
                                 [-6.0, 5.0, -8.0, 7.0]])
        assert torch.allclose(result, expected)

    def test_3d_batched(self):
        x = torch.randn(2, 4, 8)
        result = rotate_half(x)
        assert result.shape == x.shape
        # Manually verify first element of each pair
        for b in range(2):
            for s in range(4):
                assert result[b, s, 0] == pytest.approx(-x[b, s, 1].item())
                assert result[b, s, 1] == pytest.approx(x[b, s, 0].item())
                assert result[b, s, 2] == pytest.approx(-x[b, s, 3].item())
                assert result[b, s, 3] == pytest.approx(x[b, s, 2].item())

    def test_identity_like_property(self):
        """rotate_half(rotate_half(x)) = -x"""
        x = torch.randn(4, 8)
        result = rotate_half(rotate_half(x))
        assert torch.allclose(result, -x)

    def test_preserves_norm(self):
        """rotate_half should preserve L2 norm."""
        x = torch.randn(3, 8, 16)
        norm_before = torch.norm(x, dim=-1)
        norm_after = torch.norm(rotate_half(x), dim=-1)
        assert torch.allclose(norm_before, norm_after)


class TestPrecomputeRopeCosSin:
    """Tests for precompute_rope_cos_sin()."""

    def test_shapes(self):
        cos, sin = precompute_rope_cos_sin(128, 64)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

    def test_cos_range(self):
        """Cosine values should be in [-1, 1]."""
        cos, _ = precompute_rope_cos_sin(128, 64)
        assert (cos >= -1.0).all() and (cos <= 1.0).all()

    def test_sin_range(self):
        """Sine values should be in [-1, 1]."""
        _, sin = precompute_rope_cos_sin(128, 64)
        assert (sin >= -1.0).all() and (sin <= 1.0).all()

    def test_cos_even_odd_equal(self):
        """For each pair (2i, 2i+1), cos values should be equal."""
        cos, _ = precompute_rope_cos_sin(128, 64)
        for pos in range(128):
            for i in range(0, 64, 2):
                assert cos[pos, i] == pytest.approx(cos[pos, i + 1]), \
                    f"cos[{pos},{i}] != cos[{pos},{i+1}]"

    def test_sin_even_odd_equal(self):
        """For each pair (2i, 2i+1), sin values should be equal (no sign alternation)."""
        _, sin = precompute_rope_cos_sin(128, 64)
        for pos in range(128):
            for i in range(0, 64, 2):
                assert sin[pos, i] == pytest.approx(sin[pos, i + 1]), \
                    f"sin[{pos},{i}] != sin[{pos},{i+1}]"

    def test_position_zero(self):
        """At position 0, cos = 1 and sin = 0 for all dims."""
        cos, sin = precompute_rope_cos_sin(128, 64)
        assert torch.allclose(cos[0], torch.ones(64))
        assert torch.allclose(sin[0], torch.zeros(64))

    def test_orthogonality(self):
        """At each position, cos^2 + sin^2 should be ~1 for each dim."""
        cos, sin = precompute_rope_cos_sin(128, 64)
        squares = cos ** 2 + sin ** 2
        assert torch.allclose(squares, torch.ones_like(squares), atol=1e-6)

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            precompute_rope_cos_sin(128, 63)


class TestApplyRotaryEmb:
    """Tests for apply_rotary_emb()."""

    def test_basic(self, dim):
        seq_len = 32
        cos, sin = precompute_rope_cos_sin(seq_len * 2, dim)
        x = torch.randn(2, 4, seq_len, dim)  # (batch, heads, seq, dim)
        result = apply_rotary_emb(x, cos, sin)
        assert result.shape == x.shape
        # Should not be equal to input (position info added)
        assert not torch.allclose(result, x)

    def test_with_offset(self, dim):
        seq_len = 32
        cos, sin = precompute_rope_cos_sin(seq_len * 4, dim)
        x = torch.randn(2, 4, seq_len, dim)
        # Apply with offset should give different result than without offset
        result_offset = apply_rotary_emb(x, cos, sin, offset=seq_len)
        result_base = apply_rotary_emb(x, cos, sin, offset=0)
        # Different positions → different rotations
        assert not torch.allclose(result_offset, result_base)

    def test_preserves_norm_per_token(self, dim):
        """RoPE should preserve the L2 norm of each token (rotation is isometric)."""
        seq_len = 32
        cos, sin = precompute_rope_cos_sin(seq_len, dim)
        x = torch.randn(1, 1, seq_len, dim)
        result = apply_rotary_emb(x, cos, sin)
        norms_before = torch.norm(x, dim=-1)
        norms_after = torch.norm(result, dim=-1)
        assert torch.allclose(norms_before, norms_after, atol=1e-5)

    def test_odd_dim_raises(self):
        cos, sin = precompute_rope_cos_sin(16, 16)
        x = torch.randn(1, 1, 4, 15)  # odd dim!
        with pytest.raises(ValueError, match="even"):
            apply_rotary_emb(x, cos, sin)

    def test_single_token(self):
        """Apply to a single token vector."""
        cos, sin = precompute_rope_cos_sin(4, 8)
        x = torch.randn(8)  # single token
        result = apply_rotary_emb(x.unsqueeze(0), cos, sin)
        assert result.shape == (1, 8)

    def test_3d_input(self):
        """(batch, seq, dim) without heads."""
        cos, sin = precompute_rope_cos_sin(16, 32)
        x = torch.randn(3, 8, 32)
        result = apply_rotary_emb(x, cos, sin)
        assert result.shape == (3, 8, 32)

    def test_matches_numpy_reference(self):
        """Compare with exact NumPy element-wise implementation."""
        dim = 16
        seq_len = 8
        base = 10000.0

        cos, sin = precompute_rope_cos_sin(seq_len, dim, base=base)
        x = torch.randn(seq_len, dim)
        result_torch = apply_rotary_emb(x.unsqueeze(0), cos, sin).squeeze(0)

        # NumPy reference (element-wise rotation per pair)
        result_np = np.zeros((seq_len, dim))
        for p in range(seq_len):
            result_np[p] = numpy_apply_rotary_emb(x[p].numpy(), p, base)

        assert np.allclose(result_torch.numpy(), result_np, atol=1e-5)


class TestApplyRotaryEmbSingle:
    """Tests for apply_rotary_emb_single()."""

    def test_basic(self):
        dim = 16
        x = torch.randn(3, 4, dim)  # (batch, heads, dim)
        result = apply_rotary_emb_single(x, position=5)
        assert result.shape == x.shape

    def test_matches_precomputed(self):
        """Single should match precomputed for same position."""
        dim = 16
        seq_len = 8
        cos, sin = precompute_rope_cos_sin(seq_len, dim)
        x = torch.randn(2, dim)
        for p in range(seq_len):
            result_single = apply_rotary_emb_single(x, p)
            result_precomp = apply_rotary_emb(
                x.unsqueeze(-2), cos, sin, offset=p
            ).squeeze(-2)
            assert torch.allclose(result_single, result_precomp, atol=1e-6)

    def test_zero_position(self):
        """Position 0 should leave the vector unchanged."""
        x = torch.randn(8)
        result = apply_rotary_emb_single(x, 0)
        assert torch.allclose(result, x)

    def test_preserves_norm(self):
        x = torch.randn(2, 4, 16)
        result = apply_rotary_emb_single(x, 7)
        assert torch.allclose(
            torch.norm(x, dim=-1), torch.norm(result, dim=-1), atol=1e-5
        )

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            apply_rotary_emb_single(torch.randn(15), 3)


class TestGetRopeEmbeddings:
    """Tests for get_rope_embeddings()."""

    def test_contiguous(self):
        positions = torch.arange(8)
        cos, sin = get_rope_embeddings(positions, 16)
        assert cos.shape == (8, 16)
        assert sin.shape == (8, 16)

    def test_non_contiguous(self):
        positions = torch.tensor([0, 5, 100, 3])
        cos, sin = get_rope_embeddings(positions, 16)
        assert cos.shape == (4, 16)
        assert sin.shape == (4, 16)

    def test_matches_precomputed(self):
        """Should match precompute_rope_cos_sin for same positions."""
        max_len = 16
        dim = 32
        cos_pre, sin_pre = precompute_rope_cos_sin(max_len, dim)
        positions = torch.arange(max_len)
        cos_get, sin_get = get_rope_embeddings(positions, dim)
        assert torch.allclose(cos_get, cos_pre)
        assert torch.allclose(sin_get, sin_pre)

    def test_scalar_position(self):
        """Single position should work."""
        cos, sin = get_rope_embeddings(torch.tensor(3), 8)
        assert cos.shape == (8,)
        assert sin.shape == (8,)


class TestNumpyApplyRotaryEmb:
    """Tests for the NumPy reference implementation."""

    def test_basic(self):
        x = np.array([1.0, 0.0, 1.0, 0.0])
        result = numpy_apply_rotary_emb(x, 0)
        # Position 0: angle = 0, cos=1, sin=0 => identity
        assert np.allclose(result, x)

    def test_rotation_preserves_norm(self):
        np.random.seed(42)
        x = np.random.randn(8).astype(np.float32)
        norm_before = np.linalg.norm(x)
        for p in [0, 1, 5, 100]:
            result = numpy_apply_rotary_emb(x, p)
            norm_after = np.linalg.norm(result)
            assert abs(norm_before - norm_after) < 1e-6

    def test_pairwise_rotation(self):
        """Verify the 2x2 rotation for a single pair."""
        dim = 2
        x = np.array([1.0, 0.0])
        pos = 1
        theta = 1.0 / (10000.0 ** (0 / dim))  # = 1.0
        angle = pos * theta  # = 1.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        expected = np.array([cos_a, sin_a])  # [cos(1), sin(1)]
        result = numpy_apply_rotary_emb(x, pos)
        assert np.allclose(result, expected)

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            numpy_apply_rotary_emb(np.array([1.0, 2.0, 3.0]), 0)

    def test_matches_torch_single(self):
        """NumPy reference should match Torch single."""
        dim = 16
        np.random.seed(42)
        x_np = np.random.randn(dim).astype(np.float32)
        for p in [0, 1, 10, 100]:
            result_np = numpy_apply_rotary_emb(x_np, p)
            result_torch = apply_rotary_emb_single(
                torch.from_numpy(x_np), p
            )
            assert np.allclose(result_np, result_torch.numpy(), atol=1e-5)


# =========================================================================
# Position Interpolation (PI) tests
# =========================================================================

class TestPIScaler:
    """Tests for PIScaler."""

    def test_basic(self):
        scaler = PIScaler(scale=2.0)
        assert scaler.scale == 2.0
        assert scaler.base == 10000.0

    def test_scaled_position(self):
        scaler = PIScaler(scale=4.0)
        assert scaler.get_scaled_position(0) == 0.0
        assert scaler.get_scaled_position(100) == 25.0
        assert scaler.get_scaled_position(4000) == 1000.0

    def test_scaled_positions_tensor(self):
        scaler = PIScaler(scale=2.0)
        positions = torch.arange(0, 10)
        scaled = scaler.get_scaled_positions(positions)
        expected = positions.float() / 2.0
        assert torch.allclose(scaled, expected)

    def test_precompute_cos_sin(self):
        scaler = PIScaler(scale=2.0)
        cos, sin = scaler.precompute_cos_sin(16, 64)
        assert cos.shape == (16, 64)
        assert sin.shape == (16, 64)

    def test_orthogonality(self):
        """PI should preserve orthogonality."""
        scaler = PIScaler(scale=2.0)
        cos, sin = scaler.precompute_cos_sin(32, 64)
        squares = cos ** 2 + sin ** 2
        assert torch.allclose(squares, torch.ones_like(squares), atol=1e-6)

    def test_pi_half_angles(self):
        """PI(scale=2) at position 2p should match base at position p."""
        dim = 64
        base = 10000.0
        scaler = PIScaler(scale=2.0, base=base)
        cos_pi, sin_pi = scaler.precompute_cos_sin(16, dim)

        cos_base, sin_base = precompute_rope_cos_sin(8, dim, base=base)

        # PI position 2 has the same effective position as base position 1
        # (2/2 = 1), so the embeddings should match
        assert torch.allclose(cos_pi[2], cos_base[1], atol=1e-6)
        assert torch.allclose(sin_pi[2], sin_base[1], atol=1e-6)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PIScaler(scale=-1.0)

    def test_zero_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PIScaler(scale=0.0)


# =========================================================================
# NTK-aware Scaling tests
# =========================================================================

class TestNTKScaler:
    """Tests for NTKScaler."""

    def test_basic(self):
        scaler = NTKScaler(scale=2.0)
        assert scaler.scale == 2.0

    def test_ntk_base(self):
        """Verify the NTK base formula."""
        dim = 64
        scaler = NTKScaler(scale=2.0, base=10000.0)
        ntk_base = scaler.get_ntk_base(dim)
        # base' = base * alpha^{d/(d-2)}
        expected = 10000.0 * (2.0 ** (64 / 62))
        assert ntk_base == pytest.approx(expected, rel=1e-6)

    def test_ntk_base_larger(self):
        """NTK base should be > original base for scale > 1."""
        dim = 64
        scaler = NTKScaler(scale=4.0, base=10000.0)
        ntk_base = scaler.get_ntk_base(dim)
        assert ntk_base > 10000.0

    def test_ntk_base_scale_one(self):
        """Scale=1 should give original base."""
        scaler = NTKScaler(scale=1.0, base=10000.0)
        assert scaler.get_ntk_base(64) == pytest.approx(10000.0)

    def test_precompute_cos_sin(self):
        scaler = NTKScaler(scale=2.0)
        cos, sin = scaler.precompute_cos_sin(16, 64)
        assert cos.shape == (16, 64)
        assert sin.shape == (16, 64)
        # Should have valid cos/sin values
        assert (cos >= -1).all() and (cos <= 1).all()

    def test_effective_scale(self):
        """Effective scale should approximately equal requested scale."""
        dim = 64
        scaler = NTKScaler(scale=2.0, base=10000.0)
        eff = scaler.get_effective_scale(dim)
        assert eff == pytest.approx(2.0, rel=1e-6)

    def test_orthogonality(self):
        scaler = NTKScaler(scale=2.0)
        cos, sin = scaler.precompute_cos_sin(32, 64)
        squares = cos ** 2 + sin ** 2
        assert torch.allclose(squares, torch.ones_like(squares), atol=1e-6)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            NTKScaler(scale=-1.0)

    def test_small_dim(self):
        """NTK base should work for small dimensions."""
        scaler = NTKScaler(scale=2.0, base=10000.0)
        ntk_base = scaler.get_ntk_base(4)
        assert ntk_base > 0


# =========================================================================
# YaRN tests
# =========================================================================

class TestYaRNScaler:
    """Tests for YaRNScaler."""

    def test_basic(self):
        scaler = YaRNScaler(scale=2.0)
        assert scaler.scale == 2.0
        assert scaler.temperature is not None

    def test_default_temperature(self):
        """Default temperature = scale^{0.25}."""
        scaler = YaRNScaler(scale=2.0)
        expected_t = 2.0 ** 0.25
        assert scaler.temperature == pytest.approx(expected_t)

        # For large scale, temperature is capped at 32.0
        scaler_huge = YaRNScaler(scale=1000000.0)
        assert scaler_huge.temperature <= 32.0

    def test_custom_temperature(self):
        scaler = YaRNScaler(scale=2.0, temperature=0.5)
        assert scaler.temperature == 0.5

    def test_ntk_base_matches_ntk_scaler(self):
        """YaRN's NTK base should match NTKScaler's."""
        dim = 64
        ntk = NTKScaler(scale=2.0, base=50000.0)
        yarn = YaRNScaler(scale=2.0, base=50000.0)
        assert yarn.get_ntk_base(dim) == pytest.approx(ntk.get_ntk_base(dim))

    def test_scale_attention_logits(self):
        """Temperature scaling divides logits."""
        scaler = YaRNScaler(scale=2.0, temperature=2.0)
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scaled = scaler.scale_attention_logits(logits)
        assert torch.allclose(scaled, logits / 2.0)

    def test_precompute_cos_sin(self):
        scaler = YaRNScaler(scale=2.0)
        cos, sin = scaler.precompute_cos_sin(16, 64)
        assert cos.shape == (16, 64)
        assert (cos >= -1).all() and (cos <= 1).all()

    def test_orthogonality(self):
        scaler = YaRNScaler(scale=2.0)
        cos, sin = scaler.precompute_cos_sin(32, 64)
        squares = cos ** 2 + sin ** 2
        assert torch.allclose(squares, torch.ones_like(squares), atol=1e-6)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            YaRNScaler(scale=-1.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            YaRNScaler(scale=2.0, temperature=-1.0)

    def test_large_scale_temperature_capped(self):
        """Temperature should be capped at 32 for very large scales."""
        scaler = YaRNScaler(scale=1000000.0)
        assert scaler.temperature <= 32.0


# =========================================================================
# DPE tests
# =========================================================================

class TestDPE:
    """Tests for Dynamic Position Encoding."""

    def test_build_mask_local(self):
        """Within a chunk, all tokens should attend to each other."""
        config = DPEConfig(chunk_size=8)
        mask = build_dpe_mask(16, config)
        # Position 0 and 7 are in same chunk -> should be allowed (0)
        assert mask[0, 7] == 0
        assert mask[7, 0] == 0

    def test_build_mask_cross_chunk_blocked(self):
        """Across chunks (non-global tokens), attention should be blocked."""
        config = DPEConfig(chunk_size=8, global_token_per_chunk=0)
        mask = build_dpe_mask(16, config)
        # Position 0 (chunk 0) and position 8 (chunk 1)
        assert mask[0, 8] == float("-inf")

    def test_build_mask_global_allowed(self):
        """Global tokens should attend across chunks."""
        config = DPEConfig(chunk_size=8, global_token_per_chunk=1)
        mask = build_dpe_mask(16, config)
        # Position 0 (first token global) and position 8 (first token global)
        assert mask[0, 8] == 0
        assert mask[8, 0] == 0

    def test_build_mask_local_radius(self):
        """With radius, tokens beyond the radius within the same chunk are blocked."""
        config = DPEConfig(chunk_size=8, local_attention_radius=3)
        mask = build_dpe_mask(16, config)
        # Position 0 and 4: distance 4 > radius 3 → blocked
        assert mask[0, 4] == float("-inf")
        # Position 0 and 3: distance 3 <= radius 3 → allowed
        assert mask[0, 3] == 0

    def test_dpe_precompute_cos_sin(self):
        config = DPEConfig(chunk_size=2048)
        cos, sin = dpe_precompute_cos_sin(config, 64)
        assert cos.shape == (2048, 64)
        assert sin.shape == (2048, 64)
        assert (cos >= -1).all() and (cos <= 1).all()

    def test_apply_dpe_rotary(self):
        config = DPEConfig(chunk_size=16, base=10000.0)
        cos, sin = dpe_precompute_cos_sin(config, 64)
        x = torch.randn(2, 4, 32, 64)  # (batch, heads, seq, dim)
        result = apply_dpe_rotary(x, cos, sin, config)
        assert result.shape == x.shape
        # Should not equal input
        assert not torch.allclose(result, x)

    def test_dpe_chunk_position_cycle(self):
        """After chunk_size positions, DPE should repeat the embedding pattern."""
        config = DPEConfig(chunk_size=8, base=10000.0)
        cos, sin = dpe_precompute_cos_sin(config, 16)
        # Apply DPE rotary at positions that cycle: a token at chunk-local
        # position p should get the same rotation regardless of which chunk
        # it's in. Use a single token at offset 0 (chunk-local pos 0) vs
        # offset chunk_size (also chunk-local pos 0).
        x = torch.randn(1, 1, 1, 16)  # single token
        result_pos0 = apply_dpe_rotary(x, cos, sin, config, offset=0)
        result_pos8 = apply_dpe_rotary(x, cos, sin, config, offset=8)
        assert torch.allclose(result_pos0, result_pos8, atol=1e-6)
        # Also test: offset 3 vs offset 11 (both chunk-local position 3)
        result_pos3 = apply_dpe_rotary(x, cos, sin, config, offset=3)
        result_pos11 = apply_dpe_rotary(x, cos, sin, config, offset=11)
        assert torch.allclose(result_pos3, result_pos11, atol=1e-6)

    def test_dpe_preserves_norm(self):
        config = DPEConfig(chunk_size=8)
        cos, sin = dpe_precompute_cos_sin(config, 16)
        x = torch.randn(1, 1, 8, 16)
        result = apply_dpe_rotary(x, cos, sin, config)
        norms_before = torch.norm(x, dim=-1)
        norms_after = torch.norm(result, dim=-1)
        assert torch.allclose(norms_before, norms_after, atol=1e-5)


# =========================================================================
# Integration tests
# =========================================================================

class TestGetCosSinForMethod:
    """Tests for get_cos_sin_for_method()."""

    @pytest.mark.parametrize("method", ["base", "pi", "ntk", "yarn"])
    def test_valid_methods(self, method):
        cos, sin = get_cos_sin_for_method(method, 32, 64)
        assert cos.shape == (32, 64)
        assert sin.shape == (32, 64)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            get_cos_sin_for_method("invalid", 32, 64)

    def test_methods_produce_different_results(self):
        """Different methods should produce different angle tables."""
        cos_base, _ = get_cos_sin_for_method("base", 32, 64)
        cos_pi, _ = get_cos_sin_for_method("pi", 32, 64)
        cos_ntk, _ = get_cos_sin_for_method("ntk", 32, 64)
        # PI and base should differ
        assert not torch.allclose(cos_base, cos_pi)
        # NTK and base should differ
        assert not torch.allclose(cos_base, cos_ntk)
        # PI and NTK should differ
        assert not torch.allclose(cos_pi, cos_ntk)


class TestCompareAngles:
    """Tests for compare_angles()."""

    def test_basic(self):
        results = compare_angles(dim=64, max_pos=4096, scale=2.0)
        assert "base" in results
        assert "pi" in results
        assert "ntk" in results
        assert "yarn" in results
        for v in results.values():
            assert v.shape == (32,)  # dim//2

    def test_pi_angles_smaller_than_base(self):
        """PI should produce smaller angles than base (positions are downscaled)."""
        results = compare_angles(dim=64, max_pos=4096, scale=2.0)
        assert (results["pi"] < results["base"]).all()

    def test_ntk_yarn_same_angles(self):
        """NTK and YaRN should produce the same angles (same base)."""
        results = compare_angles(dim=64, max_pos=4096, scale=2.0)
        assert torch.allclose(results["ntk"], results["yarn"])

    def test_ntk_vs_pi_frequency_distribution(self):
        """NTK high-frequency angles > PI high-frequency angles."""
        results = compare_angles(dim=64, max_pos=4096, scale=2.0)
        # High frequencies are at the beginning (small i → large theta)
        n_high = 4
        pi_high_mean = results["pi"][:n_high].mean()
        base_high_mean = results["base"][:n_high].mean()
        ntk_high_mean = results["ntk"][:n_high].mean()
        # NTK angles should be between PI and base for high frequencies
        assert ntk_high_mean > pi_high_mean
        assert ntk_high_mean < base_high_mean


# =========================================================================
# End-to-end usage tests
# =========================================================================

class TestEndToEnd:
    """End-to-end scenarios that test realistic usage patterns."""

    def test_full_rope_pipeline(self):
        """Simulate applying RoPE in an attention layer."""
        batch, heads, seq, dim = 2, 4, 16, 64

        # Generate random Q and K
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)

        # Precompute cos/sin
        cos, sin = precompute_rope_cos_sin(seq * 2, dim)

        # Apply RoPE
        q_rope = apply_rotary_emb(q, cos, sin)
        k_rope = apply_rotary_emb(k, cos, sin)

        # Attention: Q @ K^T
        attn = torch.matmul(q_rope, k_rope.transpose(-2, -1))

        assert attn.shape == (batch, heads, seq, seq)
        # Attention should not be NaN
        assert not torch.isnan(attn).any()

    def test_extrapolation_pipeline_with_pi(self):
        """Use PI to handle longer sequences."""
        batch, heads, dim = 1, 2, 64
        test_len = 4096
        scale = 2.0

        scaler = PIScaler(scale=scale)
        cos, sin = scaler.precompute_cos_sin(test_len, dim)

        q = torch.randn(batch, heads, test_len, dim)
        k = torch.randn(batch, heads, test_len, dim)

        q_rope = apply_rotary_emb(q, cos, sin)
        k_rope = apply_rotary_emb(k, cos, sin)

        attn = torch.matmul(q_rope, k_rope.transpose(-2, -1))
        assert not torch.isnan(attn).any()

    def test_extrapolation_pipeline_with_ntk(self):
        """Use NTK-aware scaling for longer sequences."""
        batch, heads, dim = 1, 2, 64
        test_len = 4096

        scaler = NTKScaler(scale=2.0)
        cos, sin = scaler.precompute_cos_sin(test_len, dim)

        q = torch.randn(batch, heads, test_len, dim)
        k = torch.randn(batch, heads, test_len, dim)

        q_rope = apply_rotary_emb(q, cos, sin)
        k_rope = apply_rotary_emb(k, cos, sin)

        attn = torch.matmul(q_rope, k_rope.transpose(-2, -1))
        assert not torch.isnan(attn).any()

    def test_extrapolation_pipeline_with_yarn(self):
        """Use YaRN with temperature scaling."""
        batch, heads, dim = 1, 2, 64
        test_len = 4096

        scaler = YaRNScaler(scale=2.0)
        cos, sin = scaler.precompute_cos_sin(test_len, dim)

        q = torch.randn(batch, heads, test_len, dim)
        k = torch.randn(batch, heads, test_len, dim)

        q_rope = apply_rotary_emb(q, cos, sin)
        k_rope = apply_rotary_emb(k, cos, sin)

        attn_raw = torch.matmul(q_rope, k_rope.transpose(-2, -1))
        attn_scaled = scaler.scale_attention_logits(attn_raw)

        # With temperature > 1, scaled logits should have smaller magnitude
        if scaler.temperature > 1.0:
            assert attn_scaled.abs().mean() < attn_raw.abs().mean()

    def test_dpe_full_pipeline(self):
        """Use DPE for chunked attention with local/global masking."""
        batch, heads, dim = 1, 2, 64
        seq_len = 4096

        config = DPEConfig(chunk_size=1024, global_token_per_chunk=1)
        cos, sin = dpe_precompute_cos_sin(config, dim)
        mask = build_dpe_mask(seq_len, config)

        q = torch.randn(batch, heads, seq_len, dim)
        k = torch.randn(batch, heads, seq_len, dim)

        q_rope = apply_dpe_rotary(q, cos, sin, config)
        k_rope = apply_dpe_rotary(k, cos, sin, config)

        attn = torch.matmul(q_rope, k_rope.transpose(-2, -1))
        attn_masked = attn + mask  # additive mask

        assert attn_masked.shape == (batch, heads, seq_len, seq_len)
        # Position 2 is NOT a global token (only first=0 and last=1023 of chunk 0).
        # Position 1025 is NOT a global token (only first=1024 and last=2047 of chunk 1).
        # Non-global tokens should NOT see non-global tokens in different chunks.
        cross_chunk_masked = attn_masked[0, 0, 2, 1025]
        assert cross_chunk_masked == float("-inf")

    def test_relative_position_encoding_property(self):
        """RoPE encodes relative positions: q_m^T k_n depends on (m-n)."""
        dim = 32
        base = 10000.0

        v = torch.randn(dim)

        # q * k^T dot product for positions differing by 2
        q_m = apply_rotary_emb_single(v, 3, base)  # position 3
        k_n = apply_rotary_emb_single(v, 5, base)  # position 5
        dot1 = torch.dot(q_m, k_n)

        q_m2 = apply_rotary_emb_single(v, 1, base)  # position 1
        k_n2 = apply_rotary_emb_single(v, 3, base)  # position 3
        dot2 = torch.dot(q_m2, k_n2)

        # Same relative distance (2) → same dot product
        assert torch.allclose(dot1, dot2, atol=1e-5)

    def test_pi_angle_correctness_at_boundary(self):
        """At PI boundary, doubled positions should match original."""
        dim = 64
        train_len = 2048
        scaler = PIScaler(scale=2.0)
        cos_pi, sin_pi = scaler.precompute_cos_sin(train_len * 2, dim)
        cos_train, sin_train = precompute_rope_cos_sin(train_len, dim)

        # PI position 4094 maps to 4094/2 = 2047, matching train last position
        assert torch.allclose(cos_pi[4094], cos_train[2047], atol=1e-5)
        assert torch.allclose(sin_pi[4094], sin_train[2047], atol=1e-5)
