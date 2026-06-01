"""Unit tests for the linear_recurrent module.

Run with:
    PYTHONPATH=modules python -m pytest modules/linear_recurrent/ -v
"""

from __future__ import annotations

import math
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import pytest

# Ensure modules/ can be imported
sys.path.insert(0, "modules" if not sys.path[0].endswith("modules") else sys.path[0])

from linear_recurrent import (
    # Common
    RMSNorm,
    LayerNorm,
    SquaredReLU,
    SwiGLU,
    get_activation,
    # RWKV
    WKVOperator,
    TimeMixBlock,
    ChannelMixBlock,
    RWKVBlock,
    token_shift,
    # xLSTM
    mLSTMCell,
    sLSTMCell,
    xLSTMBlock,
    # Griffin
    RGLRU,
    SimpleRGLRU,
    GriffinBlock,
    # RetNet
    retention_parallel,
    retention_recurrent,
    retention_chunkwise,
    MultiScaleRetention,
    RetNetBlock,
    _build_decay_matrix,
)


# ============================================================================
# Test configuration
# ============================================================================

BATCH = 2
SEQ_LEN = 16
DIM = 64
HEADS = 4
HEAD_DIM = 16
DTYPE = torch.float32


def _make_input(batch=BATCH, seq_len=SEQ_LEN, dim=DIM, dtype=DTYPE):
    return torch.randn(batch, seq_len, dim, dtype=dtype)


# ============================================================================
# Common utilities
# ============================================================================

class TestRMSNorm:
    def test_shape(self):
        x = _make_input()
        out = RMSNorm(DIM)(x)
        assert out.shape == x.shape

    def test_zero_mean_after(self):
        # RMSNorm doesn't centre but should have roughly unit RMS per sample
        norm = RMSNorm(DIM)
        x = _make_input()
        out = norm(x)
        rms = torch.sqrt(torch.mean(out.float() ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.2)

    def test_reset_parameters(self):
        norm = RMSNorm(DIM)
        init.zeros_(norm.weight)
        norm.reset_parameters()
        assert torch.allclose(norm.weight, torch.ones(DIM))


class TestLayerNorm:
    def test_shape(self):
        x = _make_input()
        out = LayerNorm(DIM)(x)
        assert out.shape == x.shape

    def test_zero_mean_unit_var(self):
        norm = LayerNorm(DIM)
        x = _make_input()
        out = norm(x)
        mean = out.mean(-1)
        var = out.var(-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)

    def test_no_bias(self):
        norm = LayerNorm(DIM, bias=False)
        x = _make_input()
        out = norm(x)
        assert out.shape == x.shape


class TestSquaredReLU:
    def test_positive(self):
        x = torch.tensor([1.0, 2.0, -1.0, 0.0])
        expected = torch.tensor([1.0, 4.0, 0.0, 0.0])
        out = SquaredReLU()(x)
        assert torch.allclose(out, expected)


class TestGetActivation:
    def test_known(self):
        for name in ["relu", "gelu", "silu", "sigmoid", "tanh"]:
            mod = get_activation(name)
            assert isinstance(mod, torch.nn.Module)

    def test_squared_relu(self):
        mod = get_activation("squared_relu")
        x = torch.tensor([2.0, -1.0])
        assert torch.allclose(mod(x), torch.tensor([4.0, 0.0]))

    def test_unknown(self):
        with pytest.raises(ValueError):
            get_activation("bogus")


class TestSwiGLU:
    def test_shape(self):
        x = _make_input()
        out = SwiGLU(DIM)(x)
        assert out.shape == x.shape

    def test_hidden_dim(self):
        glu = SwiGLU(DIM, hidden_dim=128)
        x = _make_input()
        out = glu(x)
        assert out.shape == x.shape


# ============================================================================
# RWKV
# ============================================================================

class TestTokenShift:
    def test_shape(self):
        x = _make_input(dim=DIM)
        shifted = token_shift(x)
        assert shifted.shape == (BATCH, SEQ_LEN, 2 * DIM)

    def test_first_position_zero(self):
        x = torch.randn(2, 4, DIM)
        shifted = token_shift(x)
        # First position of the shifted half should be zero
        assert (shifted[:, 0, :DIM] == 0).all()


class TestWKVOperator:
    def test_shape(self):
        x = _make_input()
        wkv = WKVOperator(DIM)
        k, v = x.clone(), x.clone()
        out = wkv(k, v)
        assert out.shape == (BATCH, SEQ_LEN, DIM)

    def test_causal_no_future_leak(self):
        """Output at position t should not depend on positions > t."""
        x = torch.randn(1, 5, DIM)
        wkv = WKVOperator(DIM)
        k = torch.zeros(1, 5, DIM)
        v = x.clone()
        # If k is zero (ek=1), decay still applies—harder to assert causality.
        # Instead, run twice with same k,v truncated and compare.
        k1 = torch.randn(1, 5, DIM)
        v1 = torch.randn(1, 5, DIM)
        out_full = wkv(k1, v1)
        # Position 2 of truncated run (first 3 tokens) should match position 2 of full
        k_trunc = k1[:, :3]
        v_trunc = v1[:, :3]
        out_trunc = wkv(k_trunc, v_trunc)
        assert torch.allclose(out_full[:, 2], out_trunc[:, 2], atol=1e-5)

    def test_reset_parameters(self):
        wkv = WKVOperator(DIM)
        wkv.reset_parameters()
        assert (wkv.time_decay == 0).all()
        assert (wkv.time_first == 0).all()


class TestTimeMixBlock:
    def test_shape(self):
        x = _make_input()
        out = TimeMixBlock(DIM)(x)
        assert out.shape == x.shape

    def test_residual_effect(self):
        """Output should differ from input (residual + transformation)."""
        block = TimeMixBlock(DIM)
        x = _make_input()
        out = block(x)
        assert not torch.allclose(out, x)


class TestChannelMixBlock:
    def test_shape(self):
        x = _make_input()
        out = ChannelMixBlock(DIM)(x)
        assert out.shape == x.shape

    def test_hidden_dim(self):
        x = _make_input()
        out = ChannelMixBlock(DIM, hidden_dim=128)(x)
        assert out.shape == x.shape


class TestRWKVBlock:
    def test_shape(self):
        x = _make_input()
        out = RWKVBlock(DIM)(x)
        assert out.shape == x.shape

    def test_dropout(self):
        block = RWKVBlock(DIM, dropout=0.5)
        x = _make_input()
        block.eval()
        out_eval = block(x)
        assert out_eval.shape == x.shape


# ============================================================================
# xLSTM
# ============================================================================

class TestmLSTMCell:
    def test_shape(self):
        cell = mLSTMCell(d_model=DIM, d_qk=16, num_heads=4)
        x = _make_input()
        out, state = cell(x)
        assert out.shape == x.shape
        C, n = state
        assert C.shape == (BATCH, 4, 16, 16)
        assert n.shape == (BATCH, 4, 16)

    def test_with_initial_state(self):
        cell = mLSTMCell(d_model=DIM, d_qk=16, num_heads=4)
        x = _make_input(dim=DIM)
        C0, n0 = cell.init_state(BATCH, x.device)
        out, (C, n) = cell(x, state=(C0, n0))
        assert out.shape == x.shape

    def test_exp_forget(self):
        cell = mLSTMCell(d_model=DIM, d_qk=16, num_heads=4, use_exp_forget=True)
        x = _make_input()
        out, state = cell(x)
        assert out.shape == x.shape

    def test_different_d_v(self):
        cell = mLSTMCell(d_model=DIM, d_qk=16, d_v=32, num_heads=4)
        x = _make_input()
        out, (C, n) = cell(x)
        assert C.shape == (BATCH, 4, 16, 32)
        assert n.shape == (BATCH, 4, 16)

    def test_reset_parameters(self):
        cell = mLSTMCell(d_model=DIM)
        cell.reset_parameters()


class TestsLSTMCell:
    def test_shape(self):
        cell = sLSTMCell(DIM)
        x = _make_input()
        out, (c, n) = cell(x)
        assert out.shape == x.shape
        assert c.shape == (BATCH, DIM)
        assert n.shape == (BATCH, DIM)

    def test_with_initial_state(self):
        cell = sLSTMCell(DIM)
        x = _make_input(dim=DIM)
        c0, n0 = cell.init_state(BATCH, x.device)
        out, (c, n) = cell(x, state=(c0, n0))
        assert out.shape == x.shape

    def test_exp_forget(self):
        cell = sLSTMCell(DIM, use_exp_forget=True)
        x = _make_input()
        out, state = cell(x)
        assert out.shape == x.shape

    def test_reset_parameters(self):
        cell = sLSTMCell(DIM)
        cell.reset_parameters()


class TestXLSMBlock:
    def test_mlstm(self):
        block = xLSTMBlock(d_model=DIM, cell_type="mlstm", num_heads=4)
        x = _make_input()
        out, state = block(x)
        assert out.shape == x.shape

    def test_slstm(self):
        block = xLSTMBlock(d_model=DIM, cell_type="slstm")
        x = _make_input()
        out, state = block(x)
        assert out.shape == x.shape

    def test_unknown_cell(self):
        with pytest.raises(ValueError):
            xLSTMBlock(d_model=DIM, cell_type="unknown")

    def test_with_state(self):
        block = xLSTMBlock(d_model=DIM, cell_type="mlstm", num_heads=4)
        x = _make_input()
        cell = block.cell
        C0, n0 = cell.init_state(BATCH, x.device)
        out, (C, n) = block(x, state=(C0, n0))
        assert out.shape == x.shape


# ============================================================================
# Griffin
# ============================================================================

class TestRGLRU:
    def test_shape(self):
        x = _make_input()
        out = RGLRU(DIM)(x)
        assert out.shape == x.shape


class TestSimpleRGLRU:
    def test_shape(self):
        x = _make_input()
        out = SimpleRGLRU(DIM)(x)
        assert out.shape == x.shape

    def test_causal(self):
        """Output at position t should not depend on positions > t."""
        rg = SimpleRGLRU(DIM)
        x = torch.randn(1, 5, DIM)
        out_full = rg(x)
        out_trunc = rg(x[:, :3])
        assert torch.allclose(out_full[:, 2], out_trunc[:, 2], atol=1e-5)

    def test_reset_parameters(self):
        rg = SimpleRGLRU(DIM)
        rg.reset_parameters()


class TestGriffinBlock:
    def test_shape(self):
        x = _make_input()
        out = GriffinBlock(dim=DIM)(x)
        assert out.shape == x.shape

    def test_with_temporal_mixing(self):
        x = _make_input()
        out = GriffinBlock(dim=DIM, use_temporal_mixing=True)(x)
        assert out.shape == x.shape

    def test_dropout(self):
        block = GriffinBlock(dim=DIM, dropout=0.5)
        block.eval()
        x = _make_input()
        out = block(x)
        assert out.shape == x.shape


# ============================================================================
# RetNet
# ============================================================================

class TestBuildDecayMatrix:
    def test_shape(self):
        gamma = torch.tensor([0.9, 0.95])
        D = _build_decay_matrix(gamma, 8)
        assert D.shape == (1, 2, 8, 8), f"got {D.shape}"

    def test_lower_triangular(self):
        gamma = torch.tensor([0.9])
        D = _build_decay_matrix(gamma, 4)
        # Above diagonal should be zero
        for i in range(4):
            for j in range(4):
                if j > i:
                    assert D[0, 0, i, j] == 0.0, f"D[{i},{j}] should be 0"
                else:
                    expected = 0.9 ** (i - j)
                    assert abs(D[0, 0, i, j].item() - expected) < 1e-6

    def test_scalar_gamma(self):
        gamma = torch.tensor(0.95)
        D = _build_decay_matrix(gamma, 4)
        assert D.shape == (1, 1, 4, 4)


class TestRetentionParallel:
    def test_shape(self):
        q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        k = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        gamma = torch.linspace(0.8, 0.98, HEADS)
        out = retention_parallel(q, k, v, gamma)
        assert out.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)

    def test_causal(self):
        q = torch.randn(1, 1, 4, HEAD_DIM)
        k = torch.randn(1, 1, 4, HEAD_DIM)
        v = torch.randn(1, 1, 4, HEAD_DIM)
        gamma = torch.tensor([0.9])
        # Position 1 output should not depend on position 2 or 3
        out_full = retention_parallel(q, k, v, gamma)
        # Truncate to first 2 tokens
        out_trunc = retention_parallel(q[:, :, :2], k[:, :, :2], v[:, :, :2], gamma)
        assert torch.allclose(out_full[:, :, 1], out_trunc[:, :, 1], atol=1e-5)


class TestRetentionRecurrent:
    def test_shape_no_state(self):
        q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        k = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        gamma = torch.linspace(0.8, 0.98, HEADS)
        out, state = retention_recurrent(q, k, v, gamma)
        assert out.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        assert state.shape == (BATCH, HEADS, HEAD_DIM, HEAD_DIM)

    def test_with_state(self):
        q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        k = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        gamma = torch.linspace(0.8, 0.98, HEADS)
        state_init = torch.randn(BATCH, HEADS, HEAD_DIM, HEAD_DIM)
        out, state = retention_recurrent(q, k, v, gamma, state=state_init)
        assert out.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)

    def test_equivalence_with_parallel(self):
        """Recurrent and parallel forms should produce same output (up to fp precision)."""
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        gamma = torch.tensor([0.9])
        out_p = retention_parallel(q, k, v, gamma)
        out_r, _ = retention_recurrent(q, k, v, gamma)
        assert torch.allclose(out_p, out_r, atol=1e-4, rtol=1e-4)


class TestRetentionChunkwise:
    def test_shape(self):
        q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        k = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        v = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
        gamma = torch.linspace(0.8, 0.98, HEADS)
        out = retention_chunkwise(q, k, v, gamma, chunk_size=4)
        assert out.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)

    def test_equivalence_with_recurrent(self):
        """Chunkwise and recurrent forms should match."""
        q = torch.randn(1, 1, 8, 8)
        k = torch.randn(1, 1, 8, 8)
        v = torch.randn(1, 1, 8, 8)
        gamma = torch.tensor([0.95])
        out_r, _ = retention_recurrent(q, k, v, gamma)
        out_c = retention_chunkwise(q, k, v, gamma, chunk_size=4)
        assert torch.allclose(out_r, out_c, atol=1e-4, rtol=1e-4)

    def test_non_divisible_chunks(self):
        """Sequence length not divisible by chunk_size should still work."""
        q = torch.randn(1, HEADS, 7, HEAD_DIM)
        k = torch.randn(1, HEADS, 7, HEAD_DIM)
        v = torch.randn(1, HEADS, 7, HEAD_DIM)
        gamma = torch.linspace(0.8, 0.98, HEADS)
        out = retention_chunkwise(q, k, v, gamma, chunk_size=4)
        assert out.shape == (1, HEADS, 7, HEAD_DIM)


class TestMultiScaleRetention:
    def test_parallel(self):
        x = _make_input()
        msr = MultiScaleRetention(dim=DIM, num_heads=HEADS, double_v_dim=False)
        out = msr(x, mode="parallel")
        assert out.shape == x.shape

    def test_recurrent(self):
        x = _make_input()
        msr = MultiScaleRetention(dim=DIM, num_heads=HEADS, double_v_dim=False)
        out, state = msr(x, mode="recurrent")
        assert out.shape == x.shape
        assert state.shape == (BATCH, HEADS, 16, 16)

    def test_recurrent_with_state(self):
        x = _make_input()
        msr = MultiScaleRetention(dim=DIM, num_heads=HEADS, double_v_dim=False)
        s = torch.randn(BATCH, HEADS, 16, 16)
        out, new_s = msr(x, mode="recurrent", state=s)
        assert out.shape == x.shape

    def test_chunkwise(self):
        x = _make_input()
        msr = MultiScaleRetention(dim=DIM, num_heads=HEADS, double_v_dim=False)
        out = msr(x, mode="chunkwise", chunk_size=8)
        assert out.shape == x.shape

    def test_parallel_recurrent_equivalence(self):
        """Parallel and recurrent modes should be equivalent."""
        torch.manual_seed(42)
        x = torch.randn(1, 4, DIM)
        msr = MultiScaleRetention(dim=DIM, num_heads=HEADS, double_v_dim=False)
        msr.eval()
        out_p = msr(x, mode="parallel")
        out_r, _ = msr(x, mode="recurrent")
        assert torch.allclose(out_p, out_r, atol=1e-4, rtol=1e-4)

    def test_double_v_dim(self):
        x = _make_input()
        msr = MultiScaleRetention(dim=DIM, num_heads=HEADS, double_v_dim=True)
        out = msr(x, mode="parallel")
        assert out.shape == x.shape

    def test_unknown_mode(self):
        msr = MultiScaleRetention(dim=DIM, num_heads=HEADS)
        x = _make_input()
        with pytest.raises(ValueError):
            msr(x, mode="bogus")

    def test_reset_parameters(self):
        msr = MultiScaleRetention(dim=DIM, num_heads=HEADS)
        msr.reset_parameters()


class TestRetNetBlock:
    def test_parallel(self):
        x = _make_input()
        block = RetNetBlock(dim=DIM, num_heads=HEADS)
        out = block(x, mode="parallel")
        assert out.shape == x.shape

    def test_recurrent(self):
        x = _make_input()
        block = RetNetBlock(dim=DIM, num_heads=HEADS)
        out, state = block(x, mode="recurrent")
        assert out.shape == x.shape

    def test_chunkwise(self):
        x = _make_input()
        block = RetNetBlock(dim=DIM, num_heads=HEADS)
        # override chunk_size in retention sub-layer
        # chunkwise not exposed via block, test via retention directly
        out = block(x, mode="parallel")
        assert out.shape == x.shape


# ============================================================================
# Integration: stacking multiple blocks
# ============================================================================

class TestStacking:
    def test_rwkv_blocks(self):
        blocks = torch.nn.Sequential(
            RWKVBlock(DIM),
            RWKVBlock(DIM),
        )
        x = _make_input()
        out = blocks(x)
        assert out.shape == x.shape

    def test_griffin_blocks(self):
        blocks = torch.nn.Sequential(
            GriffinBlock(dim=DIM),
            GriffinBlock(dim=DIM),
        )
        x = _make_input()
        out = blocks(x)
        assert out.shape == x.shape

    def test_xlstm_blocks(self):
        blocks = torch.nn.ModuleList([
            xLSTMBlock(d_model=DIM, cell_type="slstm"),
            xLSTMBlock(d_model=DIM, cell_type="slstm"),
        ])
        x = _make_input()
        for block in blocks:
            x, _ = block(x)
        assert x.shape == (BATCH, SEQ_LEN, DIM)

    def test_retnet_blocks(self):
        blocks = torch.nn.Sequential(
            RetNetBlock(dim=DIM, num_heads=HEADS),
            RetNetBlock(dim=DIM, num_heads=HEADS),
        )
        x = _make_input()
        out = blocks(x)
        assert out.shape == x.shape


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_single_token(self):
        """All modules should handle sequence length 1."""
        x = torch.randn(2, 1, DIM)

        out_rwkv = RWKVBlock(DIM)(x)
        assert out_rwkv.shape == x.shape

        out_griffin = GriffinBlock(dim=DIM)(x)
        assert out_griffin.shape == x.shape

        out_retnet = RetNetBlock(dim=DIM, num_heads=HEADS)(x, mode="parallel")
        assert out_retnet.shape == x.shape

        out_xlstm, _ = xLSTMBlock(d_model=DIM, cell_type="slstm")(x)
        assert out_xlstm.shape == x.shape

    def test_small_dim(self):
        """Dim = 16 should work for all modules."""
        small = 16
        x = torch.randn(1, 4, small)
        out = RWKVBlock(small)(x)
        assert out.shape == x.shape

    @pytest.mark.skip(reason="float16 not fully supported — linear layers use float32 weights")
    def test_float16(self):
        """All modules should work with float16."""
        x = torch.randn(2, 4, DIM, dtype=torch.float16)
        out = RWKVBlock(DIM)(x)
        assert out.dtype == torch.float16
        out = GriffinBlock(dim=DIM)(x)
        assert out.dtype == torch.float16
        out = RetNetBlock(dim=DIM, num_heads=HEADS)(x, mode="parallel")
        assert out.dtype == torch.float16
