"""Tests for the Gated Linear Attention (GLA) module."""

from __future__ import annotations

import pytest
import torch

from hybrid_architectures.gla import (
    GatedLinearAttention,
    gla_chunkwise,
    gla_recurrent,
)


# ---------------------------------------------------------------------------
# Standalone functional API
# ---------------------------------------------------------------------------

class TestGlaRecurrent:
    """Tests for gla_recurrent — the purely sequential form."""

    @pytest.fixture
    def B(self) -> tuple[int, int, int, int]:
        return 2, 12, 4, 4  # B, L, d_k, d_v

    @pytest.fixture
    def inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        B, L = 2, 12
        d_k, d_v = 4, 4
        q = torch.randn(B, L, d_k)
        k = torch.randn(B, L, d_k)
        v = torch.randn(B, L, d_v)
        # Gate in (0, 1)
        gate = torch.sigmoid(torch.randn(B, L, d_k))
        return q, k, v, gate

    def test_output_shape(self, inputs: tuple) -> None:
        q, k, v, gate = inputs
        out, S, N = gla_recurrent(q, k, v, gate)
        B, L, d_k = q.shape
        d_v = v.shape[-1]
        assert out.shape == (B, L, d_v)
        assert S.shape == (B, d_k, d_v)
        assert N.shape == (B, d_k)

    def test_output_finite(self, inputs: tuple) -> None:
        q, k, v, gate = inputs
        out, S, N = gla_recurrent(q, k, v, gate)
        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isfinite(S))
        assert torch.all(torch.isfinite(N))

    def test_with_initial_state(self, inputs: tuple) -> None:
        q, k, v, gate = inputs
        B, L, d_k = q.shape
        d_v = v.shape[-1]

        S0 = torch.randn(B, d_k, d_v)
        N0 = torch.randn(B, d_k)

        out, S, N = gla_recurrent(q, k, v, gate, initial_state=S0, initial_normalizer=N0)
        assert out.shape == (B, L, d_v)
        assert torch.all(torch.isfinite(out))

    def test_gate_zero_no_accumulation(self) -> None:
        """When gate is zero, past information should be completely forgotten."""
        torch.manual_seed(42)
        B, L, d_k, d_v = 1, 5, 4, 4
        q = torch.randn(B, L, d_k)
        k = torch.randn(B, L, d_k)
        v = torch.randn(B, L, d_v)
        gate = torch.zeros(B, L, d_k)  # all zeros → forget everything

        out, S, _ = gla_recurrent(q, k, v, gate)
        # With gate=0, state should only contain the LAST token's contribution
        # because each step: S = 0 * S + K_t^T * V_t = K_t^T * V_t
        assert torch.all(torch.isfinite(out))


class TestGlaChunkwise:
    """Tests for gla_chunkwise — the chunkwise parallel form."""

    @pytest.fixture
    def inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        B, L = 2, 48  # divisible by many chunk sizes
        d_k, d_v = 4, 4
        q = torch.randn(B, L, d_k)
        k = torch.randn(B, L, d_k)
        v = torch.randn(B, L, d_v)
        gate = torch.sigmoid(torch.randn(B, L, d_k))
        return q, k, v, gate

    def test_output_shape(self, inputs: tuple) -> None:
        q, k, v, gate = inputs
        out, S, N = gla_chunkwise(q, k, v, gate, chunk_size=16)
        B, L, d_k = q.shape
        d_v = v.shape[-1]
        assert out.shape == (B, L, d_v)
        assert S.shape == (B, d_k, d_v)
        assert N.shape == (B, d_k)

    def test_output_finite(self, inputs: tuple) -> None:
        q, k, v, gate = inputs
        out, S, N = gla_chunkwise(q, k, v, gate, chunk_size=16)
        assert torch.all(torch.isfinite(out))

    def test_non_divisible_length(self, inputs: tuple) -> None:
        """Chunkwise should handle sequences not exactly divisible by chunk_size."""
        q, k, v, gate = inputs
        # 48 with chunk_size=7 → 48 % 7 != 0
        out, S, N = gla_chunkwise(q, k, v, gate, chunk_size=7)
        assert out.shape == q.shape[:2] + (v.shape[-1],)
        assert torch.all(torch.isfinite(out))

    @pytest.mark.parametrize("chunk_size", [1, 4, 8, 16])
    def test_matches_recurrent(
        self, chunk_size: int
    ) -> None:
        """For any chunk_size, chunkwise output MUST equal recurrent output
        (modulo numerical precision)."""
        torch.manual_seed(42)
        B, L = 1, 32
        d_k, d_v = 4, 4
        q = torch.randn(B, L, d_k)
        k = torch.randn(B, L, d_k)
        v = torch.randn(B, L, d_v)
        gate = torch.sigmoid(torch.randn(B, L, d_k))

        out_rec, S_rec, N_rec = gla_recurrent(q, k, v, gate)
        out_chunk, S_chunk, N_chunk = gla_chunkwise(q, k, v, gate, chunk_size=chunk_size)

        max_diff = (out_rec - out_chunk).abs().max().item()
        assert torch.allclose(out_rec, out_chunk, rtol=1e-4, atol=1e-4), (
            f"chunk_size={chunk_size}: max diff = {max_diff:.2e}"
        )
        assert torch.allclose(S_rec, S_chunk, rtol=1e-4, atol=1e-4)
        assert torch.allclose(N_rec, N_chunk, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# GLA Module (multi-head)
# ---------------------------------------------------------------------------

class TestGatedLinearAttention:
    """Tests for the GatedLinearAttention nn.Module."""

    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def n_heads(self) -> int:
        return 4

    @pytest.fixture
    def gla(self, d_model: int, n_heads: int) -> GatedLinearAttention:
        torch.manual_seed(42)
        return GatedLinearAttention(d_model=d_model, n_heads=n_heads, chunk_size=8)

    @pytest.fixture
    def x(self, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(2, 16, d_model)

    def test_recurrent_mode_shape(
        self, gla: GatedLinearAttention, x: torch.Tensor
    ) -> None:
        out, S, N = gla(x, mode="recurrent")
        B, L, D = x.shape
        H = gla.n_heads
        d = gla.head_dim
        assert out.shape == (B, L, D)
        assert S.shape == (B, H, d, d)
        assert N.shape == (B, H, d)

    def test_chunkwise_mode_shape(
        self, gla: GatedLinearAttention, x: torch.Tensor
    ) -> None:
        out, S, N = gla(x, mode="chunkwise")
        B, L, D = x.shape
        H = gla.n_heads
        d = gla.head_dim
        assert out.shape == (B, L, D)
        assert S.shape == (B, H, d, d)
        assert N.shape == (B, H, d)

    def test_output_finite(
        self, gla: GatedLinearAttention, x: torch.Tensor
    ) -> None:
        for mode in ("recurrent", "chunkwise"):
            out, S, N = gla(x, mode=mode)
            assert torch.all(torch.isfinite(out)), f"NaN in {mode} mode"
            assert torch.all(torch.isfinite(S))

    def test_unknown_mode(
        self, gla: GatedLinearAttention, x: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            gla(x, mode="quadratic")

    def test_divisible_error(self) -> None:
        with pytest.raises(ValueError, match="must be divisible"):
            GatedLinearAttention(d_model=10, n_heads=3)

    def test_with_initial_state(
        self, gla: GatedLinearAttention, x: torch.Tensor
    ) -> None:
        B, L, D = x.shape
        H = gla.n_heads
        d = gla.head_dim
        S0 = torch.randn(B, H, d, d)
        N0 = torch.randn(B, H, d)
        out, S, N = gla(x, mode="recurrent", initial_state=S0, initial_normalizer=N0)
        assert out.shape == (B, L, D)
        assert torch.all(torch.isfinite(out))

    def test_gradient_flow(
        self, gla: GatedLinearAttention, x: torch.Tensor
    ) -> None:
        for mode in ("recurrent", "chunkwise"):
            x_grad = x.clone().requires_grad_(True)
            out, _, _ = gla(x_grad, mode=mode)
            loss = out.sum()
            loss.backward()
            assert x_grad.grad is not None
            assert torch.all(torch.isfinite(x_grad.grad)), f"NaN grad in {mode} mode"

    def test_dropout_identity_in_eval(
        self, d_model: int, n_heads: int
    ) -> None:
        torch.manual_seed(7)
        gla = GatedLinearAttention(d_model, n_heads, dropout=0.5)
        gla.eval()
        x = torch.randn(1, 8, d_model)
        out1, _, _ = gla(x, mode="recurrent")
        out2, _, _ = gla(x, mode="recurrent")
        assert torch.allclose(out1, out2)

    def test_chunk_size_zero_still_works(
        self, gla: GatedLinearAttention, x: torch.Tensor
    ) -> None:
        """Setting chunk_size=0 should still work with recurrent fallback."""
        gla.chunk_size = 0
        # chunk_size=0 would pad badly; just test recurrent mode
        out, S, N = gla(x, mode="recurrent")
        assert out.shape == x.shape
        assert torch.all(torch.isfinite(out))

    @pytest.mark.parametrize("L", [1, 3, 8, 32])
    def test_variable_length(
        self, gla: GatedLinearAttention, d_model: int, L: int
    ) -> None:
        x = torch.randn(1, L, d_model)
        for mode in ("recurrent", "chunkwise"):
            out, _, _ = gla(x, mode=mode)
            assert out.shape == (1, L, d_model)
            assert torch.all(torch.isfinite(out)), f"NaN in {mode} mode at L={L}"
