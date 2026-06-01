"""Tests for the SSM module."""

from __future__ import annotations

import pytest
import torch

from hybrid_architectures.ssm import DiagonalSSM


class TestDiagonalSSM:
    """Tests for DiagonalSSM (S4D-style diagonal state space model)."""

    @pytest.fixture
    def batch_size(self) -> int:
        return 2

    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def d_state(self) -> int:
        return 8

    @pytest.fixture
    def seq_len(self) -> int:
        return 32

    @pytest.fixture
    def ssm(self, d_state: int, d_model: int) -> DiagonalSSM:
        return DiagonalSSM(d_state=d_state, d_model=d_model)

    @pytest.fixture
    def x(self, batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(batch_size, seq_len, d_model)

    # ---- Construction ----

    def test_construction(self, ssm: DiagonalSSM, d_state: int, d_model: int) -> None:
        """Verify basic model dimensions and parameter shapes."""
        assert ssm.d_state == d_state
        assert ssm.d_model == d_model
        assert ssm.A_log.shape == (d_state,)  # pyright: ignore[reportUnknownMemberType]
        assert ssm.B.shape == (d_model, d_state)
        assert ssm.C.shape == (d_model, d_state)
        assert ssm.D.shape == (d_model,)
        assert ssm.log_dt.shape == (d_model,)

    # ---- Convolution mode ----

    def test_convolution_mode_shape(
        self, ssm: DiagonalSSM, x: torch.Tensor, batch_size: int, seq_len: int, d_model: int
    ) -> None:
        """Convolution mode should preserve the input shape."""
        y = ssm(x, mode="convolution")
        assert y.shape == (batch_size, seq_len, d_model)

    def test_convolution_mode_no_nan(
        self, ssm: DiagonalSSM, x: torch.Tensor
    ) -> None:
        """Convolution mode output must be finite."""
        y = ssm(x, mode="convolution")
        assert torch.all(torch.isfinite(y))

    def test_convolution_deterministic(
        self, ssm: DiagonalSSM, x: torch.Tensor
    ) -> None:
        """Same input → same output (no randomness at inference)."""
        y1 = ssm(x, mode="convolution")
        y2 = ssm(x, mode="convolution")
        assert torch.allclose(y1, y2)

    # ---- Recurrent mode ----

    def test_recurrent_mode_shape(
        self, ssm: DiagonalSSM, x: torch.Tensor, batch_size: int, seq_len: int, d_model: int
    ) -> None:
        """Recurrent mode returns (output, final_state)."""
        y, state = ssm(x, mode="recurrent")
        assert y.shape == (batch_size, seq_len, d_model)
        assert state.shape == (batch_size, d_model, ssm.d_state)

    def test_recurrent_vs_convolution_close(
        self, d_state: int, d_model: int
    ) -> None:
        """Recurrent and convolution modes should produce very close results
        for a small, fixed setup."""
        torch.manual_seed(123)
        ssm = DiagonalSSM(d_state=d_state, d_model=d_model)
        x = torch.randn(1, 16, d_model)

        y_conv = ssm(x, mode="convolution")
        y_rec, _ = ssm(x, mode="recurrent")

        # Recurrent introduces more numerical drift; tolerance is relaxed
        assert torch.allclose(y_conv, y_rec, rtol=5e-3, atol=5e-3)

    # ---- Step (autoregressive inference) ----

    def test_step_single(
        self, ssm: DiagonalSSM, batch_size: int, d_model: int
    ) -> None:
        """Single step produces correct shapes."""
        state = ssm.reset_state(batch_size)
        u_t = torch.randn(batch_size, d_model)
        y_t, new_state = ssm.step(u_t, state)
        assert y_t.shape == (batch_size, d_model)
        assert new_state.shape == state.shape

    def test_step_matches_recurrent(
        self, d_state: int, d_model: int
    ) -> None:
        """Stepping token-by-token should match recurrent mode output."""
        torch.manual_seed(77)
        ssm = DiagonalSSM(d_state=d_state, d_model=d_model)
        B, L = 1, 20
        x = torch.randn(B, L, d_model)

        y_rec, final_rec = ssm(x, mode="recurrent")

        # Manual step
        state = ssm.reset_state(B)
        y_step_list = []
        for t in range(L):
            y_t, state = ssm.step(x[:, t, :], state)
            y_step_list.append(y_t)
        y_step = torch.stack(y_step_list, dim=1)

        assert torch.allclose(y_rec, y_step, rtol=1e-5, atol=1e-5)
        assert torch.allclose(final_rec, state, rtol=1e-5, atol=1e-5)

    # ---- State management ----

    def test_reset_state_default(
        self, ssm: DiagonalSSM, batch_size: int
    ) -> None:
        """reset_state returns a zero tensor with correct shape."""
        state = ssm.reset_state(batch_size)
        assert state.shape == (batch_size, ssm.d_model, ssm.d_state)
        assert torch.all(state == 0.0)

    def test_reset_state_explicit_device(
        self, ssm: DiagonalSSM
    ) -> None:
        """Reset state on a specific device."""
        state = ssm.reset_state(3, device=torch.device("cpu"))
        assert state.shape == (3, ssm.d_model, ssm.d_state)
        assert state.device.type == "cpu"

    # ---- Discretisation ----

    def test_discretize_output_shapes(
        self, ssm: DiagonalSSM, d_model: int, d_state: int
    ) -> None:
        """A_bar and B_bar have correct shapes after discretisation."""
        A_bar, B_bar = ssm._discretize()
        assert A_bar.shape == (d_model, d_state)
        assert B_bar.shape == (d_model, d_state)

    def test_discretize_A_bar_lt_one(
        self, ssm: DiagonalSSM
    ) -> None:
        """Discrete diagonal A_bar entries should be strictly < 1 for
        stability when dt > 0 and A < 0 (bilinear transform)."""
        A_bar, _ = ssm._discretize()
        assert torch.all(A_bar < 1.0)

    # ---- Convolution kernel ----

    def test_compute_kernel_shape(
        self, ssm: DiagonalSSM, d_model: int, seq_len: int
    ) -> None:
        """Convolution kernel has shape (d_model, seq_len)."""
        A_bar, B_bar = ssm._discretize()
        K = ssm._compute_kernel(A_bar, B_bar, seq_len)
        assert K.shape == (d_model, seq_len)

    # ---- Error paths ----

    def test_unknown_mode_raises(self, ssm: DiagonalSSM, x: torch.Tensor) -> None:
        """Passing an unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            ssm(x, mode="transformer")

    # ---- Gradient flow ----

    def test_convolution_backward(
        self, ssm: DiagonalSSM, x: torch.Tensor
    ) -> None:
        """Gradients should flow through the convolution mode."""
        x_grad = x.clone().requires_grad_(True)
        y = ssm(x_grad, mode="convolution")
        loss = y.sum()
        loss.backward()
        assert x_grad.grad is not None
        assert torch.all(torch.isfinite(x_grad.grad))

    def test_recurrent_backward(
        self, ssm: DiagonalSSM, x: torch.Tensor
    ) -> None:
        """Gradients should flow through the recurrent mode."""
        x_grad = x.clone().requires_grad_(True)
        y, _ = ssm(x_grad, mode="recurrent")
        loss = y.sum()
        loss.backward()
        assert x_grad.grad is not None
        assert torch.all(torch.isfinite(x_grad.grad))

    # ---- Custom dt range ----

    def test_custom_dt_range(self) -> None:
        """SSM with a very narrow dt range should still work."""
        ssm = DiagonalSSM(d_state=4, d_model=8, dt_min=0.1, dt_max=0.1)
        x = torch.randn(1, 8, 8)
        y = ssm(x, mode="convolution")
        assert torch.all(torch.isfinite(y))
