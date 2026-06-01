"""Tests for the H3 (Hungry Hungry Hippos) module."""

from __future__ import annotations

import pytest
import torch

from hybrid_architectures.h3 import H3Layer


class TestH3Layer:
    """Tests for H3Layer combining shift SSM and diagonal SSM with gating."""

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
        return 20

    @pytest.fixture
    def h3(self, d_model: int, d_state: int) -> H3Layer:
        torch.manual_seed(42)
        return H3Layer(d_model=d_model, d_state=d_state)

    @pytest.fixture
    def x(self, batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(batch_size, seq_len, d_model)

    # ---- Construction ----

    def test_construction(self, h3: H3Layer, d_model: int) -> None:
        assert h3.d_model == d_model
        assert hasattr(h3, "shift_ssm")
        assert hasattr(h3, "diag_ssm_k")
        assert hasattr(h3, "diag_ssm_v")

    # ---- Convolution mode ----

    def test_convolution_shape(
        self, h3: H3Layer, x: torch.Tensor
    ) -> None:
        y = h3(x, mode="convolution")
        assert y.shape == x.shape

    def test_convolution_finite(
        self, h3: H3Layer, x: torch.Tensor
    ) -> None:
        y = h3(x, mode="convolution")
        assert torch.all(torch.isfinite(y))

    # ---- Recurrent mode ----

    def test_recurrent_shape(
        self, h3: H3Layer, x: torch.Tensor
    ) -> None:
        y = h3(x, mode="recurrent")
        assert y.shape == x.shape

    def test_recurrent_finite(
        self, h3: H3Layer, x: torch.Tensor
    ) -> None:
        y = h3(x, mode="recurrent")
        assert torch.all(torch.isfinite(y))

    # ---- Step (autoregressive) ----

    def test_step_single(
        self, h3: H3Layer, batch_size: int, d_model: int
    ) -> None:
        states = h3.reset_states(batch_size)
        x_t = torch.randn(batch_size, d_model)
        y_t, new_states = h3.step(x_t, *states)
        assert y_t.shape == (batch_size, d_model)
        assert len(new_states) == 3
        for ns in new_states:
            assert ns.shape == torch.Size((batch_size, d_model, h3.shift_ssm.d_state))

    def test_step_matches_recurrent(
        self, h3: H3Layer, d_model: int
    ) -> None:
        """Manual stepping should match recurrent mode."""
        torch.manual_seed(99)
        B, L = 1, 12
        x = torch.randn(B, L, d_model)

        y_rec = h3(x, mode="recurrent")

        states = h3.reset_states(B)
        y_step_list = []
        for t in range(L):
            y_t, states = h3.step(x[:, t, :], *states)
            y_step_list.append(y_t)
        y_step = torch.stack(y_step_list, dim=1)

        assert torch.allclose(y_rec, y_step, rtol=1e-4, atol=1e-4)

    # ---- State management ----

    def test_reset_states_shape(
        self, h3: H3Layer, batch_size: int
    ) -> None:
        states = h3.reset_states(batch_size)
        assert len(states) == 3
        for s in states:
            assert s.shape == (batch_size, h3.d_model, h3.shift_ssm.d_state)
            assert torch.all(s == 0.0)

    # ---- Gradient flow ----

    def test_convolution_backward(
        self, h3: H3Layer, x: torch.Tensor
    ) -> None:
        x_grad = x.clone().requires_grad_(True)
        y = h3(x_grad, mode="convolution")
        loss = y.sum()
        loss.backward()
        assert x_grad.grad is not None
        assert torch.all(torch.isfinite(x_grad.grad))

    def test_recurrent_backward(
        self, h3: H3Layer, d_model: int
    ) -> None:
        torch.manual_seed(55)
        x = torch.randn(1, 8, d_model, requires_grad=True)
        y = h3(x, mode="recurrent")
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    # ---- Dropout ----

    def test_dropout_eval(
        self, d_model: int, d_state: int
    ) -> None:
        """Dropout should be no-op in eval mode."""
        torch.manual_seed(7)
        h3 = H3Layer(d_model=d_model, d_state=d_state, dropout=0.5)
        h3.eval()
        x = torch.randn(1, 8, d_model)
        y1 = h3(x)
        y2 = h3(x)
        assert torch.allclose(y1, y2)

    # ---- Various sequence lengths ----

    @pytest.mark.parametrize("L", [1, 3, 16, 32])
    def test_variable_sequence_length(
        self, h3: H3Layer, d_model: int, L: int
    ) -> None:
        x = torch.randn(1, L, d_model)
        y = h3(x, mode="convolution")
        assert y.shape == (1, L, d_model)
        assert torch.all(torch.isfinite(y))
