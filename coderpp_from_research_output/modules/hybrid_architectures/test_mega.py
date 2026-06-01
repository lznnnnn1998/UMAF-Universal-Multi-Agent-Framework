"""Tests for the Mega module."""

from __future__ import annotations

import pytest
import torch

from hybrid_architectures.mega import (
    ExponentialMovingAverage,
    MegaGatedAttention,
    MegaLayer,
)


class TestExponentialMovingAverage:
    """Tests for the ExponentialMovingAverage module."""

    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def ema(self, d_model: int) -> ExponentialMovingAverage:
        return ExponentialMovingAverage(d_model=d_model)

    @pytest.fixture
    def x(self, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(2, 10, d_model)

    def test_output_shape(
        self, ema: ExponentialMovingAverage, x: torch.Tensor
    ) -> None:
        y = ema(x)
        assert y.shape == x.shape

    def test_output_finite(
        self, ema: ExponentialMovingAverage, x: torch.Tensor
    ) -> None:
        y = ema(x)
        assert torch.all(torch.isfinite(y))

    def test_causal(self, ema: ExponentialMovingAverage, d_model: int) -> None:
        """EMA at position t should not depend on positions > t."""
        torch.manual_seed(42)
        # Zero out positions > 2
        x = torch.randn(1, 5, d_model)
        x_zero = x.clone()
        x_zero[:, 3:] = 0.0

        y = ema(x)         # full
        y_zero = ema(x_zero)  # positions 3+ zeroed

        # First 3 positions should be identical
        assert torch.allclose(y[:, :3], y_zero[:, :3], rtol=1e-5, atol=1e-5)

    def test_bidirectional(
        self, d_model: int
    ) -> None:
        ema = ExponentialMovingAverage(d_model=d_model, bidirectional=True)
        x = torch.randn(2, 10, d_model)
        y = ema(x)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_bidirectional_different_from_forward(
        self, d_model: int
    ) -> None:
        """Bidirectional EMA should differ from forward-only for non-trivial inputs."""
        ema_fwd = ExponentialMovingAverage(d_model=d_model, bidirectional=False)
        ema_bi = ExponentialMovingAverage(d_model=d_model, bidirectional=True)

        x = torch.randn(2, 10, d_model)
        y_fwd = ema_fwd(x)
        y_bi = ema_bi(x)

        assert y_fwd.shape == y_bi.shape
        # They should NOT be equal for random inputs
        assert not torch.allclose(y_fwd, y_bi)

    def test_gradient_flow(
        self, ema: ExponentialMovingAverage, x: torch.Tensor
    ) -> None:
        x_grad = x.clone().requires_grad_(True)
        y = ema(x_grad)
        loss = y.sum()
        loss.backward()
        assert x_grad.grad is not None
        assert torch.all(torch.isfinite(x_grad.grad))

    def test_parameters_learnable(self, d_model: int) -> None:
        ema = ExponentialMovingAverage(d_model=d_model)
        params = list(ema.parameters())
        assert len(params) > 0
        for p in params:
            assert p.requires_grad

    @pytest.mark.parametrize("L", [1, 3, 16])
    def test_variable_length(
        self, ema: ExponentialMovingAverage, d_model: int, L: int
    ) -> None:
        x = torch.randn(1, L, d_model)
        y = ema(x)
        assert y.shape == (1, L, d_model)
        assert torch.all(torch.isfinite(y))


class TestMegaGatedAttention:
    """Tests for the MegaGatedAttention module."""

    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def ga(self, d_model: int) -> MegaGatedAttention:
        return MegaGatedAttention(d_model=d_model)

    @pytest.fixture
    def x(self, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(2, 10, d_model)

    def test_output_shape(
        self, ga: MegaGatedAttention, x: torch.Tensor
    ) -> None:
        y = ga(x)
        assert y.shape == x.shape

    def test_output_finite(
        self, ga: MegaGatedAttention, x: torch.Tensor
    ) -> None:
        y = ga(x)
        assert torch.all(torch.isfinite(y))

    def test_causal(
        self, ga: MegaGatedAttention, d_model: int
    ) -> None:
        """Output at position t should only depend on x_{≤t}."""
        torch.manual_seed(42)
        x = torch.randn(1, 4, d_model, requires_grad=True)
        y = ga(x)
        # Gradient of y[0, 2] w.r.t. x[0, 3] should be zero (causal)
        loss = y[0, 2].sum()
        loss.backward()
        assert x.grad is not None
        grad_from_future = x.grad[0, 3]  # type: ignore[index]
        assert torch.allclose(grad_from_future, torch.zeros_like(grad_from_future))

    def test_gradient_flow(
        self, ga: MegaGatedAttention, x: torch.Tensor
    ) -> None:
        x_grad = x.clone().requires_grad_(True)
        y = ga(x_grad)
        loss = y.sum()
        loss.backward()
        assert x_grad.grad is not None
        assert torch.all(torch.isfinite(x_grad.grad))

    def test_dropout_identity_in_eval(
        self, d_model: int
    ) -> None:
        torch.manual_seed(7)
        ga = MegaGatedAttention(d_model=d_model, dropout=0.5)
        ga.eval()
        x = torch.randn(1, 8, d_model)
        y1 = ga(x)
        y2 = ga(x)
        assert torch.allclose(y1, y2)

    def test_with_mask(
        self, ga: MegaGatedAttention, d_model: int
    ) -> None:
        x = torch.randn(1, 6, d_model)
        # Regular forward should not NaN
        y = ga(x, mask=None)
        assert torch.all(torch.isfinite(y))

    @pytest.mark.parametrize("L", [1, 3, 16])
    def test_variable_length(
        self, ga: MegaGatedAttention, d_model: int, L: int
    ) -> None:
        x = torch.randn(1, L, d_model)
        y = ga(x)
        assert y.shape == (1, L, d_model)
        assert torch.all(torch.isfinite(y))


class TestMegaLayer:
    """Tests for the full MegaLayer (EMA + Gated Attention)."""

    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def mega(self, d_model: int) -> MegaLayer:
        return MegaLayer(d_model=d_model)

    @pytest.fixture
    def x(self, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(2, 10, d_model)

    def test_output_shape(
        self, mega: MegaLayer, x: torch.Tensor
    ) -> None:
        y = mega(x)
        assert y.shape == x.shape

    def test_output_finite(
        self, mega: MegaLayer, x: torch.Tensor
    ) -> None:
        y = mega(x)
        assert torch.all(torch.isfinite(y))

    def test_gradient_flow(
        self, mega: MegaLayer, x: torch.Tensor
    ) -> None:
        x_grad = x.clone().requires_grad_(True)
        y = mega(x_grad)
        loss = y.sum()
        loss.backward()
        assert x_grad.grad is not None
        assert torch.all(torch.isfinite(x_grad.grad))

    def test_bidirectional(
        self, d_model: int
    ) -> None:
        mega = MegaLayer(d_model=d_model, bidirectional_ema=True)
        x = torch.randn(2, 8, d_model)
        y = mega(x)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_dropout_identity_in_eval(
        self, d_model: int
    ) -> None:
        torch.manual_seed(7)
        mega = MegaLayer(d_model=d_model, dropout=0.5)
        mega.eval()
        x = torch.randn(1, 8, d_model)
        y1 = mega(x)
        y2 = mega(x)
        assert torch.allclose(y1, y2)

    @pytest.mark.parametrize("L", [1, 3, 16])
    def test_variable_length(
        self, mega: MegaLayer, d_model: int, L: int
    ) -> None:
        x = torch.randn(1, L, d_model)
        y = mega(x)
        assert y.shape == (1, L, d_model)
        assert torch.all(torch.isfinite(y))
