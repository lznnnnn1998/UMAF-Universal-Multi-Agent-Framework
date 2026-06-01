"""Tests for the attention module."""

from __future__ import annotations

import pytest
import torch

from hybrid_architectures.attention import (
    LinearAttention,
    SlidingWindowAttention,
    _causal_sliding_mask,
)


# ---------------------------------------------------------------------------
# Causal sliding mask
# ---------------------------------------------------------------------------

class TestCausalSlidingMask:
    """Tests for the _causal_sliding_mask helper."""

    def test_square_shape(self) -> None:
        mask = _causal_sliding_mask(10, 5)
        assert mask.shape == (10, 10)

    def test_causal(self) -> None:
        """Position i may NOT attend to j > i (strictly upper triangular)."""
        mask = _causal_sliding_mask(8, 3)
        for i in range(8):
            for j in range(i + 1, 8):
                assert not mask[i, j].item(), f"position {i} attended to future {j}"

    def test_self_attention(self) -> None:
        """Position i must always attend to itself."""
        mask = _causal_sliding_mask(8, 2)
        for i in range(8):
            assert mask[i, i].item()

    def test_window_boundary(self) -> None:
        """Position i attends exactly to the last window_size positions."""
        L, W = 10, 3
        mask = _causal_sliding_mask(L, W)
        for i in range(L):
            earliest = max(0, i - W + 1)
            for j in range(L):
                expected = earliest <= j <= i
                assert mask[i, j].item() == expected, f"({i},{j}): got {mask[i,j]} != {expected}"

    def test_window_larger_than_seq(self) -> None:
        """When window ≥ seq_len all causal pairs should be True."""
        mask = _causal_sliding_mask(5, 10)
        for i in range(5):
            for j in range(5):
                if j <= i:
                    assert mask[i, j].item()
                else:
                    assert not mask[i, j].item()

    def test_window_one(self) -> None:
        """Window size 1 → only diagonal is True."""
        mask = _causal_sliding_mask(6, 1)
        for i in range(6):
            for j in range(6):
                assert mask[i, j].item() == (i == j)


# ---------------------------------------------------------------------------
# Sliding Window Attention
# ---------------------------------------------------------------------------

class TestSlidingWindowAttention:
    """Tests for SlidingWindowAttention."""

    @pytest.fixture
    def batch_size(self) -> int:
        return 2

    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def n_heads(self) -> int:
        return 4

    @pytest.fixture
    def window_size(self) -> int:
        return 8

    @pytest.fixture
    def seq_len(self) -> int:
        return 20

    @pytest.fixture
    def attn(
        self, d_model: int, n_heads: int, window_size: int
    ) -> SlidingWindowAttention:
        return SlidingWindowAttention(
            d_model=d_model, n_heads=n_heads, window_size=window_size
        )

    @pytest.fixture
    def x(self, batch_size: int, seq_len: int, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(batch_size, seq_len, d_model)

    def test_output_shape(
        self, attn: SlidingWindowAttention, x: torch.Tensor, batch_size: int, seq_len: int, d_model: int
    ) -> None:
        y = attn(x)
        assert y.shape == (batch_size, seq_len, d_model)

    def test_output_finite(self, attn: SlidingWindowAttention, x: torch.Tensor) -> None:
        y = attn(x)
        assert torch.all(torch.isfinite(y))

    def test_with_extra_mask(
        self, attn: SlidingWindowAttention, x: torch.Tensor
    ) -> None:
        """Extra mask should combine with the window mask without NaN."""
        B, L, _ = x.shape
        # Mask out position 0 attending to position 1 if within window.
        # (Masking ALL positions produces all -inf softmax → NaN.)
        extra = torch.zeros(L, L)
        extra[0, 1] = float("-inf")
        y = attn(x, mask=extra)
        assert torch.all(torch.isfinite(y))

    def test_divisible_error(self) -> None:
        """d_model not divisible by n_heads → ValueError."""
        with pytest.raises(ValueError, match="must be divisible"):
            SlidingWindowAttention(d_model=10, n_heads=3, window_size=4)

    def test_gradient_flow(
        self, attn: SlidingWindowAttention, d_model: int
    ) -> None:
        torch.manual_seed(99)
        x = torch.randn(2, 10, d_model, requires_grad=True)
        y = attn(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_no_window_leakage(
        self, d_model: int
    ) -> None:
        """When window=1, output at position t should only depend on x_t."""
        torch.manual_seed(42)
        attn = SlidingWindowAttention(d_model=d_model, n_heads=4, window_size=1)
        x = torch.randn(1, 5, d_model, requires_grad=True)
        y = attn(x)
        # Gradient of y[:, 4] w.r.t. x[:, 0] should be zero (window=1)
        loss = y[:, 4].sum()
        loss.backward()
        grad_due_to_x0 = x.grad[:, 0, :]  # type: ignore[index]
        assert grad_due_to_x0 is not None
        assert torch.allclose(grad_due_to_x0, torch.zeros_like(grad_due_to_x0))

    def test_dropout_identity_in_eval(
        self, d_model: int, n_heads: int, window_size: int
    ) -> None:
        """In eval mode, dropout should not change the output."""
        torch.manual_seed(7)
        attn = SlidingWindowAttention(d_model, n_heads, window_size, dropout=0.5)
        attn.eval()
        x = torch.randn(1, 10, d_model)
        y1 = attn(x)
        y2 = attn(x)
        assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Linear Attention
# ---------------------------------------------------------------------------

class TestLinearAttention:
    """Tests for LinearAttention."""

    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def n_heads(self) -> int:
        return 4

    @pytest.fixture
    def la(self, d_model: int, n_heads: int) -> LinearAttention:
        return LinearAttention(d_model=d_model, n_heads=n_heads)

    @pytest.fixture
    def x(self, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(2, 10, d_model)

    def test_output_shape(
        self, la: LinearAttention, x: torch.Tensor
    ) -> None:
        y = la(x)
        assert y.shape == x.shape

    def test_output_finite(
        self, la: LinearAttention, x: torch.Tensor
    ) -> None:
        y = la(x)
        assert torch.all(torch.isfinite(y))

    def test_divisible_error(self) -> None:
        with pytest.raises(ValueError, match="must be divisible"):
            LinearAttention(d_model=10, n_heads=3)

    def test_relu_feature_map(self) -> None:
        la = LinearAttention(d_model=16, n_heads=4, feature_map="relu")
        x = torch.randn(2, 8, 16)
        y = la(x)
        assert torch.all(torch.isfinite(y))
        assert y.shape == x.shape

    def test_unknown_feature_map(self) -> None:
        la = LinearAttention(d_model=16, n_heads=4, feature_map="unknown")
        x = torch.randn(2, 8, 16)
        with pytest.raises(ValueError, match="Unknown feature map"):
            la(x)

    def test_gradient_flow(
        self, la: LinearAttention, x: torch.Tensor
    ) -> None:
        x_grad = x.clone().requires_grad_(True)
        y = la(x_grad)
        loss = y.sum()
        loss.backward()
        assert x_grad.grad is not None
        assert torch.all(torch.isfinite(x_grad.grad))

    def test_feature_map_is_positive(self, la: LinearAttention) -> None:
        """ELU + 1 should produce strictly positive values."""
        x = torch.randn(2, 8, 16)
        qkv = la.qkv(x)
        # Just check the feature map on an arbitrary projection
        positive = LinearAttention._elu_feature_map(torch.randn(10))
        assert torch.all(positive > 0)
