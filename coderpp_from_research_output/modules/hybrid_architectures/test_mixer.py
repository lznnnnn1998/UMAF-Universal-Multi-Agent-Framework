"""Tests for the unified sequence mixer module."""

from __future__ import annotations

import pytest
import torch

from hybrid_architectures.mixer import (
    AttentionMixer,
    HybridMixer,
    KernelFusionPath,
    MixerMode,
    SSMMixer,
)


# ---------------------------------------------------------------------------
# SSMMixer
# ---------------------------------------------------------------------------

class TestSSMMixer:
    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def mixer(self, d_model: int) -> SSMMixer:
        return SSMMixer(d_model=d_model, d_state=8)

    @pytest.fixture
    def x(self, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(2, 10, d_model)

    def test_recurrent_mode(
        self, mixer: SSMMixer, x: torch.Tensor
    ) -> None:
        y = mixer(x, mode=MixerMode.RECURRENT)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_hybrid_mode(
        self, mixer: SSMMixer, x: torch.Tensor
    ) -> None:
        y = mixer(x, mode=MixerMode.HYBRID)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_attention_raises(
        self, mixer: SSMMixer, x: torch.Tensor
    ) -> None:
        with pytest.raises(NotImplementedError, match="does not support attention"):
            mixer(x, mode=MixerMode.ATTENTION)

    def test_string_mode(
        self, mixer: SSMMixer, x: torch.Tensor
    ) -> None:
        y = mixer(x, mode="recurrent")
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_gradient_flow(
        self, mixer: SSMMixer, d_model: int
    ) -> None:
        x = torch.randn(1, 8, d_model, requires_grad=True)
        y = mixer(x, mode=MixerMode.HYBRID)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


# ---------------------------------------------------------------------------
# AttentionMixer
# ---------------------------------------------------------------------------

class TestAttentionMixer:
    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def mixer(self, d_model: int) -> AttentionMixer:
        return AttentionMixer(d_model=d_model, n_heads=4, window_size=8)

    @pytest.fixture
    def x(self, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(2, 10, d_model)

    def test_attention_mode(
        self, mixer: AttentionMixer, x: torch.Tensor
    ) -> None:
        y = mixer(x, mode=MixerMode.ATTENTION)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_hybrid_mode(
        self, mixer: AttentionMixer, x: torch.Tensor
    ) -> None:
        y = mixer(x, mode=MixerMode.HYBRID)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_recurrent_raises(
        self, mixer: AttentionMixer, x: torch.Tensor
    ) -> None:
        with pytest.raises(NotImplementedError, match="does not support recurrent"):
            mixer(x, mode=MixerMode.RECURRENT)

    def test_gradient_flow(
        self, mixer: AttentionMixer, d_model: int
    ) -> None:
        x = torch.randn(1, 8, d_model, requires_grad=True)
        y = mixer(x, mode=MixerMode.HYBRID)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestMixerMode:
    def test_from_string(self) -> None:
        assert MixerMode("attention") == MixerMode.ATTENTION
        assert MixerMode("recurrent") == MixerMode.RECURRENT
        assert MixerMode("hybrid") == MixerMode.HYBRID

    def test_value(self) -> None:
        assert MixerMode.ATTENTION.value == "attention"
        assert MixerMode.RECURRENT.value == "recurrent"
        assert MixerMode.HYBRID.value == "hybrid"


class TestKernelFusionPath:
    def test_from_string(self) -> None:
        assert KernelFusionPath("serial") == KernelFusionPath.SERIAL
        assert KernelFusionPath("parallel") == KernelFusionPath.PARALLEL
        assert KernelFusionPath("interleaved") == KernelFusionPath.INTERLEAVED


# ---------------------------------------------------------------------------
# HybridMixer
# ---------------------------------------------------------------------------

class TestHybridMixer:
    """Tests for the configurable HybridMixer across all fusion paths."""

    @pytest.fixture
    def d_model(self) -> int:
        return 16

    @pytest.fixture
    def x(self, d_model: int) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(2, 10, d_model)

    # ---- Serial ----

    def test_serial_hybrid(
        self, d_model: int, x: torch.Tensor
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.SERIAL)
        y = mixer(x, mode=MixerMode.HYBRID)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_serial_attention(
        self, d_model: int, x: torch.Tensor
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.SERIAL)
        y = mixer(x, mode=MixerMode.ATTENTION)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_serial_recurrent(
        self, d_model: int, x: torch.Tensor
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.SERIAL)
        y = mixer(x, mode=MixerMode.RECURRENT)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    # ---- Parallel ----

    def test_parallel_hybrid(
        self, d_model: int, x: torch.Tensor
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.PARALLEL)
        y = mixer(x, mode=MixerMode.HYBRID)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_parallel_gradient(
        self, d_model: int
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.PARALLEL)
        x = torch.randn(1, 8, d_model, requires_grad=True)
        y = mixer(x, mode=MixerMode.HYBRID)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    # ---- Interleaved ----

    def test_interleaved_hybrid(
        self, d_model: int, x: torch.Tensor
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.INTERLEAVED,
                            num_blocks=2)
        y = mixer(x, mode=MixerMode.HYBRID)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_interleaved_gradient(
        self, d_model: int
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.INTERLEAVED,
                            num_blocks=3)
        x = torch.randn(1, 8, d_model, requires_grad=True)
        y = mixer(x, mode=MixerMode.HYBRID)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_interleaved_multiple_blocks(
        self, d_model: int
    ) -> None:
        """Verify it works with different numbers of blocks."""
        for nb in (1, 2, 4):
            mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                                fusion_path=KernelFusionPath.INTERLEAVED,
                                num_blocks=nb)
            x = torch.randn(2, 16, d_model)
            y = mixer(x, mode=MixerMode.HYBRID)
            assert y.shape == x.shape
            assert torch.all(torch.isfinite(y))

    # ---- String fusion_path ----

    def test_string_fusion_path(
        self, d_model: int, x: torch.Tensor
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path="interleaved")
        y = mixer(x, mode=MixerMode.HYBRID)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    # ---- Supported modes ----

    def test_supported_modes(
        self, d_model: int
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.SERIAL)
        modes = mixer.supported_modes
        assert MixerMode.ATTENTION in modes
        assert MixerMode.RECURRENT in modes
        assert MixerMode.HYBRID in modes

    # ---- Default window ----

    def test_default_window(
        self, d_model: int
    ) -> None:
        """None window_size → defaults to 128."""
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.SERIAL,
                            window_size=None)
        x = torch.randn(2, 10, d_model)
        y = mixer(x, mode=MixerMode.HYBRID)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    # ---- Unknown mode ----

    def test_unknown_mode(
        self, d_model: int
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8)
        x = torch.randn(2, 10, d_model)
        with pytest.raises(ValueError, match="transformer"):
            mixer(x, mode="transformer")

    @pytest.mark.parametrize("L", [1, 3, 16, 33])
    def test_variable_length(
        self, d_model: int, L: int
    ) -> None:
        mixer = HybridMixer(d_model, n_heads=4, d_state=8,
                            fusion_path=KernelFusionPath.SERIAL)
        x = torch.randn(1, L, d_model)
        y = mixer(x, mode=MixerMode.HYBRID)
        assert y.shape == (1, L, d_model)
        assert torch.all(torch.isfinite(y))
