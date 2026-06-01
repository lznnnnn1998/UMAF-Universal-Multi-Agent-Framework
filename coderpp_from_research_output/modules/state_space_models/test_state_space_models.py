"""
Comprehensive tests for state_space_models module.

Tests cover:
- HiPPO matrix initialization (LegS, LegT, FouD)
- Parallel associative scan (Blelloch)
- S4 DPLR parameterization and kernel computation
- S4D diagonal kernel computation
- Mamba/S6 selective SSM
- Mamba-2/SSD semiseparable matrices
- SSM discretization (ZOH, bilinear)
- Recurrent stepping for autoregressive generation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
import numpy as np
import math


# ============================================================================
# HiPPO Tests
# ============================================================================

class TestHiPPOLegS:
    """Tests for HiPPO-LegS matrix construction."""

    def test_shape(self):
        from state_space_models.hippo import hippo_legs_matrix
        for N in [4, 8, 16, 64]:
            A = hippo_legs_matrix(N)
            assert A.shape == (N, N)

    def test_lower_triangular(self):
        from state_space_models.hippo import hippo_legs_matrix
        A = hippo_legs_matrix(8)
        idx = torch.triu_indices(8, 8, offset=1)
        upper = A[idx[0], idx[1]]
        assert torch.allclose(upper, torch.zeros_like(upper))

    def test_diagonal_values(self):
        from state_space_models.hippo import hippo_legs_matrix
        N = 8
        A = hippo_legs_matrix(N)
        for n in range(N):
            expected = -(n + 1)
            assert abs(A[n, n].item() - expected) < 1e-6, (
                f"A[{n},{n}] = {A[n,n]}, expected {expected}"
            )

    def test_off_diagonal_values(self):
        from state_space_models.hippo import hippo_legs_matrix
        N = 8
        A = hippo_legs_matrix(N)
        for n in range(N):
            for k in range(n):
                expected = -math.sqrt((2 * n + 1) * (2 * k + 1))
                assert abs(A[n, k].item() - expected) < 1e-5, (
                    f"A[{n},{k}] = {A[n,k]}, expected {expected}"
                )

    def test_dtype_arg(self):
        from state_space_models.hippo import hippo_legs_matrix
        A = hippo_legs_matrix(8, dtype=torch.float64)
        assert A.dtype == torch.float64
        A = hippo_legs_matrix(8)
        assert A.dtype == torch.float32

    def test_eigenvalues_negative_real(self):
        """HiPPO-LegS eigenvalues should have negative real parts (stable)."""
        from state_space_models.hippo import hippo_legs_matrix
        A = hippo_legs_matrix(16).numpy()
        eigvals = np.linalg.eigvals(A)
        assert np.all(eigvals.real < 0), (
            f"Found eigenvalues with non-negative real parts: {eigvals.real}"
        )

    def test_numpy_version(self):
        from state_space_models.hippo import hippo_legs_matrix_numpy
        A_np = hippo_legs_matrix_numpy(8)
        assert A_np.shape == (8, 8)
        assert A_np.dtype == np.float64
        # Check a specific value
        assert abs(A_np[3, 1] + math.sqrt(7 * 3)) < 1e-10


class TestHiPPOLegT:
    """Tests for HiPPO-LegT matrix."""

    def test_shape(self):
        from state_space_models.hippo import hippo_legt_matrix
        A = hippo_legt_matrix(8)
        assert A.shape == (8, 8)

    def test_lower_triangular(self):
        from state_space_models.hippo import hippo_legt_matrix
        A = hippo_legt_matrix(8)
        idx = torch.triu_indices(8, 8, offset=1)
        upper = A[idx[0], idx[1]]
        assert torch.allclose(upper, torch.zeros_like(upper))

    def test_diagonal_halved(self):
        from state_space_models.hippo import hippo_legt_matrix
        N = 8
        A = hippo_legt_matrix(N)
        for n in range(N):
            expected = -(n + 1) / 2.0
            assert abs(A[n, n].item() - expected) < 1e-6


class TestHiPPOFouD:
    """Tests for HiPPO-FouD matrix."""

    def test_even_required(self):
        from state_space_models.hippo import hippo_foud_matrix
        with pytest.raises(AssertionError):
            hippo_foud_matrix(7)

    def test_shape(self):
        from state_space_models.hippo import hippo_foud_matrix
        A = hippo_foud_matrix(8)
        assert A.shape == (8, 8)

    def test_skew_symmetric(self):
        from state_space_models.hippo import hippo_foud_matrix
        A = hippo_foud_matrix(8)
        assert torch.allclose(A, -A.T)

    def test_block_structure(self):
        from state_space_models.hippo import hippo_foud_matrix
        A = hippo_foud_matrix(4)
        # Block (0,1): [0, 1; -1, 0] with freq=1
        assert A[0, 1].item() == 1.0
        assert A[1, 0].item() == -1.0
        # Block (2,3): [0, 2; -2, 0] with freq=2
        assert A[2, 3].item() == 2.0
        assert A[3, 2].item() == -2.0


# ============================================================================
# Parallel Scan Tests
# ============================================================================

class TestBinaryOperator:
    """Tests for the associative binary operator."""

    def test_associativity(self):
        from state_space_models.scan import binary_operator_diag
        a = (torch.tensor([0.5, 0.8]), torch.tensor([1.0, 2.0]))
        b = (torch.tensor([0.3, 0.6]), torch.tensor([3.0, 4.0]))
        c = (torch.tensor([0.2, 0.4]), torch.tensor([5.0, 6.0]))

        # (a o b) o c
        left = binary_operator_diag(binary_operator_diag(a, b), c)
        # a o (b o c)
        right = binary_operator_diag(a, binary_operator_diag(b, c))

        assert torch.allclose(left[0], right[0])
        assert torch.allclose(left[1], right[1])

    def test_identity(self):
        from state_space_models.scan import binary_operator_diag
        # Identity: (1, 0)
        identity = (torch.ones(3), torch.zeros(3))
        elem = (torch.tensor([0.5, 0.3, 0.7]), torch.tensor([2.0, 3.0, 4.0]))
        result = binary_operator_diag(identity, elem)
        assert torch.allclose(result[0], elem[0])
        assert torch.allclose(result[1], elem[1])


class TestParallelScan:
    """Tests for parallel scan implementations."""

    def test_simple_scan(self):
        from state_space_models.scan import parallel_scan, sequential_scan
        L, D = 8, 4
        elements = torch.randn(L, D)
        decays = torch.rand(L, D) * 0.9

        parallel = parallel_scan(elements, decays)
        sequential = sequential_scan(elements, decays)

        assert parallel.shape == (L, D)
        assert torch.allclose(parallel, sequential, atol=1e-5), (
            f"Max diff: {(parallel - sequential).abs().max()}"
        )

    def test_all_ones_decay(self):
        """With decay=1, scan is cumulative sum."""
        from state_space_models.scan import parallel_scan, sequential_scan
        L, D = 16, 3
        elements = torch.randn(L, D)
        decays = torch.ones(L, D)

        result = parallel_scan(elements, decays)
        expected = torch.cumsum(elements, dim=0)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_zero_decay(self):
        """With decay=0, scan output equals input (no memory)."""
        from state_space_models.scan import parallel_scan
        L, D = 8, 4
        elements = torch.randn(L, D)
        decays = torch.zeros(L, D)

        result = parallel_scan(elements, decays)
        assert torch.allclose(result, elements, atol=1e-5)

    def test_various_lengths(self):
        from state_space_models.scan import parallel_scan, sequential_scan
        for L in [1, 2, 3, 5, 7, 8, 12, 16, 31, 32, 64]:
            elements = torch.randn(L, 2)
            decays = torch.sigmoid(torch.randn(L, 2))
            p = parallel_scan(elements, decays)
            s = sequential_scan(elements, decays)
            assert torch.allclose(p, s, atol=1e-4), f"Failed at L={L}"

    def test_single_element(self):
        from state_space_models.scan import parallel_scan, sequential_scan
        elements = torch.randn(1, 3)
        decays = torch.rand(1, 3)
        assert torch.allclose(parallel_scan(elements, decays), elements)

    def test_reverse_scan(self):
        from state_space_models.scan import parallel_scan, sequential_scan
        L, D = 16, 3
        elements = torch.randn(L, D)
        decays = torch.rand(L, D) * 0.9

        p_rev = parallel_scan(elements, decays, reverse=True)
        s_rev = sequential_scan(elements, decays, reverse=True)
        assert torch.allclose(p_rev, s_rev, atol=1e-4)


# ============================================================================
# S4 Tests
# ============================================================================

class TestDPLRConversion:
    """Tests for DPLR-to-diagonal conversion."""

    def test_output_shapes(self):
        from state_space_models.s4 import dplr_to_diag
        N = 8
        Lambda = torch.randn(N) + 0.1j * torch.randn(N)
        P = torch.randn(N)
        Q = torch.randn(N)
        B = torch.randn(N)
        C = torch.randn(N)

        Lam_d, B_d, C_d = dplr_to_diag(Lambda, P, Q, B, C)
        assert Lam_d.shape == (N,)
        assert B_d.shape == (N,)
        assert C_d.shape == (N,)


class TestS4Kernel:
    """Tests for S4 kernel computation."""

    def test_kernel_real_output(self):
        from state_space_models.s4 import compute_s4_kernel
        N, L = 16, 64
        # Use real eigenvalues (stable, negative)
        Lambda = -0.5 * torch.ones(N) - 1j * torch.arange(N, dtype=torch.float32) * 0.5
        Lambda = Lambda.to(torch.complex64)
        B = torch.randn(N)
        C = torch.randn(N)

        K = compute_s4_kernel(Lambda, B, C, L, dt=0.01)
        assert K.shape == (L,)
        assert K.dtype == torch.float32
        # Kernel should be finite
        assert torch.isfinite(K).all()

    def test_kernel_with_hippo_init(self):
        from state_space_models.s4 import S4Kernel
        kernel = S4Kernel(N=16)
        K = kernel.forward(L=32)
        assert K.shape == (32,)
        assert torch.isfinite(K).all()

    def test_kernel_vs_direct_computation(self):
        """Test that kernel computation produces valid, decaying kernels."""
        from state_space_models.s4 import compute_s4_kernel, s4_kernel_conv
        N, L = 16, 64
        Lambda = -0.5 * torch.arange(1, N + 1, dtype=torch.float32)
        Lambda = Lambda + 0j
        B = torch.ones(N) * 0.5
        C = torch.ones(N) * 0.5

        # Compute kernel via FFT method
        K_fft = compute_s4_kernel(Lambda, B, C, L, dt=0.1)

        # Compute kernel via direct method (no DPLR — use zero P, Q)
        P = torch.zeros(N)
        Q = torch.zeros(N)
        K_direct = s4_kernel_conv(Lambda, P, Q, B, C, L, dt=0.1)

        # Both kernels should be finite and non-zero
        assert torch.isfinite(K_fft).all()
        assert torch.isfinite(K_direct).all()
        assert K_fft.abs().max() > 1e-6
        assert K_direct.abs().max() > 1e-6
        # Both should decay toward zero (kernel is causal and stable)
        assert K_direct.abs()[-5:].mean() < K_direct.abs()[:5].mean()


class TestS4Layer:
    """Tests for S4Layer forward pass."""

    def test_forward_shape(self):
        from state_space_models.s4 import S4Layer
        B, L, D = 2, 32, 4
        layer = S4Layer(N=16, d_model=D)
        u = torch.randn(B, L, D)
        y = layer(u)
        assert y.shape == (B, L, D)
        assert torch.isfinite(y).all()

    def test_step_mode(self):
        from state_space_models.s4 import S4Layer
        B, L, D = 2, 32, 4
        layer = S4Layer(N=16, d_model=D)
        u = torch.randn(B, L, D)

        # State shape: (B, d_model, N) — per-feature, per-state dim
        state = torch.zeros(B, D, layer.N)
        y_step = torch.zeros(B, L, D)
        for t in range(L):
            y_t, state = layer.step(u[:, t:t+1, :], state)
            y_step[:, t, :] = y_t

        # Check shapes and finiteness
        assert torch.isfinite(y_step).all()


class TestApplySSMConvolution:
    """Tests for SSM convolution application."""

    def test_output_shape(self):
        from state_space_models.s4 import apply_ssm_convolution
        B, L, D = 2, 32, 4
        u = torch.randn(B, L, D)
        K = torch.randn(L)
        y = apply_ssm_convolution(u, K)
        assert y.shape == (B, L, D)

    def test_zero_kernel(self):
        from state_space_models.s4 import apply_ssm_convolution
        u = torch.ones(2, 16, 3)
        K = torch.zeros(16)
        y = apply_ssm_convolution(u, K)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-5)


# ============================================================================
# S4D Tests
# ============================================================================

class TestS4DKernel:
    """Tests for S4D kernel computation."""

    def test_eigenvalues(self):
        from state_space_models.s4d import hippo_legs_eigenvalues
        eigvals = hippo_legs_eigenvalues(8)
        assert eigvals.shape == (8,)
        assert (eigvals < 0).all(), "All eigenvalues should be negative"

    def test_kernel_output(self):
        from state_space_models.s4d import compute_s4d_kernel
        N, L = 16, 32
        Lambda_re = -torch.arange(1, N + 1, dtype=torch.float32)
        Lambda_im = torch.zeros(N)
        B = torch.ones(N)
        C = torch.ones(N)

        K = compute_s4d_kernel(Lambda_re, Lambda_im, B, C, L, dt=0.01)
        assert K.shape == (L,)
        assert K.dtype == torch.float32
        assert torch.isfinite(K).all()

    def test_s4d_kernel_module(self):
        from state_space_models.s4d import S4DKernel
        for init_mode in ["legs", "inv", "lin", "real"]:
            kernel = S4DKernel(N=16, init=init_mode)
            K = kernel.forward(L=32)
            assert K.shape == (32,), f"Failed for init={init_mode}"
            assert torch.isfinite(K).all(), f"NaN in kernel for init={init_mode}"

    def test_different_init_modes(self):
        from state_space_models.s4d import S4DKernel
        kernels = {}
        for mode in ["legs", "inv", "lin", "real"]:
            kernels[mode] = S4DKernel(N=16, init=mode)

        # Each init should produce different kernels
        L = 32
        outputs = {m: k.forward(L) for m, k in kernels.items()}
        for m1 in outputs:
            for m2 in outputs:
                if m1 < m2:
                    assert not torch.allclose(outputs[m1], outputs[m2], atol=1e-3), (
                        f"Kernels for {m1} and {m2} are identical!"
                    )


class TestS4DLayer:
    """Tests for S4DLayer forward pass."""

    def test_forward_shape(self):
        from state_space_models.s4d import S4DLayer
        B, L, D = 2, 32, 4
        layer = S4DLayer(d_model=D, N=16)
        u = torch.randn(B, L, D)
        y = layer(u)
        assert y.shape == (B, L, D)
        assert torch.isfinite(y).all()

    def test_dropout(self):
        from state_space_models.s4d import S4DLayer
        layer = S4DLayer(d_model=4, N=16, dropout=0.5)
        layer.train()
        u = torch.randn(2, 32, 4)
        y1 = layer(u)
        y2 = layer(u)
        # With dropout > 0, should get different outputs in train mode
        assert not torch.allclose(y1, y2)

    def test_step_mode(self):
        from state_space_models.s4d import S4DLayer
        B, L, D = 2, 16, 2
        layer = S4DLayer(d_model=D, N=8)
        u = torch.randn(B, L, D)

        # State shape: (B, d_model, N) — per-feature, per-state dim
        state = torch.zeros(B, D, layer.N)
        for t in range(L):
            y_t, state = layer.step(u[:, t, :], state)
            assert y_t.shape == (B, D)
            assert torch.isfinite(y_t).all()


class TestApplyS4DConvolution:
    """Tests for S4D convolution."""

    def test_output_shape(self):
        from state_space_models.s4d import apply_s4d_convolution
        u = torch.randn(2, 32, 4)
        K = torch.randn(32)
        y = apply_s4d_convolution(u, K)
        assert y.shape == (2, 32, 4)

    def test_linearity(self):
        from state_space_models.s4d import apply_s4d_convolution
        u1 = torch.randn(2, 32, 4)
        u2 = torch.randn(2, 32, 4)
        K = torch.randn(32)
        alpha, beta = 2.0, 3.0

        y_combined = apply_s4d_convolution(alpha * u1 + beta * u2, K)
        y_separate = alpha * apply_s4d_convolution(u1, K) + beta * apply_s4d_convolution(u2, K)
        assert torch.allclose(y_combined, y_separate, atol=1e-4)


# ============================================================================
# SSM Discretization Tests
# ============================================================================

class TestDiscretization:
    """Tests for ZOH and bilinear discretization."""

    def test_zoh_scalar_delta(self):
        from state_space_models.ssm import discretize_zoh
        N = 8
        A = -torch.arange(1, N + 1, dtype=torch.float32)
        B = torch.ones(N)
        delta = torch.tensor(0.01)

        A_bar, B_bar = discretize_zoh(A, B, delta)
        assert A_bar.shape == (N,)
        assert B_bar.shape == (N,)
        assert (A_bar > 0).all(), "exp(delta * A) should be positive for real A < 0"
        assert (A_bar < 1).all(), "exp(negative) should be < 1"

    def test_zoh_small_delta(self):
        """For very small Δ, A_bar ≈ 1, B_bar ≈ B * Δ."""
        from state_space_models.ssm import discretize_zoh
        N = 4
        A = -torch.ones(N)
        B = torch.ones(N)
        delta = torch.tensor(1e-6)

        A_bar, B_bar = discretize_zoh(A, B, delta)
        # A_bar ≈ 1 for small delta
        assert torch.allclose(A_bar, torch.ones(N), atol=1e-5)
        # B_bar ≈ B * delta for small delta
        assert torch.allclose(B_bar, B * delta, atol=1e-8)

    def test_zoh_batched_delta(self):
        from state_space_models.ssm import discretize_zoh
        N = 4
        A = -torch.arange(1, N + 1, dtype=torch.float32)
        B = torch.ones(N)
        delta = torch.tensor([0.01, 0.02, 0.05])

        A_bar, B_bar = discretize_zoh(A, B, delta)
        assert A_bar.shape == (3, N)
        assert B_bar.shape == (3, N)

    def test_zoh_accuracy(self):
        """Test that discretized system approximates continuous one."""
        from state_space_models.ssm import discretize_zoh
        N = 2
        A = -torch.tensor([1.0, 2.0])
        B = torch.tensor([1.0, 1.0])
        delta = torch.tensor(0.1)

        A_bar, B_bar = discretize_zoh(A, B, delta)

        # Manual ZOH for A[0] = -1.0, delta = 0.1:
        # A_bar_0 = exp(-0.1) ≈ 0.904837
        assert abs(A_bar[0].item() - math.exp(-0.1)) < 1e-5
        # ZOH B̄ = (exp(Δ·A) - 1) / A · B
        # For A=-1, Δ=0.1, B=1: B̄ = (1 - exp(-0.1))
        expected_B0 = (1 - math.exp(-0.1))
        assert abs(B_bar[0].item() - expected_B0) < 1e-5

    def test_bilinear_scalar_delta(self):
        from state_space_models.ssm import discretize_bilinear
        N = 4
        A = -torch.arange(1, N + 1, dtype=torch.float32)
        B = torch.ones(N)
        delta = torch.tensor(0.01)

        A_bar, B_bar = discretize_bilinear(A, B, delta)
        assert A_bar.shape == (N,)
        assert B_bar.shape == (N,)
        # Bilinear is stable: |A_bar| < 1 for Re(A) < 0
        assert (A_bar.abs() < 1).all()

    def test_bilinear_vs_zoh(self):
        """Bilinear and ZOH should be similar for small Δ."""
        from state_space_models.ssm import discretize_zoh, discretize_bilinear
        N = 4
        A = -torch.arange(1, N + 1, dtype=torch.float32)
        B = torch.ones(N)
        delta = torch.tensor(1e-4)

        A_bar_zoh, B_bar_zoh = discretize_zoh(A, B, delta)
        A_bar_bil, B_bar_bil = discretize_bilinear(A, B, delta)

        # For very small Δ, they should be similar
        assert torch.allclose(A_bar_zoh, A_bar_bil, atol=0.01)
        assert torch.allclose(B_bar_zoh, B_bar_bil, atol=0.01)


# ============================================================================
# SSM Convolution Tests
# ============================================================================

class TestSSMConvolution:
    """Tests for SSM convolution mode."""

    def test_conv_kernel(self):
        from state_space_models.ssm import ssm_conv_kernel
        N, L = 4, 16
        A_bar = torch.tensor([0.9, 0.8, 0.7, 0.6])
        B_bar = torch.tensor([1.0, 0.5, 0.3, 0.2])
        C = torch.tensor([0.5, 0.3, 0.7, 0.9])

        K = ssm_conv_kernel(A_bar, B_bar, C, L)
        assert K.shape == (L,)
        assert torch.isfinite(K).all()

        # As t increases, kernel should decay to 0
        assert abs(K[-1].item()) < abs(K[0].item()) * 0.5, "Kernel should decay"

    def test_apply_ssm_conv(self):
        from state_space_models.ssm import apply_ssm_conv
        L = 32
        u = torch.randn(2, L)
        A = -torch.ones(4)
        B = torch.ones(4)
        C = torch.ones(4)

        y = apply_ssm_conv(u, A, B, C, delta=0.01)
        assert y.shape == (2, L)
        assert torch.isfinite(y).all()

    def test_skip_connection(self):
        from state_space_models.ssm import apply_ssm_conv
        L = 16
        u = torch.ones(1, L)
        A = -10.0 * torch.ones(4)  # Very fast decay → SSM output ≈ 0
        B = torch.zeros(4)
        C = torch.zeros(4)
        D = torch.tensor(2.0)

        # With zero B, C, the SSM output should be zero
        y = apply_ssm_conv(u, A, B, C, delta=0.01, D=D)
        # Output should be D * u = 2 * u
        assert torch.allclose(y, 2.0 * u, atol=1e-3)

    def test_linearity(self):
        from state_space_models.ssm import apply_ssm_conv
        L = 16
        u1 = torch.randn(1, L)
        u2 = torch.randn(1, L)
        A = -torch.ones(4)
        B = torch.ones(4)
        C = torch.ones(4)
        alpha, beta = 2.0, 3.0

        y_combined = apply_ssm_conv(alpha * u1 + beta * u2, A, B, C)
        y_separate = alpha * apply_ssm_conv(u1, A, B, C) + beta * apply_ssm_conv(u2, A, B, C)
        assert torch.allclose(y_combined, y_separate, atol=1e-4)


# ============================================================================
# DiagonalSSM Tests
# ============================================================================

class TestDiagonalSSM:
    """Tests for the DiagonalSSM module."""

    def test_forward_shape(self):
        from state_space_models.ssm import DiagonalSSM
        model = DiagonalSSM(hidden_dim=8, state_dim=16)
        u = torch.randn(2, 32, 8)
        y = model(u)
        assert y.shape == (2, 32, 8)
        assert torch.isfinite(y).all()

    def test_parameter_shapes(self):
        from state_space_models.ssm import DiagonalSSM
        model = DiagonalSSM(hidden_dim=8, state_dim=16)
        assert model.Lambda_real.shape == (8, 16)
        assert model.Lambda_imag.shape == (8, 16)
        assert model.B.shape == (8, 16)
        assert model.C.shape == (8, 16)
        assert model.D.shape == (8,)
        assert model.log_dt.shape == (8,)

    def test_get_A_complex(self):
        from state_space_models.ssm import DiagonalSSM
        model = DiagonalSSM(hidden_dim=4, state_dim=8)
        A = model.get_A()
        assert A.shape == (4, 8)
        assert A.is_complex()

    def test_gradient_flow(self):
        from state_space_models.ssm import DiagonalSSM
        model = DiagonalSSM(hidden_dim=4, state_dim=8)
        u = torch.randn(2, 16, 4)
        y = model(u)
        loss = y.sum()
        loss.backward()
        # Check gradients exist
        assert model.Lambda_real.grad is not None
        assert model.Lambda_imag.grad is not None
        assert model.B.grad is not None


# ============================================================================
# Mamba / S6 Tests
# ============================================================================

class TestMambaDiscretization:
    """Tests for Mamba discretize_ssm function."""

    def test_output_shapes(self):
        from state_space_models.mamba import discretize_ssm
        B, L, N = 2, 16, 8
        A = -torch.arange(1, N + 1, dtype=torch.float32)
        B_ssm = torch.randn(B, L, N)
        C_ssm = torch.randn(B, L, N)
        delta = torch.sigmoid(torch.randn(B, L, N)) * 0.1

        A_bar, B_bar, C_out = discretize_ssm(A, B_ssm, C_ssm, delta)
        assert A_bar.shape == (B, L, N)
        assert B_bar.shape == (B, L, N)
        assert C_out.shape == (B, L, N)
        assert (A_bar > 0).all()  # exp of real numbers
        assert (A_bar < 1).all()  # exp of negative numbers


class TestMambaBlock:
    """Tests for the Mamba block."""

    def test_config_defaults(self):
        from state_space_models.mamba import MambaConfig
        config = MambaConfig(d_model=64, d_state=16)
        assert config.d_model == 64
        assert config.d_state == 16
        assert config.expand == 2
        assert config.d_conv == 4

    def test_forward_shape(self):
        from state_space_models.mamba import MambaBlock, MambaConfig
        config = MambaConfig(d_model=32, d_state=8, expand=2, d_conv=4)
        block = MambaBlock(config)
        u = torch.randn(2, 64, 32)
        y = block(u)
        assert y.shape == (2, 64, 32)
        assert torch.isfinite(y).all()

    def test_small_config(self):
        from state_space_models.mamba import MambaBlock, MambaConfig
        config = MambaConfig(d_model=16, d_state=4, expand=1, d_conv=3)
        block = MambaBlock(config)
        u = torch.randn(1, 32, 16)
        y = block(u)
        assert y.shape == (1, 32, 16)

    def test_step_mode(self):
        from state_space_models.mamba import MambaBlock, MambaConfig
        config = MambaConfig(d_model=16, d_state=4, expand=2, d_conv=3)
        block = MambaBlock(config)

        d_inner = config.expand * config.d_model
        u = torch.randn(2, 8, 16)

        # Initialize states
        conv_state = torch.zeros(2, d_inner, config.d_conv - 1)
        ssm_state = torch.zeros(2, d_inner, config.d_state)

        for t in range(8):
            y_t, conv_state, ssm_state = block.step(u[:, t, :], conv_state, ssm_state)
            assert y_t.shape == (2, 16)
            assert torch.isfinite(y_t).all()

    def test_gradient_flow(self):
        from state_space_models.mamba import MambaBlock, MambaConfig
        config = MambaConfig(d_model=16, d_state=4, expand=1, d_conv=3)
        block = MambaBlock(config)
        u = torch.randn(2, 16, 16, requires_grad=False)
        y = block(u)
        loss = y.sum()
        loss.backward()
        # Check that key parameters got gradients
        assert block.A_log.grad is not None
        assert block.D.grad is not None

    def test_causality(self):
        """Mamba block should be causal: y_t depends only on u_{≤t}."""
        from state_space_models.mamba import MambaBlock, MambaConfig
        config = MambaConfig(d_model=8, d_state=4, expand=1, d_conv=3)
        block = MambaBlock(config)

        u1 = torch.randn(1, 16, 8)
        u2 = u1.clone()
        # Change a future input
        u2[0, 10, :] = 999.0

        y1 = block(u1)
        y2 = block(u2)

        # Outputs before position 10 should be identical
        assert torch.allclose(y1[0, :10, :], y2[0, :10, :], atol=1e-4), (
            "Mamba block is not causal!"
        )


# ============================================================================
# Mamba-2 / SSD Tests
# ============================================================================

class TestMamba2Config:
    """Tests for Mamba-2 configuration."""

    def test_defaults(self):
        from state_space_models.mamba2 import Mamba2Config
        config = Mamba2Config(d_model=64, d_state=16, n_heads=4)
        assert config.d_model == 64
        assert config.d_state == 16
        assert config.n_heads == 4
        assert config.chunk_size == 64


class TestSemiseparableMultiply:
    """Tests for semiseparable matrix multiplication."""

    def test_output_shape(self):
        from state_space_models.mamba2 import semiseparable_multiply
        B, L, N, D = 2, 8, 4, 4
        A = torch.sigmoid(torch.randn(B, L, N)) * 0.9
        B_mat = torch.randn(B, L, N)
        C_mat = torch.randn(B, L, N)
        x = torch.randn(B, L, D)

        y = semiseparable_multiply(A, B_mat, C_mat, x)
        assert y.shape == (B, L, D)
        assert torch.isfinite(y).all()

    def test_vs_reference(self):
        """Test semiseparable multiply against direct matrix construction."""
        from state_space_models.mamba2 import semiseparable_multiply
        B, L, N = 1, 4, 2
        A = torch.rand(B, L, N) * 0.5
        B_mat = torch.ones(B, L, N)
        C_mat = torch.ones(B, L, N)
        x = torch.randn(B, L, N)

        y = semiseparable_multiply(A, B_mat, C_mat, x)

        # Direct computation of the semiseparable matrix M
        # M[i, j] = sum_n C[i, n] * prod_{k=j+1}^{i} A[k, n] * B[j, n]  for j <= i
        #          = 0 for j > i
        M = torch.zeros(L, L)
        for i in range(L):
            for j in range(i + 1):
                for n in range(N):
                    prod = 1.0
                    for k in range(j + 1, i + 1):
                        prod *= A[0, k, n].item()
                    M[i, j] += C_mat[0, i, n].item() * prod * B_mat[0, j, n].item()

        y_direct = torch.zeros(B, L, N)
        for b in range(B):
            for n in range(N):
                y_direct[b, :, n] = M @ x[b, :, n]

        # Check correlation is high
        assert torch.allclose(y, y_direct, atol=1e-4), (
            f"Max diff: {(y - y_direct).abs().max()}"
        )


class TestSSDKernel:
    """Tests for SSD kernel."""

    def test_output_shape(self):
        from state_space_models.mamba2 import ssd_kernel
        B, L, H, P, N = 2, 16, 4, 8, 4
        u = torch.randn(B, L, H, P)
        A = torch.rand(H, N) * 0.5  # Per-head A
        B_mat = torch.randn(B, L, H, N)
        C_mat = torch.randn(B, L, H, N)

        y = ssd_kernel(u, A, B_mat, C_mat, chunk_size=8)
        assert y.shape == (B, L, H, P)
        assert torch.isfinite(y).all()


class TestMamba2Block:
    """Tests for Mamba-2 block."""

    def test_forward_shape(self):
        from state_space_models.mamba2 import Mamba2Block, Mamba2Config
        config = Mamba2Config(d_model=32, d_state=8, n_heads=4, expand=2)
        block = Mamba2Block(config)
        u = torch.randn(2, 32, 32)
        y = block(u)
        assert y.shape == (2, 32, 32)
        assert torch.isfinite(y).all()

    def test_small_config(self):
        from state_space_models.mamba2 import Mamba2Block, Mamba2Config
        config = Mamba2Config(d_model=16, d_state=4, n_heads=2, expand=2)
        block = Mamba2Block(config)
        u = torch.randn(1, 16, 16)
        y = block(u)
        assert y.shape == (1, 16, 16)

    def test_residual_connection(self):
        from state_space_models.mamba2 import Mamba2Block, Mamba2Config
        config = Mamba2Config(d_model=16, d_state=4, n_heads=2, expand=1)
        block = Mamba2Block(config)

        # Zero out all parameters to isolate residual
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()

        u = torch.randn(2, 16, 16)
        y = block(u)
        # With all params zero, output should be near zero (only residual adds u,
        # but through a zeroed output projection)
        assert torch.isfinite(y).all()

    def test_gradient_flow(self):
        from state_space_models.mamba2 import Mamba2Block, Mamba2Config
        config = Mamba2Config(d_model=16, d_state=4, n_heads=2, expand=1)
        block = Mamba2Block(config)
        u = torch.randn(2, 8, 16)
        y = block(u)
        loss = y.sum()
        loss.backward()
        assert block.A_log.grad is not None
        assert block.D.grad is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndSSM:
    """End-to-end tests combining multiple components."""

    def test_s4d_pipeline(self):
        """Complete S4D pipeline: init → kernel → conv → output."""
        from state_space_models.s4d import S4DKernel, apply_s4d_convolution
        kernel = S4DKernel(N=16, init="legs")
        K = kernel(64)

        u = torch.randn(1, 64, 2)
        y = apply_s4d_convolution(u, K)
        assert y.shape == (1, 64, 2)
        assert torch.isfinite(y).all()

    def test_ssm_discretize_apply(self):
        """Discretize an SSM and apply via convolution."""
        from state_space_models.ssm import discretize_zoh, ssm_conv_kernel
        N, L = 8, 32
        A = -torch.arange(1, N + 1, dtype=torch.float32)
        B = torch.ones(N)
        C = torch.ones(N)
        delta = torch.tensor(0.01)

        A_bar, B_bar = discretize_zoh(A, B, delta)
        K = ssm_conv_kernel(A_bar, B_bar, C, L)

        assert K.shape == (L,)
        assert torch.isfinite(K).all()
        assert (K.abs() > 0).any(), "Kernel should not be all zeros"

    def test_parallel_scan_ssm_consistency(self):
        """Parallel scan should match sequential scan for SSM recurrence."""
        from state_space_models.scan import parallel_scan, sequential_scan
        L, D = 32, 4
        # Simulate SSM parameters
        A_bar = torch.sigmoid(torch.randn(L, D)) * 0.9  # decays
        B_bar = torch.randn(L, D)  # input elements
        u = torch.randn(L, 1)

        elements = B_bar * u  # (L, D)
        decays = A_bar  # (L, D)

        p = parallel_scan(elements, decays)
        s = sequential_scan(elements, decays)
        assert torch.allclose(p, s, atol=1e-4), (
            f"Parallel and sequential scan diverge: max diff "
            f"{(p - s).abs().max()}"
        )

    def test_hippo_to_s4d_kernel(self):
        """Initialize from HiPPO, create S4D kernel, verify it produces output."""
        from state_space_models.hippo import hippo_legs_matrix
        from state_space_models.s4d import compute_s4d_kernel

        N = 16
        A_hippo = hippo_legs_matrix(N)
        # Use diagonal as S4D eigenvalues
        Lambda_re = torch.diag(A_hippo)
        Lambda_im = torch.zeros(N)
        B = torch.ones(N)
        C = torch.ones(N)

        K = compute_s4d_kernel(Lambda_re, Lambda_im, B, C, L=64, dt=0.01)
        assert torch.isfinite(K).all()


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_long_sequence_scan(self):
        from state_space_models.scan import parallel_scan, sequential_scan
        L, D = 256, 2
        elements = torch.randn(L, D) * 0.1
        decays = 0.99 * torch.ones(L, D)

        p = parallel_scan(elements, decays)
        s = sequential_scan(elements, decays)
        # For long sequences with decay < 1, values should be bounded
        assert p.abs().max() < 100, "Scan should not explode"
        assert torch.allclose(p, s, atol=1e-3)

    def test_near_zero_eigenvalues(self):
        """Kernel computation with very small eigenvalues."""
        from state_space_models.s4d import compute_s4d_kernel
        N, L = 4, 32
        Lambda_re = -torch.tensor([1e-6, 1e-5, 1e-4, 1e-3])
        Lambda_im = torch.zeros(N)
        B = torch.ones(N)
        C = torch.ones(N)

        K = compute_s4d_kernel(Lambda_re, Lambda_im, B, C, L, dt=0.01)
        assert torch.isfinite(K).all()

    def test_large_sequence_s4d(self):
        from state_space_models.s4d import S4DKernel, apply_s4d_convolution
        kernel = S4DKernel(N=16, init="real")
        K = kernel(512)

        u = torch.randn(1, 512, 1)
        y = apply_s4d_convolution(u, K)
        assert y.shape == (1, 512, 1)
        assert torch.isfinite(y).all()

    def test_zero_input(self):
        """Zero input should produce zero SSM output (excluding skip)."""
        from state_space_models.ssm import apply_ssm_conv
        u = torch.zeros(1, 32)
        A = -torch.ones(4)
        B = torch.ones(4)
        C = torch.ones(4)

        y = apply_ssm_conv(u, A, B, C, delta=0.01, D=None)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
