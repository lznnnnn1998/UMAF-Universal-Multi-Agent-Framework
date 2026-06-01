"""
SSM: State Space Model utilities and modules.

Provides discretization methods, convolution kernel generation,
and a DiagonalSSM module for SSM experiments.
"""

import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Discretization
# ============================================================================


def discretize_zoh(
    A: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-Order Hold (ZOH) discretization.

    For diagonal A:
        A_bar_n = exp(delta * lambda_n)
        B_bar_n = (exp(delta * lambda_n) - 1) / lambda_n * B_n

    Args:
        A: (N,) diagonal entries of the state matrix.
        B: (N,) input projection vector.
        delta: Step size — scalar tensor (,) or batched (M,).

    Returns:
        (A_bar, B_bar):
        - If delta is scalar: A_bar (N,), B_bar (N,).
        - If delta is batched (M,): A_bar (M, N), B_bar (M, N).
    """
    if delta.dim() == 0:
        # Scalar delta
        dA = delta * A  # (N,)
        A_bar = torch.exp(dA)
        # B_bar = (exp(dA) - 1) / A * B
        # Handle near-zero A safely
        dA_small = torch.abs(dA) < 1e-4
        B_bar = torch.where(
            dA_small,
            delta * B * (1.0 + dA / 2.0),  # Taylor
            (A_bar - 1.0) / (A + 1e-10) * B,
        )
        return A_bar, B_bar
    else:
        # Batched delta: (M,)
        M = delta.shape[0]
        dA = delta.unsqueeze(1) * A.unsqueeze(0)  # (M, N)
        A_bar = torch.exp(dA)
        dA_small = torch.abs(dA) < 1e-4
        B_bar = torch.where(
            dA_small,
            delta.unsqueeze(1) * B.unsqueeze(0) * (1.0 + dA / 2.0),
            (A_bar - 1.0) / (A.unsqueeze(0) + 1e-10) * B.unsqueeze(0),
        )
        return A_bar, B_bar


def discretize_bilinear(
    A: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (Tustin) discretization.

    For diagonal A:
        A_bar_n = (1 + delta*lambda_n/2) / (1 - delta*lambda_n/2)
        B_bar_n = sqrt(delta) * B_n / (1 - delta*lambda_n/2)

    Args:
        A: (N,) diagonal entries.
        B: (N,) input vector.
        delta: Step size — scalar tensor (,) or batched (M,).

    Returns:
        (A_bar, B_bar).
    """
    if delta.dim() == 0:
        half_dA = 0.5 * delta * A
        denom = 1.0 - half_dA + 1e-10
        A_bar = (1.0 + half_dA) / denom
        B_bar = torch.sqrt(delta) * B / denom
        return A_bar, B_bar
    else:
        delta_exp = delta.unsqueeze(1)  # (M, 1)
        A_exp = A.unsqueeze(0)  # (1, N)
        B_exp = B.unsqueeze(0)  # (1, N)
        half_dA = 0.5 * delta_exp * A_exp
        denom = 1.0 - half_dA + 1e-10
        A_bar = (1.0 + half_dA) / denom
        B_bar = torch.sqrt(delta_exp) * B_exp / denom
        return A_bar, B_bar


# ============================================================================
# SSM Convolution
# ============================================================================


def ssm_conv_kernel(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    C: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """Generate SSM convolution kernel from discretized parameters.

    K[t] = sum_n C_n * A_bar_n^t * B_bar_n  for t = 0, ..., L-1

    Args:
        A_bar: (N,) discretized diagonal A.
        B_bar: (N,) discretized input vector.
        C: (N,) output vector.
        L: Kernel length.

    Returns:
        Convolution kernel, shape (L,).
    """
    N = A_bar.shape[0]
    device = A_bar.device

    t = torch.arange(L, device=device, dtype=A_bar.dtype)  # (L,)
    A_pow = A_bar.unsqueeze(0).pow(t.unsqueeze(1))  # (L, N)

    K = torch.sum(C.unsqueeze(0) * A_pow * B_bar.unsqueeze(0), dim=-1)  # (L,)

    if torch.is_complex(K):
        K = K.real

    return K


def _causal_conv1d(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """1D causal convolution using FFT."""
    L = u.shape[0]
    K = kernel.shape[0]
    if K > L:
        kernel = kernel[:L]
        K = L

    n_fft = L + K - 1
    U = torch.fft.rfft(u, n=n_fft)
    V = torch.fft.rfft(kernel, n=n_fft)
    y = torch.fft.irfft(U * V, n=n_fft)
    return y[:L]


def apply_ssm_conv(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    delta: float = 1.0,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a time-invariant diagonal SSM using convolution mode.

    Args:
        u: (B, L) input signal.
        A: (N,) diagonal state matrix.
        B: (N,) input vector.
        C: (N,) output vector.
        delta: Step size.
        D: Optional skip connection scalar.

    Returns:
        y: (B, L) output signal.
    """
    B_dim, L = u.shape
    dt = torch.tensor(delta, dtype=A.dtype, device=A.device)

    A_bar, B_bar = discretize_zoh(A, B, dt)

    K = ssm_conv_kernel(A_bar, B_bar, C, L)

    y = torch.zeros_like(u)
    for b in range(B_dim):
        y[b] = _causal_conv1d(u[b], K)

    if D is not None:
        y = y + D * u

    return y


def ssm_recurrent_step(
    h: torch.Tensor,
    u_t: torch.Tensor,
    A_bar_t: torch.Tensor,
    B_bar_t: torch.Tensor,
    C_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of the SSM recurrence.

    h_new = A_bar_t * h + B_bar_t * u_t
    y_t = C_t^T @ h_new

    Args:
        h: (..., N) hidden state.
        u_t: (...,) scalar input.
        A_bar_t: (..., N) discretized A.
        B_bar_t: (..., N) discretized B.
        C_t: (..., N) output vector.

    Returns:
        (h_new, y_t).
    """
    h_new = A_bar_t * h + B_bar_t * u_t.unsqueeze(-1)
    y_t = (C_t * h_new).sum(dim=-1)
    return h_new, y_t


# ============================================================================
# DiagonalSSM Module
# ============================================================================


class DiagonalSSM(nn.Module):
    """A diagonal SSM layer with learnable parameters.

    Parameters:
      - Lambda_real: (H, N) real part of diagonal A
      - Lambda_imag: (H, N) imaginary part of diagonal A
      - B: (H, N) input vector
      - C: (H, N) output vector
      - D: (H,) skip connection
      - log_dt: (H,) log step size
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        state_dim: int = 64,
        dt_scale: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.Lambda_real = nn.Parameter(torch.randn(hidden_dim, state_dim))
        self.Lambda_imag = nn.Parameter(torch.randn(hidden_dim, state_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim, state_dim) / math.sqrt(state_dim))
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim) / math.sqrt(state_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))
        self.log_dt = nn.Parameter(torch.full((hidden_dim,), math.log(dt_scale)))

    def get_A(self) -> torch.Tensor:
        """Get the complex diagonal A matrix: Lambda_real + i * Lambda_imag."""
        return torch.complex(self.Lambda_real, self.Lambda_imag)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply the SSM to input signal u.

        Args:
            u: (B, L, H) input signal.

        Returns:
            y: (B, L, H) output signal.
        """
        B_batch, L, H = u.shape
        N = self.state_dim

        A = self.get_A()  # (H, N)
        delta = torch.exp(self.log_dt)  # (H,)

        # Discretize per channel
        # For each hidden dim h, discretize A[h] (N,) with dt[h] (scalar)
        A_bar_list = []
        B_bar_list = []
        for h in range(H):
            a_bar, b_bar = discretize_zoh(A[h], self.B[h], delta[h])
            A_bar_list.append(a_bar.real)
            B_bar_list.append(b_bar.real)

        A_bar = torch.stack(A_bar_list, dim=0)  # (H, N)
        B_bar = torch.stack(B_bar_list, dim=0)  # (H, N)

        # Compute kernel per channel: (H, L)
        K = torch.stack([
            ssm_conv_kernel(A_bar[h], B_bar[h], self.C[h], L)
            for h in range(H)
        ], dim=0)  # (H, L)

        # Apply convolution per batch and channel
        y = torch.zeros_like(u)
        for b in range(B_batch):
            for h in range(H):
                y[b, :, h] = _causal_conv1d(u[b, :, h], K[h])

        # Skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def init_hippo(self, measure: str = "legs"):
        """Initialize Lambda using HiPPO eigenvalues (S4D initialization)."""
        from .hippo import hippo_legs_matrix
        A_hippo = hippo_legs_matrix(self.state_dim)
        lam = torch.diag(A_hippo).float()

        with torch.no_grad():
            self.Lambda_real.copy_(lam.unsqueeze(0).expand(self.hidden_dim, -1))
            self.Lambda_imag.zero_()
