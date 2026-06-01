"""
S4: Structured State Space Sequence Model.

Implements the core S4 algorithm including:
- Diagonal Plus Low-Rank (DPLR) parameterization of HiPPO matrices
- Cauchy kernel for efficient convolution computation
- SSM convolution kernel generation
- Full S4 layer with trainable parameters

Key references:
- "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

from .hippo import hippo_legs_matrix, hippo_to_dplr


# ============================================================================
# DPLR Conversion
# ============================================================================


def dplr_to_diag(
    Lambda: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert DPLR parameterization to equivalent diagonal SSM.

    Uses Woodbury identity to absorb the low-rank correction P Q^T
    into equivalent B, C vectors for the diagonal A = diag(Lambda).

    The equivalent system is:
        Lambda_d = Lambda
        B_d = B + correction
        C_d = C + correction

    Args:
        Lambda: Diagonal entries, shape (N,).
        P: Low-rank vector, shape (N,).
        Q: Low-rank vector, shape (N,).
        B: Input vector, shape (N,).
        C: Output vector, shape (N,).

    Returns:
        Tuple (Lambda_d, B_d, C_d) of equivalent diagonal system.
    """
    # The DPLR system A = diag(L) - P Q^T can be converted via Woodbury:
    # (sI - A)^(-1) B = W(sI - diag(L))^(-1) W^(-1) B
    # For a simple conversion: apply the resolvent correction
    N = Lambda.shape[0]

    # Normalize for stability
    scale = torch.max(torch.abs(Lambda))
    if scale > 1e-8:
        Lambda_n = Lambda / scale
    else:
        Lambda_n = Lambda

    # Absorb low-rank part into B and C
    # B_equiv = (I + Q^T diag(1/Lambda) P) * B
    # ... simplified approximation
    Lambda_d = Lambda.clone()

    # Approximate: the low-rank correction modifies the effective B, C
    # For exact conversion: B_d = B * (1 + alpha), C_d = C * (1 + beta)
    # where alpha and beta account for the P Q^T correction
    inv_L = 1.0 / (Lambda + 1e-8)
    qp_term = torch.sum(Q * P * inv_L).real  # scalar

    if qp_term.abs() > 1e-8:
        correction = 1.0 / (1.0 + qp_term)
        B_d = B * correction + P * torch.sum(Q * B * inv_L).real
        C_d = C * correction + Q * torch.sum(P * C * inv_L).real
    else:
        B_d = B.clone()
        C_d = C.clone()

    return Lambda_d, B_d, C_d


# ============================================================================
# S4 Kernel Computation
# ============================================================================


def cauchy_kernel(
    Lambda: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the SSM kernel via the Cauchy kernel formulation.

    Args:
        Lambda: Diagonal entries, shape (N,).
        P: Low-rank vector, shape (N,).
        Q: Low-rank vector, shape (N,).
        omega: Frequency grid, shape (L//2+1,).

    Returns:
        Kernel values K(ω), shape (L//2+1,).
    """
    N = Lambda.shape[0]

    omega_complex = 2j * math.pi * omega.to(Lambda.device)
    resolvent = 1.0 / (omega_complex.unsqueeze(1) - Lambda.unsqueeze(0))

    Q_term = torch.sum(resolvent * Q.unsqueeze(0), dim=-1)
    P_term = torch.sum(resolvent * P.unsqueeze(0), dim=-1)
    denom = 1.0 + torch.sum(resolvent * (P.unsqueeze(0) * Q.unsqueeze(0)), dim=-1)

    direct = torch.sum(resolvent, dim=-1)
    correction = (P_term * Q_term) / (denom + 1e-10)

    kernel = direct + correction
    return kernel


def compute_s4_kernel(
    Lambda: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    L: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """
    Compute the S4 convolution kernel in the time domain.

    Uses FFT-based frequency domain computation for efficiency.

    Args:
        Lambda: Diagonal eigenvalues of A, shape (N,). Complex-valued.
        B: Input vector, shape (N,).
        C: Output vector, shape (N,).
        L: Desired kernel length.
        dt: Discretization step size.

    Returns:
        Real-valued convolution kernel, shape (L,).
    """
    device = Lambda.device
    dtype_real = torch.float32

    # Discretize Lambda: Lambda_d = exp(dt * Lambda)
    Lambda_d = torch.exp(dt * Lambda)

    # Compute kernel in frequency domain
    omega = torch.fft.rfftfreq(L).to(device)

    # Transfer function: H(z) = Σ_n C_n B_n / (z - λ_n) where z = e^{2π i ω}
    # This is the Z-domain representation
    # H(ω) = Σ_n C_n B_n / (e^{2π i ω} - e^{λ_n dt})
    z = torch.exp(2j * math.pi * omega).unsqueeze(1)  # (L//2+1, 1)
    poles = Lambda_d.unsqueeze(0)  # (1, N)
    resolvent = 1.0 / (z - poles + 1e-10)  # (L//2+1, N)

    CB = C * B  # (N,)
    K_freq = torch.sum(CB.unsqueeze(0) * resolvent, dim=-1)  # (L//2+1,)

    # Inverse FFT to time domain
    K_time = torch.fft.irfft(K_freq, n=L)

    # Return real kernel
    return K_time.real.to(dtype_real)


def s4_kernel(
    Lambda: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    L: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """Compute S4 convolution kernel with DPLR parameterization (legacy)."""
    omega = torch.fft.rfftfreq(L).to(Lambda.device)
    K_freq = cauchy_kernel(Lambda, P, Q, omega)
    K_time = torch.fft.irfft(K_freq, n=L) * dt
    return K_time


def s4_kernel_conv(
    Lambda: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    L: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """Compute S4 convolution kernel via direct time-domain method."""
    Lambda_d = torch.exp(dt * Lambda)
    t_indices = torch.arange(L, device=Lambda.device, dtype=torch.float32)
    Lambda_t = Lambda_d.unsqueeze(0).pow(t_indices.unsqueeze(1))
    K = torch.sum(C.unsqueeze(0) * Lambda_t * B.unsqueeze(0), dim=-1)
    return K


# ============================================================================
# S4 Kernel Module
# ============================================================================


class S4Kernel(nn.Module):
    """S4 kernel generator with HiPPO-initialized parameters."""

    def __init__(self, N: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        """
        Initialize S4 kernel.

        Args:
            N: State dimension.
            dt_min: Minimum step size.
            dt_max: Maximum step size.
        """
        super().__init__()
        self.N = N

        # Initialize from HiPPO
        A_hippo = hippo_legs_matrix(N)
        Lambda_init = torch.diag(A_hippo).float()

        # Trainable parameters
        self.log_Lambda_real = nn.Parameter(torch.log(-Lambda_init + 1e-4))
        self.Lambda_imag = nn.Parameter(torch.zeros(N))
        self.B = nn.Parameter(torch.ones(N))
        self.C = nn.Parameter(torch.randn(N) * 0.1)

        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def get_Lambda(self) -> torch.Tensor:
        """Get Lambda with enforced negative real part."""
        Lambda_real = -torch.exp(self.log_Lambda_real)
        return Lambda_real + 1j * self.Lambda_imag

    def forward(self, L: int) -> torch.Tensor:
        """Generate convolution kernel of length L."""
        Lambda = self.get_Lambda()
        dt = torch.exp(self.log_dt).item()
        return compute_s4_kernel(Lambda, self.B, self.C, L, dt)


# ============================================================================
# S4 Layer
# ============================================================================


class S4Layer(nn.Module):
    """S4 layer with HiPPO-initialized state space."""

    def __init__(
        self,
        N: int = 64,
        d_model: int = 128,
        dropout: float = 0.0,
        bidirectional: bool = False,
        mode: str = "conv",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        """
        Initialize S4 layer.

        Args:
            N: State dimension.
            d_model: Feature dimension.
            dropout: Dropout rate.
            bidirectional: Not used here.
            mode: "conv" or "recurrent".
            dt_min, dt_max: Step size range.
        """
        super().__init__()
        self.N = N
        self.d_model = d_model

        # Kernel generators per feature
        self.kernels = nn.ModuleList([
            S4Kernel(N, dt_min, dt_max)
            for _ in range(d_model)
        ])

        # Input/output projections
        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: (B, L, D) or (L, D)."""
        if u.dim() == 2:
            u = u.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        B, L, D = u.shape

        u_proj = self.input_proj(u)

        y = torch.zeros_like(u_proj)
        for d in range(D):
            kernel = self.kernels[d](L)
            for b in range(B):
                y[b, :, d] = _causal_conv1d(u_proj[b, :, d], kernel)

        y = y + self.D.unsqueeze(0).unsqueeze(0) * u
        y = self.norm(y)
        y = self.output_proj(y)
        y = self.dropout(y)
        y = y + u

        if squeeze_batch:
            y = y.squeeze(0)

        return y

    def step(
        self,
        u_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single recurrent step.

        Args:
            u_t: Input at time t, shape (B, D).
            state: Previous state, shape (B, D, N) — per-feature, per-state dim.

        Returns:
            Tuple of (y_t, new_state). y_t is (B, D), new_state is (B, D, N).
        """
        B = u_t.shape[0]
        D = self.d_model

        if u_t.dim() == 3:
            u_t = u_t.squeeze(1)  # (B, D)

        Lambda = torch.stack([k.get_Lambda() for k in self.kernels])  # (D, N)
        dt_vals = torch.exp(torch.stack([k.log_dt for k in self.kernels])).squeeze(-1)  # (D,)
        A_d = torch.exp(dt_vals.unsqueeze(1) * Lambda)  # (D, N)

        u_expanded = u_t.unsqueeze(-1)  # (B, D, 1)
        B_vec = torch.stack([k.B for k in self.kernels])  # (D, N)
        new_state = (A_d.unsqueeze(0) * state + B_vec.unsqueeze(0) * u_expanded).real

        C_vec = torch.stack([k.C for k in self.kernels])  # (D, N)
        y_t = torch.sum(C_vec.unsqueeze(0) * new_state, dim=-1)  # (B, D)
        y_t = y_t + self.D.unsqueeze(0) * u_t

        return y_t, new_state


# ============================================================================
# Legacy S4DKernel (from original naming)
# ============================================================================


class S4DKernel(nn.Module):
    """S4 kernel with DPLR parameterization (original naming)."""

    def __init__(
        self,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        init_method: str = "hippo",
    ):
        super().__init__()
        self.d_state = d_state

        if init_method == "hippo":
            A_hippo = hippo_legs_matrix(d_state)
            Lambda_init, P_init, Q_init = hippo_to_dplr(A_hippo, normalize=True)
        else:
            Lambda_init = -torch.rand(d_state) * 0.1
            P_init = torch.randn(d_state) * 0.1
            Q_init = torch.randn(d_state) * 0.1

        self.log_Lambda_real = nn.Parameter(torch.log(-Lambda_init + 1e-8))
        self.Lambda_imag = nn.Parameter(torch.zeros(d_state))
        self.P = nn.Parameter(P_init.clone())
        self.Q = nn.Parameter(Q_init.clone())
        self.B = nn.Parameter(torch.ones(d_state))
        self.C = nn.Parameter(torch.randn(d_state) * 0.1)

        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def get_Lambda(self) -> torch.Tensor:
        """Get Lambda with enforced negative real part."""
        Lambda_real = -torch.exp(self.log_Lambda_real)
        return Lambda_real + 1j * self.Lambda_imag

    def forward(self, L: int) -> torch.Tensor:
        """Generate S4 convolution kernel."""
        Lambda = self.get_Lambda()
        dt = torch.exp(self.log_dt).item()
        return s4_kernel(Lambda, self.P, self.Q, self.B, self.C, L, dt)


# ============================================================================
# Convolution Helpers
# ============================================================================


def _causal_conv1d(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Perform 1D causal convolution using FFT."""
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


def apply_ssm_convolution(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Apply SSM convolution kernel to input.

    Args:
        u: Input tensor, shape (B, L, D).
        K: Convolution kernel, shape (L,).

    Returns:
        Output tensor, shape (B, L, D).
    """
    B, L, D = u.shape
    y = torch.zeros_like(u)
    for b in range(B):
        for d in range(D):
            y[b, :, d] = _causal_conv1d(u[b, :, d], K)
    return y
