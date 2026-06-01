"""
S4D: Diagonal State Space Model.

S4D simplifies S4 by using a purely diagonal state matrix A,
removing the low-rank correction (P, Q). The diagonal form
A = diag(λ₁, ..., λ_N) gives efficient O(N+L) computation.

Key references:
- "On the Parameterization and Initialization of Diagonal State Space
  Models" (Gu et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

from .hippo import hippo_legs_matrix


# ============================================================================
# Eigenvalues
# ============================================================================


def hippo_legs_eigenvalues(N: int) -> torch.Tensor:
    """
    Extract eigenvalues from the HiPPO-LegS matrix diagonal.

    These serve as good initial eigenvalues for S4D models.

    Args:
        N: Number of eigenvalues/state dimension.

    Returns:
        Tensor of N negative eigenvalues.
    """
    A = hippo_legs_matrix(N)
    return torch.diag(A)


# ============================================================================
# S4D Kernel Computation
# ============================================================================


def compute_s4d_kernel(
    Lambda_re: torch.Tensor,
    Lambda_im: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    L: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """
    Compute the S4D convolution kernel for diagonal A.

    K[t] = Σ_n C_n * B_n * exp(λ_n * t * dt)

    Uses FFT for efficient computation.

    Args:
        Lambda_re: Real parts of eigenvalues, shape (N,).
        Lambda_im: Imaginary parts of eigenvalues, shape (N,).
        B: Input vector, shape (N,).
        C: Output vector, shape (N,).
        L: Kernel length.
        dt: Step size.

    Returns:
        Real-valued convolution kernel, shape (L,).
    """
    N = Lambda_re.shape[0]
    device = Lambda_re.device

    # Construct complex eigenvalues
    Lambda = Lambda_re + 1j * Lambda_im  # (N,)

    # Discretize: λ_d = exp(λ * dt)
    Lambda_d = torch.exp(dt * Lambda)

    # Frequency domain computation via FFT
    omega = torch.fft.rfftfreq(L).to(device)  # (L//2+1,)

    # Transfer function: H(ω) = Σ_n C_n B_n / (e^{2πiω} - e^{λ_ndt})
    z = torch.exp(2j * math.pi * omega).unsqueeze(1)  # (L//2+1, 1)
    poles = Lambda_d.unsqueeze(0)  # (1, N)
    resolvent = 1.0 / (z - poles + 1e-10)  # (L//2+1, N)

    CB = C * B  # (N,)
    K_freq = torch.sum(CB.unsqueeze(0) * resolvent, dim=-1)  # (L//2+1,)

    # Inverse FFT to time domain
    K_time = torch.fft.irfft(K_freq, n=L)

    # Scale by dt
    return K_time.real


def s4d_kernel_diag(
    Lambda: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    L: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """Compute S4D kernel directly in time domain (legacy)."""
    t = torch.arange(L, dtype=torch.float32, device=Lambda.device)
    exponents = Lambda.unsqueeze(0) * t.unsqueeze(1) * dt
    exponents = torch.clamp(exponents, min=-50.0, max=50.0)
    Lambda_pow = torch.exp(exponents)
    K = torch.sum(C.unsqueeze(0) * Lambda_pow * B.unsqueeze(0), dim=-1)
    return K


def s4d_kernel_fft(
    Lambda: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    L: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """Compute S4D kernel using FFT (legacy)."""
    return compute_s4d_kernel(Lambda.real, Lambda.imag, B, C, L, dt)


# ============================================================================
# S4D Kernel Module
# ============================================================================


class S4DKernel(nn.Module):
    """S4D kernel generator with diagonal state matrix."""

    def __init__(
        self,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        init: str = "legs",
        lambda_init_scale: float = 1.0,
    ):
        """
        Initialize S4D kernel.

        Args:
            N: State dimension.
            dt_min: Minimum step size.
            dt_max: Maximum step size.
            init: Lambda initialization method:
                - "legs": From HiPPO-LegS diagonal.
                - "inv": 1/(n+1) decay.
                - "lin": Linearly spaced.
                - "real": Real eigenvalues as in S4D-Real.
            lambda_init_scale: Scale factor.
        """
        super().__init__()
        self.N = N

        # Initialize Lambda
        if init == "legs":
            Lambda = hippo_legs_eigenvalues(N) * lambda_init_scale
        elif init == "inv":
            n = torch.arange(1, N + 1, dtype=torch.float32)
            Lambda = -1.0 / (n) * lambda_init_scale
        elif init == "lin":
            Lambda = -torch.linspace(0.5, 10.0, N) * lambda_init_scale
        elif init == "real":
            Lambda = -0.5 * torch.ones(N) * lambda_init_scale
        else:
            raise ValueError(f"Unknown init: {init}")

        # Trainable parameters
        self.log_Lambda_real = nn.Parameter(torch.log(-Lambda + 1e-4))
        self.Lambda_imag = nn.Parameter(torch.zeros(N))
        self.B = nn.Parameter(torch.ones(N) + 0.01 * torch.randn(N))
        self.C = nn.Parameter(torch.randn(N) * 0.1)

        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def get_Lambda(self) -> torch.Tensor:
        """Get Lambda with enforced negative real part."""
        Lambda_real = -torch.exp(self.log_Lambda_real)
        return Lambda_real + 1j * self.Lambda_imag

    def forward(self, L: int) -> torch.Tensor:
        """Generate S4D convolution kernel of length L."""
        Lambda = self.get_Lambda()
        dt = torch.exp(self.log_dt).item()
        return compute_s4d_kernel(Lambda.real, Lambda.imag, self.B, self.C, L, dt)


# ============================================================================
# S4D Layer
# ============================================================================


class S4DLayer(nn.Module):
    """S4D layer with diagonal state matrices."""

    def __init__(
        self,
        d_model: int = 128,
        N: int = 64,
        dropout: float = 0.0,
        bidirectional: bool = False,
        mode: str = "conv",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        init: str = "legs",
    ):
        """
        Initialize S4D layer.

        Args:
            d_model: Input/output feature dimension.
            N: State dimension.
            dropout: Dropout rate.
            bidirectional: Not used.
            mode: "conv" or "recurrent".
            dt_min, dt_max: Step size range.
            init: Lambda initialization method.
        """
        super().__init__()
        self.N = N
        self.d_model = d_model

        self.kernels = nn.ModuleList([
            S4DKernel(N, dt_min, dt_max, init)
            for _ in range(d_model)
        ])

        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: (B, L, d_model) or (L, d_model)."""
        squeeze_batch = False
        if u.dim() == 2:
            u = u.unsqueeze(0)
            squeeze_batch = True

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
            u_t: Input at time t, shape (B, d_model).
            state: Previous state, shape (B, d_model, N) — per-feature, per-state dim.

        Returns:
            Tuple of (y_t, new_state). y_t: (B, d_model), new_state: (B, d_model, N).
        """
        B, D = u_t.shape

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


def apply_s4d_convolution(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Apply S4D convolution kernel to input.

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
