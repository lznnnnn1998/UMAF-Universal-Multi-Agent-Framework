"""
HiPPO Matrix Initialization for State Space Models.

Implements the HiPPO-LegS (Scaled Legendre) matrix and related variants
used to initialize the state transition matrix A in S4 models.

The HiPPO (High-order Polynomial Projection Operator) framework provides
mathematically principled matrices for online function approximation.

Key references:
- "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (Gu et al., 2020)
- "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., 2021)
"""

import torch
import numpy as np
from typing import Tuple


def hippo_legs_matrix(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Compute the HiPPO-LegS (Scaled Legendre) matrix A of size N×N.

    A[n,k] = -sqrt((2n+1)(2k+1)) if n > k
    A[n,k] = -(n+1)               if n == k
    A[n,k] = 0                     if n < k

    Args:
        N: State dimension.
        dtype: Torch dtype for the output tensor (default: float32).

    Returns:
        Tensor of shape (N, N) containing the HiPPO-LegS matrix.
    """
    A = torch.zeros(N, N, dtype=dtype)
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt((2 * n + 1) * (2 * k + 1))
            elif n == k:
                A[n, k] = -(n + 1)
    return A


def hippo_legs_matrix_numpy(N: int) -> np.ndarray:
    """NumPy version of hippo_legs_matrix."""
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt((2 * n + 1) * (2 * k + 1))
            elif n == k:
                A[n, k] = -(n + 1)
    return A


def hippo_legs_matrix_vectorized(N: int) -> torch.Tensor:
    """Vectorized computation of the HiPPO-LegS matrix."""
    n = torch.arange(N, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    k = torch.arange(N, dtype=torch.float32).unsqueeze(0)  # (1, N)
    lower_mask = (n > k).float()
    diag_mask = (n == k).float()
    lower_entries = -torch.sqrt((2 * n + 1) * (2 * k + 1))
    diag_entries = -(n + 1)
    A = lower_mask * lower_entries + diag_mask * diag_entries
    return A


def hippo_legt_matrix(N: int) -> torch.Tensor:
    """
    Compute the HiPPO-LegT (Translated Legendre) matrix.

    Lower-triangular variant with halved diagonal for T-normalization.
    A[n,k] = -sqrt((2n+1)(2k+1)) if n > k
    A[n,k] = -(n+1)/2              if n == k
    A[n,k] = 0                     if n < k

    Args:
        N: State dimension.

    Returns:
        Tensor of shape (N, N) containing the HiPPO-LegT matrix.
    """
    n = torch.arange(N, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(N, dtype=torch.float32).unsqueeze(0)

    sqrt_term = torch.sqrt((2 * n + 1) * (2 * k + 1))
    diag = -(n + 1) / 2.0

    A = torch.where(n > k, -sqrt_term,
                    torch.where(n < k, torch.zeros_like(sqrt_term), diag))
    return A


def hippo_fout_matrix(N: int) -> torch.Tensor:
    """
    Compute the HiPPO-FouD (Fourier) matrix.

    Skew-symmetric block-diagonal Fourier matrix for recurrent
    frequency decomposition.

    Args:
        N: State dimension.

    Returns:
        Tensor of shape (N, N) containing the HiPPO-FouD matrix.

    Raises:
        AssertionError: If N is not even.
    """
    assert N % 2 == 0, f"HiPPO-FouD requires even N, got {N}"

    A = torch.zeros(N, N)
    num_blocks = N // 2
    for i in range(num_blocks):
        freq = i + 1
        r, c = 2 * i, 2 * i + 1
        A[r, c] = freq
        A[c, r] = -freq
    return A


def hippo_foud_matrix(N: int) -> torch.Tensor:
    """Alias for hippo_fout_matrix matching the test import name."""
    return hippo_fout_matrix(N)


def hippo_lagm_matrix(N: int) -> torch.Tensor:
    """Compute the HiPPO-LagM (Generalized Laguerre) matrix."""
    A = torch.zeros(N, N)
    for n in range(N):
        for k in range(N):
            if k <= n:
                A[n, k] = -1.0
    return A


def hippo_to_dplr(
    A: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a HiPPO matrix to Diagonal Plus Low-Rank (DPLR) form.

    A = diag(Lambda) - P @ Q^T

    Args:
        A: HiPPO matrix of shape (N, N).
        normalize: Whether to normalize for stability.

    Returns:
        Tuple of (Lambda, P, Q).
    """
    N = A.shape[0]
    n = torch.arange(N, dtype=torch.float32)
    Lambda = torch.diag(A).float()

    if normalize:
        Lambda = Lambda / torch.max(torch.abs(Lambda)) * (-0.5)

    P = torch.sqrt(2 * n + 1) / 2.0
    Q = P.clone()

    return Lambda, P, Q


def hippo_initial_state(N: int, L: int = 1) -> torch.Tensor:
    """Compute the initial state for a HiPPO-LegS system."""
    n = torch.arange(N, dtype=torch.float32)
    x0 = torch.sqrt(2 * n + 1) / np.sqrt(L)
    return x0


def hippo_reconstruct(
    x: torch.Tensor,
    eval_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct a function from HiPPO-LegS coefficients."""
    if eval_points is None:
        eval_points = torch.linspace(0, 1, 100)

    if torch.any(eval_points < 0) or torch.any(eval_points > 1):
        raise ValueError("eval_points must be in [0, 1]")

    N = x.shape[-1]
    M = len(eval_points)

    P = torch.zeros(N, M)
    if N > 0:
        P[0, :] = 1.0
    if N > 1:
        P[1, :] = 2 * eval_points - 1

    for n_val in range(2, N):
        P[n_val, :] = ((2 * n_val - 1) * (2 * eval_points - 1) * P[n_val - 1, :]
                       - (n_val - 1) * P[n_val - 2, :]) / n_val

    n_idx = torch.arange(N).unsqueeze(1)
    scale = torch.sqrt(2 * n_idx + 1)
    scaled_P = scale * P

    if x.dim() == 1:
        result = torch.sum(x.unsqueeze(1) * scaled_P, dim=0)
    else:
        result = torch.einsum('...n,nm->...m', x, scaled_P)

    return result
