"""
Parallel Associative Scan for State Space Models.

Implements efficient parallel scan (prefix sum) operations using the
Blelloch scan algorithm and the associative scan pattern. Core primitive
for efficient recurrent computation in Mamba/S6 architectures.

The scan computes all prefixes for the recurrence:
    x_t = a_t * x_{t-1} + b_t

Key references:
- "Prefix Sums and Their Applications" (Blelloch, 1990)
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
"""

import torch
from typing import Any, Callable


def binary_operator_diag(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Binary associative operator for diagonal SSM recurrence.

    op((a1, b1), (a2, b2)) = (a2 * a1, a2 * b1 + b2)

    Args:
        a: Tuple (a_A, a_Bu) representing first segment.
        b: Tuple (b_A, b_Bu) representing second segment.

    Returns:
        Tuple (combined_A, combined_Bu).
    """
    a_A, a_Bu = a
    b_A, b_Bu = b
    combined_A = b_A * a_A
    combined_Bu = b_A * a_Bu + b_Bu
    return (combined_A, combined_Bu)


def binary_operator_matmul(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Binary associative operator for full-matrix SSM recurrence.

    Args:
        a: Tuple (a_A, a_Bu) where a_A is (D, D).
        b: Tuple (b_A, b_Bu).

    Returns:
        Tuple (combined_A, combined_Bu).
    """
    a_A, a_Bu = a
    b_A, b_Bu = b
    combined_A = torch.matmul(b_A, a_A)
    combined_Bu = torch.matmul(b_A.unsqueeze(-2), a_Bu.unsqueeze(-1)).squeeze(-1) + b_Bu
    return (combined_A, combined_Bu)


def blelloch_scan(
    elements: list[Any],
    op: Callable = binary_operator_diag,
) -> list[Any]:
    """
    Blelloch parallel prefix scan (exclusive scan).

    Each element should be compatible with the binary operator `op`.
    With the default `binary_operator_diag`, elements should be tuples of
    the form (A_factor, Bu_term) representing SSM segments.

    Args:
        elements: List of N elements to scan (tuples of tensors for the
            default binary_operator_diag).
        op: Associative binary operator.

    Returns:
        List of N elements representing the exclusive scan results.
    """
    N = len(elements)
    if N == 0:
        return []
    if N == 1:
        return [_identity_like(elements[0])]

    n_pow2 = 1
    while n_pow2 < N:
        n_pow2 <<= 1

    padded = list(elements) + [_identity_like(elements[0])] * (n_pow2 - N)

    # Up-sweep
    data = list(padded)
    for d in range(int(torch.log2(torch.tensor(n_pow2)))):
        stride = 1 << d
        for i in range(0, n_pow2, 2 * stride):
            if i + stride < n_pow2:
                data[i + 2 * stride - 1] = op(data[i + stride - 1],
                                               data[i + 2 * stride - 1])

    # Down-sweep
    data[-1] = _identity_like(elements[0])
    for d in range(int(torch.log2(torch.tensor(n_pow2))) - 1, -1, -1):
        stride = 1 << d
        for i in range(0, n_pow2, 2 * stride):
            if i + stride < n_pow2:
                t = data[i + stride - 1]
                data[i + stride - 1] = data[i + 2 * stride - 1]
                data[i + 2 * stride - 1] = op(t, data[i + 2 * stride - 1])

    return data[:N]


def parallel_scan(
    elements: torch.Tensor,
    decays: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:
    """
    Parallel associative scan for the recurrence:
        x_t = decays[t] * x_{t-1} + elements[t]

    Computes all prefix states efficiently using a GPU-friendly
    parallel scan with O(log L) depth.

    Args:
        elements: Input elements (Bu terms), shape (L, D).
        decays: Diagonal A factors, shape (L, D).
        reverse: If True, computes reverse scan (future-to-past).

    Returns:
        Hidden states x_t for all t, shape (L, D).
    """
    if reverse:
        elements = torch.flip(elements, dims=[0])
        decays = torch.flip(decays, dims=[0])

    L = elements.shape[0]

    if L == 1:
        result = elements.clone()
        if reverse:
            result = torch.flip(result, dims=[0])
        return result

    # Pad to power of 2
    n_pow2 = 1
    while n_pow2 < L:
        n_pow2 <<= 1

    if n_pow2 > L:
        pad_shape_a = list(decays.shape)
        pad_shape_b = list(elements.shape)
        pad_shape_a[0] = n_pow2 - L
        pad_shape_b[0] = n_pow2 - L
        identity_a = torch.ones(pad_shape_a, dtype=decays.dtype, device=decays.device)
        zero_b = torch.zeros(pad_shape_b, dtype=elements.dtype, device=elements.device)
        a = torch.cat([decays, identity_a], dim=0)
        b = torch.cat([elements, zero_b], dim=0)
    else:
        a = decays.clone()
        b = elements.clone()

    log_n = int(torch.log2(torch.tensor(n_pow2)).item())

    for d in range(log_n):
        stride = 2 ** d

        a_shifted = torch.roll(a, shifts=stride, dims=0)
        b_shifted = torch.roll(b, shifts=stride, dims=0)

        # For first 'stride' elements, keep unchanged
        mask = torch.zeros_like(a)
        mask[stride:] = 1.0

        a_new = torch.where(mask.bool(), a * a_shifted, a)
        b_new = torch.where(mask.bool(), a * b_shifted + b, b)

        a = a_new
        b = b_new

    # Trim padding
    result = b[:L]

    if reverse:
        result = torch.flip(result, dims=[0])

    return result


def sequential_scan(
    elements: torch.Tensor,
    decays: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:
    """
    Sequential scan (reference implementation) for verification.

    Computes x_t = decays[t] * x_{t-1} + elements[t] sequentially.

    Args:
        elements: Input elements, shape (L, D).
        decays: Decay factors, shape (L, D).
        reverse: If True, scans from future to past.

    Returns:
        Hidden states, shape (L, D).
    """
    L = elements.shape[0]

    if reverse:
        elements = torch.flip(elements, dims=[0])
        decays = torch.flip(decays, dims=[0])

    x = torch.zeros_like(elements)
    if L > 0:
        x[0] = elements[0]
        for t in range(1, L):
            x[t] = decays[t] * x[t - 1] + elements[t]

    if reverse:
        x = torch.flip(x, dims=[0])

    return x


def associative_scan(
    A_factors: torch.Tensor,
    Bu_terms: torch.Tensor,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel associative scan for SSM recurrence.

    Computes all prefixes for: x_t = a_t * x_{t-1} + b_t
    where a_t = A_factors[t], b_t = Bu_terms[t].

    Args:
        A_factors: Diagonal A factors, shape (..., L).
        Bu_terms: B*u terms, same shape as A_factors.
        reverse: If True, computes reverse scan.

    Returns:
        Tuple (X, final_state).
    """
    if reverse:
        A_factors = torch.flip(A_factors, dims=[-1])
        Bu_terms = torch.flip(Bu_terms, dims=[-1])

    L = A_factors.shape[-1]

    n_pow2 = 1
    while n_pow2 < L:
        n_pow2 <<= 1

    orig_shape_A = A_factors.shape
    if n_pow2 > L:
        pad_shape = list(orig_shape_A)
        pad_shape[-1] = n_pow2 - L
        identity_A = torch.ones(pad_shape, dtype=A_factors.dtype, device=A_factors.device)
        zero_Bu = torch.zeros(pad_shape, dtype=Bu_terms.dtype, device=Bu_terms.device)
        A_factors = torch.cat([A_factors, identity_A], dim=-1)
        Bu_terms = torch.cat([Bu_terms, zero_Bu], dim=-1)

    a = A_factors.clone()
    b = Bu_terms.clone()

    log_n = int(torch.log2(torch.tensor(n_pow2)).item())

    for d in range(log_n):
        stride = 2 ** d

        a_shifted = torch.roll(a, shifts=stride, dims=-1)
        b_shifted = torch.roll(b, shifts=stride, dims=-1)

        mask = torch.zeros_like(a)
        mask[..., stride:] = 1.0

        a_new = torch.where(mask.bool(), a * a_shifted, a)
        b_new = torch.where(mask.bool(), a * b_shifted + b, b)

        a = a_new
        b = b_new

    if n_pow2 > L:
        a = a[..., :L]
        b = b[..., :L]

    final_state = b[..., -1:].squeeze(-1) if b.dim() > 0 else b

    if reverse:
        b = torch.flip(b, dims=[-1])

    return b, final_state


def associative_scan_matrix(
    A_matrices: torch.Tensor,
    Bu_terms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel associative scan for matrix-valued A (full SSM).

    Args:
        A_matrices: Transition matrices, shape (..., L, D, D).
        Bu_terms: Input projections, shape (..., L, D).

    Returns:
        Tuple (X, final_state).
    """
    L = A_matrices.shape[-3]
    D = A_matrices.shape[-1]

    n_pow2 = 1
    while n_pow2 < L:
        n_pow2 <<= 1

    batch_shape = A_matrices.shape[:-3]
    if L < n_pow2:
        pad_len = n_pow2 - L
        identity = torch.eye(D, dtype=A_matrices.dtype, device=A_matrices.device)
        identity = identity.expand(*batch_shape, pad_len, D, D)
        A_matrices = torch.cat([A_matrices, identity], dim=-3)

        zero_b = torch.zeros(*batch_shape, pad_len, D, dtype=Bu_terms.dtype,
                             device=Bu_terms.device)
        Bu_terms = torch.cat([Bu_terms, zero_b], dim=-2)

    a = A_matrices.clone()
    b = Bu_terms.clone()

    log_n = int(torch.log2(torch.tensor(n_pow2)).item())

    for d in range(log_n):
        stride = 2 ** d

        a_shifted = torch.roll(a, shifts=stride, dims=-3)
        b_shifted = torch.roll(b, shifts=stride, dims=-2)

        a_new = torch.matmul(a, a_shifted)
        b_new = torch.matmul(a, b_shifted.unsqueeze(-1)).squeeze(-1) + b

        mask = torch.ones_like(a[..., 0, 0])
        mask[..., :stride] = 0.0
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        a = torch.where(mask.bool(), a_new, a)

        mask_b = torch.ones_like(b[..., 0])
        mask_b[..., :stride] = 0.0
        mask_b = mask_b.unsqueeze(-1)
        b = torch.where(mask_b.bool(), b_new, b)

    if L < n_pow2:
        b = b[..., :L, :]

    final_state = b[..., -1, :]
    return b, final_state


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Selective scan: the core S6 / Mamba recurrence.

    Args:
        u: Input sequence, shape (B, L, D).
        delta: Step size, shape (B, L, D).
        A: State matrix (diagonal), shape (D, N).
        B: Input projection, shape (B, L, N).
        C: Output projection, shape (B, L, N).
        D: Skip connection, shape (D,) or None.
        delta_softplus: Apply softplus to delta.

    Returns:
        Output y, shape (B, L, D).
    """
    B_dim, L, D = u.shape
    N = B.shape[-1]

    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    delta_expanded = delta.unsqueeze(-1)  # (B, L, D, 1)
    A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, D, N)

    A_d = torch.exp(delta_expanded * A_expanded)
    B_d = delta_expanded * B.unsqueeze(2)

    u_expanded = u.unsqueeze(-1)
    Bu = B_d * u_expanded

    A_scan = A_d.permute(0, 2, 3, 1).reshape(B_dim * D, N, L)
    Bu_scan = Bu.permute(0, 2, 3, 1).reshape(B_dim * D, N, L)

    x, _ = associative_scan(A_scan, Bu_scan)
    x = x.reshape(B_dim, D, N, L).permute(0, 3, 1, 2)

    y = torch.sum(C.unsqueeze(2) * x, dim=-1)

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(0)

    return y


def _identity_like(x: torch.Tensor) -> torch.Tensor:
    """Return identity element for the tensor type used in scan.

    For the SSM binary operator op((a1,b1), (a2,b2)) = (a2*a1, a2*b1+b2),
    the identity element is (1, 0), since op((1,0), (a,b)) = (a*1, a*0+b) = (a,b).
    """
    if isinstance(x, tuple):
        return (torch.ones_like(x[0]), torch.zeros_like(x[1]))
    return torch.zeros_like(x)
