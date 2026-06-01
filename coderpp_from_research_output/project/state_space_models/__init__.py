"""
State Space Models: S4 through Mamba-2.

Implements state space model architectures for sequence modeling:
- S4: Structured State Space with DPLR parameterization
- S4D: Diagonal State Space Model (simplified S4)
- S6/Mamba: Selective SSM with input-dependent parameters
- Mamba-2/SSD: Structured State Space Duality

Key components:
- hippo: HiPPO matrix initialization and utilities
- scan: Parallel associative scan for efficient recurrence
- s4: Core S4 with Cauchy kernel and DPLR
- s4d: S4D diagonal variant
- s6: Mamba/S6 selective state space
- mamba2: Mamba-2 with SSD (semiseparable matrices)
"""

from .hippo import (
    hippo_legs_matrix,
    hippo_legs_matrix_numpy,
    hippo_legs_matrix_vectorized,
    hippo_legt_matrix,
    hippo_fout_matrix,
    hippo_lagm_matrix,
    hippo_to_dplr,
    hippo_initial_state,
    hippo_reconstruct,
)

from .scan import (
    binary_operator_diag,
    binary_operator_matmul,
    blelloch_scan,
    associative_scan,
    associative_scan_matrix,
    selective_scan,
)

from .s4 import (
    cauchy_kernel,
    s4_kernel,
    s4_kernel_conv,
    S4DKernel,
    S4Layer,
)

from .s4d import (
    s4d_kernel_diag,
    s4d_kernel_fft,
    S4DKernel as S4DDKernel,
    S4DLayer,
)

from .s6 import (
    MambaBlock,
    MambaModel,
)

from .mamba2 import (
    Mamba2Config,
    semiseparable_matrix,
    semiseparable_multiply,
    ssd_kernel,
    Mamba2Block,
    SSDModel,
)

__all__ = [
    # HiPPO
    "hippo_legs_matrix",
    "hippo_legs_matrix_numpy",
    "hippo_legs_matrix_vectorized",
    "hippo_legt_matrix",
    "hippo_fout_matrix",
    "hippo_lagm_matrix",
    "hippo_to_dplr",
    "hippo_initial_state",
    "hippo_reconstruct",
    # Scan
    "binary_operator_diag",
    "binary_operator_matmul",
    "blelloch_scan",
    "associative_scan",
    "associative_scan_matrix",
    "selective_scan",
    # S4
    "cauchy_kernel",
    "s4_kernel",
    "s4_kernel_conv",
    "S4DKernel",
    "S4Layer",
    # S4D
    "s4d_kernel_diag",
    "s4d_kernel_fft",
    "S4DDKernel",
    "S4DLayer",
    # S6 (Mamba)
    "MambaBlock",
    "MambaModel",
    # Mamba-2 (SSD)
    "Mamba2Config",
    "semiseparable_matrix",
    "semiseparable_multiply",
    "ssd_kernel",
    "Mamba2Block",
    "SSDModel",
]
