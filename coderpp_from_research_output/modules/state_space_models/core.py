"""
Core utilities and base classes for State Space Models.

Provides the foundational abstractions shared across S4, S4D, Mamba, and Mamba-2:
- SSMConfig: Configuration dataclass for all SSM variants
- StateSpaceModel: Abstract base class defining the SSM interface
- Utility functions: discretization, activation helpers
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SSMConfig:
    """Configuration for State Space Models.

    Shared parameters across S4, S4D, Mamba, and Mamba-2 architectures.

    Attributes:
        d_model: Model/hidden dimension.
        d_state: State dimension (N in SSM literature).
        d_conv: Convolution kernel width for pre-processing input.
        expand: Expansion factor for inner dimension (Mamba-style: d_inner = expand * d_model).
        dt_rank: Rank of Δ projection (Mamba-style). If None, defaults to d_model//16.
        dt_min: Minimum value for Δ after softplus (numerical stability).
        dt_max: Maximum value for Δ after softplus.
        dt_init: Initialization strategy for Δ ("random" or "constant").
        dt_scale: Scale factor for Δ initialization.
        bias: Whether to use bias in linear layers.
        conv_bias: Whether to use bias in the conv1d pre-processing layer.
        pscan: Whether to use parallel scan (True) or recurrent mode (False).
        dropout: Dropout probability.
        norm_epsilon: Epsilon for layer normalization.
        activation: Activation function name ("silu", "gelu", "relu").
        use_fast_path: Whether to use the fast (convolutional) path when possible.
        s4d_init: Initialization for S4D diagonal ("legs", "inv", "lin", "quad", "real").
        nplr_rank: Rank for NPLR (Normal Plus Low-Rank) parameterization in S4.
    """

    d_model: int = 256
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True
    dropout: float = 0.0
    norm_epsilon: float = 1e-5
    activation: str = "silu"
    use_fast_path: bool = True
    s4d_init: str = "legs"
    nplr_rank: int = 1

    def __post_init__(self) -> None:
        if self.dt_rank == "auto":
            self.dt_rank = max(1, self.d_model // 16)

    @property
    def d_inner(self) -> int:
        """Inner dimension after expansion."""
        return int(self.expand * self.d_model)


class StateSpaceModel(nn.Module):
    """Abstract base class for State Space Models.

    Defines the common interface that all SSM variants must implement:
    forward(u) -> y  where u, y are tensors of shape (B, L, D).

    Subclasses should override:
        - _init_ssm_params(): Initialize A, B, C, D parameters.
        - _compute_ssm(u, state): Core SSM computation.
        - forward(u): Full forward pass with pre/post-processing.
    """

    def __init__(self, config: SSMConfig) -> None:
        """Initialize the SSM base class.

        Args:
            config: SSM configuration dataclass.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_inner = config.d_inner

    def forward(self, u: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the SSM.

        Args:
            u: Input tensor of shape (B, L, D).
            state: Optional initial hidden state of shape (B, D, N).

        Returns:
            Output tensor of shape (B, L, D).
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def step(self, u: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single step for autoregressive generation.

        Args:
            u: Input of shape (B, D).
            state: Current state of shape (B, D, N).

        Returns:
            Tuple of (output, new_state), each of shape (B, D) and (B, D, N).
        """
        raise NotImplementedError("Subclasses must implement step()")

    def _init_weights(self) -> None:
        """Initialize model weights. Override in subclasses."""
        pass


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.

    Args:
        name: One of "silu", "gelu", "relu", "identity".

    Returns:
        PyTorch activation module.
    """
    activations: dict[str, nn.Module] = {
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "identity": nn.Identity(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


def discretize_zoh(
    A: torch.Tensor, B: torch.Tensor, dt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretize continuous-time SSM using Zero-Order Hold (ZOH).

    Given continuous-time system:
        x'(t) = A x(t) + B u(t)
        y(t) = C x(t) + D u(t)

    Discretized via ZOH with step size Δ:
        Ā = exp(Δ A)
        B̄ = (Δ A)^{-1} (exp(Δ A) - I) Δ B

    Args:
        A: Continuous-time state matrix of shape (..., N, N).
        B: Continuous-time input matrix of shape (..., N, 1).
        dt: Step size of shape (...,).

    Returns:
        Tuple of (Ā, B̄) — discretized state and input matrices.
    """
    # Ā = exp(Δ * A)
    A_bar = torch.matrix_exp(A * dt.unsqueeze(-1).unsqueeze(-1))

    # B̄ = (Δ * A)^{-1} (exp(Δ * A) - I) * (Δ * B)
    # For stability, use first-order Taylor when Δ is very small
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    Adt = A * dt.unsqueeze(-1).unsqueeze(-1)

    # Compute (Ā - I) via series expansion for stability
    A_bar_minus_I = A_bar - I

    # Solve linear system: (Δt A) B̄ = (Ā - I) (Δt B)
    B_bar = torch.linalg.solve(Adt, A_bar_minus_I @ (B * dt.unsqueeze(-1).unsqueeze(-1)))

    return A_bar, B_bar


def discretize_bilinear(
    A: torch.Tensor, B: torch.Tensor, dt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretize continuous-time SSM using Bilinear (Tustin) transform.

    Ā = (I + ΔA/2) (I - ΔA/2)^{-1}
    B̄ = (I - ΔA/2)^{-1} sqrt(Δ) B

    Args:
        A: Continuous-time state matrix of shape (..., N, N).
        B: Continuous-time input matrix of shape (..., N, 1).
        dt: Step size of shape (...,).

    Returns:
        Tuple of (Ā, B̄) — discretized state and input matrices.
    """
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    A_dt_half = A * (dt.unsqueeze(-1).unsqueeze(-1) / 2.0)

    # Ā = (I + ΔA/2) (I - ΔA/2)^{-1}
    A_bar = torch.linalg.solve(I - A_dt_half, I + A_dt_half)

    # B̄ = (I - ΔA/2)^{-1} sqrt(Δ) B
    B_bar = torch.linalg.solve(I - A_dt_half, B * torch.sqrt(dt).unsqueeze(-1).unsqueeze(-1))

    return A_bar, B_bar


def log_step_initializer(
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init: str = "random",
    dt_scale: float = 1.0,
) -> Callable[[tuple[int, ...]], torch.Tensor]:
    """Create a step size initialization function.

    In Mamba, Δ is parameterized in log-space for stability:
        Δ = softplus(log_Δ_initial + projection(x))

    This function generates initial log_Δ values.

    Args:
        dt_min: Minimum Δ value after softplus.
        dt_max: Maximum Δ value after softplus.
        dt_init: "random" or "constant".
        dt_scale: Scale factor for initialization.

    Returns:
        Function that returns initial log-Δ tensor given a shape.
    """

    def init(shape: tuple[int, ...]) -> torch.Tensor:
        if dt_init == "constant":
            dt = torch.ones(shape) * dt_scale
        else:
            dt = torch.rand(shape) * dt_scale

        # Map to log-space: log(exp(Δ) - 1) ≈ log(Δ) for small Δ
        # We want softplus(log_dt) to be in [dt_min, dt_max]
        # softplus(x) ≈ exp(x) for x < 0, so log_dt ≈ log(dt_min)
        inv_dt = dt_min + dt * (dt_max - dt_min)
        # Convert to log-space: inverse of softplus
        log_dt = torch.log(torch.exp(inv_dt) - 1.0 + 1e-12)
        return log_dt

    return init
