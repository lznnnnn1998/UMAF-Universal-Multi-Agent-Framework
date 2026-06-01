"""Common utilities for linear recurrent architectures.

Provides normalization layers, activation functions, and initialization
utilities shared across RWKV, xLSTM, Griffin, and RetNet implementations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Computes: x * weight / sqrt(mean(x^2) + eps).
    More efficient than LayerNorm — no mean subtraction, no bias term.
    Used by RWKV, Griffin, and RetNet.

    Args:
        dim: Feature dimension.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        return (x_f32 / rms * self.weight.float()).to(dtype)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class LayerNorm(nn.Module):
    """Standard Layer Normalization.

    Args:
        dim: Feature dimension.
        eps: Small constant for numerical stability.
        bias: Whether to include a learnable bias term.
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps) * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class SquaredReLU(nn.Module):
    """Squared ReLU activation: max(0, x)^2.

    Used in RWKV's channel-mixing block as the key activation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2


_ACTIVATION_MAP: dict[str, nn.Module] = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}


def get_activation(act_name: str) -> nn.Module:
    """Get activation function module by name.

    Args:
        act_name: One of 'relu', 'gelu', 'silu'/'swish',
                  'squared_relu', 'sigmoid', 'tanh'.

    Returns:
        An nn.Module implementing the activation.

    Raises:
        ValueError: If the activation name is not recognised.
    """
    if act_name == "squared_relu":
        return SquaredReLU()
    if act_name in _ACTIVATION_MAP:
        return _ACTIVATION_MAP[act_name]
    raise ValueError(f"Unknown activation: {act_name}. "
                     f"Valid options: {list(_ACTIVATION_MAP.keys())} + squared_relu")


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: SiLU(gate) * up, projected back to dim.

    Splits the intermediate projection into gate and up branches,
    applies SiLU to the gate, and multiplies element-wise.

    Args:
        dim: Input/output dimension.
        hidden_dim: Intermediate dimension (default: 4 * dim, rounded to
            a multiple of 256 for GPU efficiency).
        dropout: Dropout probability applied after the output projection.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 4 * 2 / 3)
            hidden_dim = int(math.ceil(hidden_dim / 256.0) * 256)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

    def reset_parameters(self):
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
