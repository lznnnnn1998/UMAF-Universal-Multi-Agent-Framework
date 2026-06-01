"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

Mamba (S6) makes SSM parameters input-dependent (selective), enabling
content-aware reasoning. Key components:
1. Selective Scan: B, C, and Δ are computed as functions of the input.
2. Hardware-Aware Algorithm: Parallel associative scan for efficient GPU execution.
3. Simplified Architecture: Gated MLP-style block with Conv1D + SSM + gate.

Reference: Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling
with Selective State Spaces. arXiv:2312.00752.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import parallel_scan, sequential_scan


@dataclass
class MambaConfig:
    """Configuration for Mamba/S6 model.

    Attributes:
        d_model: Hidden dimension.
        d_state: State dimension (N).
        d_conv: Convolution kernel width.
        expand: Expansion factor (d_inner = expand * d_model).
        dt_rank: Rank for Δ projection input.
        bias: Whether to use bias.
        conv_bias: Whether to use bias in conv1d.
        dropout: Dropout rate.
        norm_epsilon: Epsilon for RMSNorm.
        use_fast_path: Whether to use the selective scan fast path.
    """

    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"
    bias: bool = False
    conv_bias: bool = True
    dropout: float = 0.0
    norm_epsilon: float = 1e-5
    use_fast_path: bool = True

    def __post_init__(self) -> None:
        if self.dt_rank == "auto":
            self.dt_rank = max(1, math.ceil(self.d_model / 16))

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def discretize_ssm(
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Discretize selective SSM parameters (Mamba-style).

    A_bar = exp(delta * A)    (element-wise exponential)
    B_bar = delta * B         (Euler discretization)
    C stays unchanged

    Args:
        A: (N,) diagonal state matrix entries.
        B_ssm: (B, L, N) input-dependent B.
        C_ssm: (B, L, N) input-dependent C.
        delta: (B, L, N) step sizes.

    Returns:
        Tuple of (A_bar, B_bar, C_out). All shape (B, L, N).
    """
    if delta.shape[-1] != A.shape[0]:
        delta_exp = delta.unsqueeze(-1)
    else:
        delta_exp = delta

    A_bar = torch.exp(delta_exp * A)

    if B_ssm.dim() >= 3:
        B_bar = delta_exp * B_ssm
    else:
        B_bar = delta_exp * B_ssm.unsqueeze(0)

    # Ensure consistent output shapes
    if A_bar.dim() > 3:
        A_bar = A_bar.reshape(A_bar.shape[0], A_bar.shape[1], -1)
    if B_bar.dim() > 3:
        B_bar = B_bar.reshape(B_bar.shape[0], B_bar.shape[1], -1)

    return A_bar, B_bar, C_ssm


class MambaBlock(nn.Module):
    """Mamba/S6 Block: Selective State Space Model.

    Implements the core Mamba block with:
    1. Input projection (d_model → 2 * d_inner) for SSM + gate
    2. 1D causal convolution for local feature extraction
    3. Selective SSM scan (input-dependent B, C, Δ)
    4. Gated output with SiLU activation
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_inner = config.d_inner
        self.d_conv = config.d_conv
        self.dt_rank = config.dt_rank

        # Input projection: x → (z, x') for gate and SSM
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)

        # 1D convolution for local features
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
            bias=config.conv_bias,
        )

        # Δ projection: x → Δ (step size)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Δ rank reduction + B, C: x → dt_hidden + B + C
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )

        # Δ bias initialization
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.data.copy_(inv_dt)

        # Initialize A_log (state matrix eigenvalues in log-space)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))

        # Direct connection (D parameter)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block.

        Args:
            x: Input tensor of shape (B, L, D).

        Returns:
            Output tensor of shape (B, L, D).
        """
        B_shape, L, D = x.shape

        # Input projection: x → (x_ssm, z_gate)
        x_proj = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = x_proj.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Causal convolution
        x_conv = self.conv1d(x_ssm.transpose(1, 2))  # (B, d_inner, L + padding)
        x_conv = x_conv[:, :, :L]  # Remove causal padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)

        # SiLU activation after conv
        x_act = F.silu(x_conv)

        # Compute B, C, Δ projections
        x_proj_out = self.x_proj(x_act)  # (B, L, dt_rank + 2*d_state)
        dt_hidden, B_hidden, C_hidden = x_proj_out.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Δ = softplus(dt_proj(dt_hidden) + dt_bias)
        delta = F.softplus(self.dt_proj(dt_hidden))  # (B, L, d_inner)

        # A = -exp(A_log)  (diagonal state matrix)
        A = -torch.exp(self.A_log)  # (d_state,)

        # Discretize: A_bar[b,l,d,n] = exp(delta[b,l,d] * A[n])
        A_bar = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)

        # B_bar = delta * B * x_act (Euler discretization)
        B_bar = delta.unsqueeze(-1) * B_hidden.unsqueeze(2) * x_act.unsqueeze(-1)
        # (B, L, d_inner, d_state)

        # Selective scan: use parallel scan over L for each (b, d_inner, d_state)
        # Reshape to (B * d_inner * d_state, L) and scan over last dim
        A_scan = A_bar.permute(0, 2, 3, 1).reshape(-1, L)  # (B*d_inner*d_state, L)
        Bu_scan = B_bar.permute(0, 2, 3, 1).reshape(-1, L)  # (B*d_inner*d_state, L)

        # Use parallel_scan which scans over dim 0
        # parallel_scan expects (L, D) — transpose
        h_flat = parallel_scan(Bu_scan.T, A_scan.T)  # (L, B*d_inner*d_state) → transpose back
        h_flat = h_flat.T  # (B*d_inner*d_state, L)

        # Reshape back
        h = h_flat.reshape(B_shape, self.d_inner, self.d_state, L).permute(0, 3, 1, 2)
        # h: (B, L, d_inner, d_state)

        # Output: y[b,l,d] = Σ_n C[b,l,n] * h[b,l,d,n]
        y_ssm = torch.sum(h * C_hidden.unsqueeze(2), dim=-1)  # (B, L, d_inner)

        # Skip connection
        y = y_ssm + self.D * x_act

        # Gate with SiLU
        y_gated = y * F.silu(z)

        # Output projection
        y_out = self.out_proj(y_gated)
        y_out = self.dropout(y_out)

        return y_out

    def step(
        self,
        u: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single autoregressive step for Mamba.

        Args:
            u: Input of shape (B, D).
            conv_state: Conv1d state of shape (B, d_inner, d_conv-1).
            ssm_state: SSM state of shape (B, d_inner, d_state).

        Returns:
            Tuple of (y, new_conv_state, new_ssm_state).
        """
        B_shape, D = u.shape

        # Input projection
        x_proj = self.in_proj(u)  # (B, 2*d_inner)
        x_ssm, z = x_proj.chunk(2, dim=-1)  # Each (B, d_inner)

        # Convolution step (prepend new input to state buffer)
        # Initial conv_state has d_conv-1 elements; prepend new to get d_conv elements
        conv_state_new = torch.cat([
            x_ssm.unsqueeze(-1), conv_state
        ], dim=-1)  # (B, d_inner, d_conv)
        # Apply conv kernel weights as a simple linear combination
        x_conv = (self.conv1d.weight.squeeze(1).unsqueeze(0) * conv_state_new).sum(dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias.unsqueeze(0)

        # Update conv state: drop oldest element to maintain d_conv-1 size
        new_conv_state = conv_state_new[..., :-1]

        x_act = F.silu(x_conv)

        # Compute B, C, delta
        x_proj_out = self.x_proj(x_act)  # (B, dt_rank + 2*d_state)
        dt_hidden, B_hidden, C_hidden = x_proj_out.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(dt_hidden))  # (B, d_inner)
        A = -torch.exp(self.A_log)  # (d_state,)

        # Discretize
        A_bar = torch.exp(delta.unsqueeze(-1) * A)  # (B, d_inner, d_state)
        B_bar = delta.unsqueeze(-1) * B_hidden.unsqueeze(1) * x_act.unsqueeze(-1)

        # SSM step
        new_ssm_state = A_bar * ssm_state + B_bar
        y_ssm = torch.sum(new_ssm_state * C_hidden.unsqueeze(1), dim=-1)  # (B, d_inner)

        y = y_ssm + self.D * x_act
        y_gated = y * F.silu(z)

        y_out = self.out_proj(y_gated)
        return y_out, new_conv_state, new_ssm_state


class MambaModel(nn.Module):
    """Full Mamba model composed of stacked Mamba blocks.

    Architecture:
        x → [Block 1] → [Block 2] → ... → [Block N] → RMSNorm → Linear
    """

    def __init__(
        self,
        config: MambaConfig,
        n_layers: int = 24,
        vocab_size: int | None = None,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(n_layers)
        ])

        self.norm_f = RMSNorm(config.d_model, eps=config.norm_epsilon)

        if num_classes is not None:
            self.lm_head = nn.Linear(config.d_model, num_classes, bias=False)
        elif vocab_size is not None:
            self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        else:
            self.lm_head = None

        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self._init_weights()

    def _init_weights(self) -> None:
        if hasattr(self, 'embedding'):
            nn.init.normal_(self.embedding.weight, std=0.02)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.in_proj.weight)
            nn.init.xavier_uniform_(layer.out_proj.weight)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'embedding'):
            x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        if self.lm_head is not None:
            x = self.lm_head(x)

        return x
