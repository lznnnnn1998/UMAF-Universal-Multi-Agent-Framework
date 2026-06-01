"""
S6 / Mamba: Selective State Space Model.

Implements the Mamba architecture with selective state spaces (S6).
Unlike S4 which uses fixed (input-independent) parameters, Mamba makes
B, C, and Δ input-dependent, enabling content-aware reasoning.

Key innovations over S4:
1. Selective B, C matrices: B = s_B(x), C = s_C(x) where s_B, s_C are
   learned linear projections.
2. Selective Δ step size: Δ = softplus(Δ_proj(x)), enabling the model
   to selectively attend to or ignore inputs.
3. Hardware-aware algorithm: parallel associative scan with kernel fusion
   for efficient GPU training despite the selective recurrence.

The Mamba block consists of:
    x' = A x + B(x) u     (selective state equation)
    y = C(x) x + D u      (selective output)

where A is still fixed (HiPPO-initialized diagonal), but B, C, and Δ
depend on the input u.

Key references:
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  (Gu & Dao, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

from .scan import associative_scan, selective_scan
from .hippo import hippo_legs_matrix


class MambaBlock(nn.Module):
    """
    Mamba S6 block: selective state space with input-dependent parameters.

    Architecture:
        u → SiLU(Conv1d(Linear(u))) → SSM → residual + u

    The selective SSM computes:
        B = Linear_B(x)
        C = Linear_C(x)
        Δ = softplus(Linear_Δ(x) + bias_Δ)
        A_d, B_d = discretize(A, B, Δ)
        y = selective_scan(u, A_d, B_d, C) + D * u
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = False,
        conv_bias: bool = True,
        use_fast_scan: bool = True,
    ):
        """
        Initialize Mamba block.

        Args:
            d_model: Input/output feature dimension.
            d_state: State space dimension N (typically 16).
            d_conv: Convolution kernel width for local mixing.
            expand: Expansion factor for inner dimension (typically 2).
            dt_rank: Rank of Δ projection. Defaults to ceil(d_model/16).
            dt_min, dt_max: Range for Δ initialization.
            bias: Whether to use bias in linear projections.
            conv_bias: Whether to use bias in convolution.
            use_fast_scan: Use optimized scan implementation.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.use_fast_scan = use_fast_scan

        if dt_rank is None:
            dt_rank = max(1, math.ceil(d_model / 16))

        # Input projection: d_model → (expand * 2) * d_model
        # Produces x (for input) and z (for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 1D convolution for local feature mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,  # Depthwise convolution
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # Δ projection: d_inner → dt_rank → d_inner
        self.dt_proj = nn.Sequential(
            nn.Linear(dt_rank, self.d_inner, bias=True),
            nn.Linear(self.d_inner, self.d_inner, bias=True),
        )

        # x → dt_rank projection (low-rank Δ)
        self.x_proj = nn.Linear(self.d_inner, dt_rank, bias=False)

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse softplus
        self.dt_proj[0].bias.data.copy_(inv_dt)

        # A: State matrix initialized with HiPPO
        A = hippo_legs_matrix(d_state)
        A_diag = torch.diag(A).float()
        # Store log of negative eigenvalues for stability
        self.A_log = nn.Parameter(torch.log(-A_diag))
        self.A_imag = nn.Parameter(torch.zeros(d_state))  # If complex wanted

        # D: Skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # B and C projections: d_inner → d_state per inner channel
        self.B_proj = nn.Linear(self.d_inner, self.d_inner * d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_inner * d_state, bias=False)

        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _get_A(self) -> torch.Tensor:
        """Get the state matrix A (diagonal) with negative entries."""
        return -torch.exp(self.A_log)  # (d_state,) strictly negative

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mamba block.

        Args:
            u: Input tensor, shape (B, L, d_model).

        Returns:
            Output tensor, shape (B, L, d_model).
        """
        Batch, L, D = u.shape
        A = self._get_A()  # (d_state,)

        # Input projection
        xz = self.in_proj(u)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Convolution for local mixing
        # Conv1d expects (B, C, L)
        x_conv = x.transpose(1, 2)  # (B, d_inner, L)

        # Apply depthwise conv with causal padding
        # Padding on the left for causality
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)
        # Remove extra padding on right (from causal conv)
        x_conv = x_conv[..., :L]  # (B, d_inner, L)

        # Activation
        x_act = F.silu(x_conv)  # (B, d_inner, L)
        x_act = x_act.transpose(1, 2)  # (B, L, d_inner)

        # Compute Δ = softplus(x_proj + dt_bias)
        dt_input = self.x_proj(x_act)  # (B, L, dt_rank)
        dt = self.dt_proj(dt_input)  # (B, L, d_inner)
        dt = F.softplus(dt)  # Ensure positive

        # Compute B, C from input: per inner channel, per state dimension
        B_ssm = self.B_proj(x_act).reshape(Batch, L, self.d_inner, self.d_state)
        C_ssm = self.C_proj(x_act).reshape(Batch, L, self.d_inner, self.d_state)

        # Discretize A: A_d[b,l,d,n] = exp(dt[b,l,d] * A[n])
        A_d = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)

        # Discretize B: B_d = Δ * B
        B_d = dt.unsqueeze(-1) * B_ssm  # (B, L, d_inner, d_state)

        # Apply selective scan
        y_ssm = _selective_scan_loop(x_act, A_d, B_d, C_ssm, self.D)

        # Gate with z
        y_gated = y_ssm * F.silu(z)

        # Output projection
        y = self.out_proj(y_gated)  # (B, L, d_model)

        return y


def _selective_scan_loop(
    u: torch.Tensor,
    A_d: torch.Tensor,
    B_d: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Selective scan using parallel scan over each (d_inner, d_state) pair.

    For the recurrence (per inner channel d, per state dim n):
        x_{t,d,n} = A_d[t,d,n] * x_{t-1,d,n} + B_d[t,d,n] * u[t,d]
        y_{t,d} = Σ_n C[t,d,n] * x_{t,d,n} + D[d] * u[t,d]

    Args:
        u: Input, shape (B, L, d_inner).
        A_d: Discretized A (diagonal), shape (B, L, d_inner, d_state).
        B_d: Discretized B, shape (B, L, d_inner, d_state).
        C: Output projection, shape (B, L, d_inner, d_state).
        D: Skip connection parameter, shape (d_inner,).

    Returns:
        y: Output, shape (B, L, d_inner).
    """
    B_dim, L, d_inner = u.shape
    d_state = A_d.shape[-1]
    device = u.device

    y = torch.zeros(B_dim, L, d_inner, device=device)

    for d in range(d_inner):
        # Extract per-channel parameters: (B, L, d_state)
        A_ch = A_d[:, :, d, :]  # (B, L, d_state)
        B_ch = B_d[:, :, d, :]  # (B, L, d_state)
        C_ch = C[:, :, d, :]    # (B, L, d_state)

        # Bu = B * u in the input term: (B, L, d_state)
        Bu_ch = B_ch * u[:, :, d:d + 1]

        # Flatten B and d_state dims for parallel scan: (B*d_state, L)
        A_flat = A_ch.permute(0, 2, 1).reshape(B_dim * d_state, L)
        Bu_flat = Bu_ch.permute(0, 2, 1).reshape(B_dim * d_state, L)

        # Parallel scan
        x_flat, _ = associative_scan(A_flat, Bu_flat)

        # Reshape: (B*d_state, L) → (B, L, d_state)
        x = x_flat.reshape(B_dim, d_state, L).permute(0, 2, 1)

        # Output: y[t,d] = Σ_n C[t,n] * x[t,n]
        y[:, :, d] = torch.sum(C_ch * x, dim=-1)

    # Add skip connection
    y = y + u * D.unsqueeze(0).unsqueeze(0)

    return y


class MambaModel(nn.Module):
    """
    Stack of Mamba blocks forming a complete sequence model.

    Can be used as a drop-in replacement for Transformer-based models
    on sequence tasks.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        vocab_size: int | None = None,
        dropout: float = 0.1,
    ):
        """
        Initialize Mamba model.

        Args:
            d_model: Hidden dimension.
            n_layers: Number of Mamba blocks.
            d_state: SSM state dimension.
            d_conv: Convolution kernel width.
            expand: Expansion factor.
            vocab_size: If provided, adds embedding and output layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # Optional embedding
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = None

        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        # Normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output projection (if vocab_size provided)
        if vocab_size is not None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            # Tie weights
            self.lm_head.weight = self.embedding.weight
        else:
            self.lm_head = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token ids, shape (B, L).
            inputs_embeds: Pre-computed embeddings, shape (B, L, d_model).

        Returns:
            Hidden states, shape (B, L, d_model), or logits if vocab_size set.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds required")
            x = self.embedding(input_ids)
        else:
            x = inputs_embeds

        for layer in self.layers:
            residual = x
            x = layer(x)
            x = self.dropout(x)
            x = x + residual

        x = self.norm(x)

        if self.lm_head is not None:
            x = self.lm_head(x)

        return x
