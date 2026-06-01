"""
Mamba-2: Structured State Space Duality (SSD).

Implements the Mamba-2 architecture which reveals a fundamental duality
between state space models and structured matrix multiplication.

Key insight (State Space Duality):
The SSM computation y = SSM(A, B, C)(u) can be expressed as:
    y = M * u  where M is a semiseparable matrix

A semiseparable matrix M has entries:
    M[i, j] = C_i^T * A_{i:j} * B_j

where A_{i:j} = A_i * A_{i-1} * ... * A_{j+1} (product over range).

This connects SSMs to:
1. Linear attention: y = (Q K^T * M_mask) * V
2. Matrix multiplication with structured sparsity

Key references:
- "Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality" (Dao & Gu, 2024)
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hippo import hippo_legs_matrix


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class Mamba2Config:
    """Configuration for Mamba-2 / SSD model.

    Attributes:
        d_model: Hidden dimension.
        d_state: State dimension (N) per head.
        n_heads: Number of SSM heads.
        d_conv: Convolution kernel width.
        expand: Expansion factor (d_inner = expand * d_model).
        chunk_size: Block size for chunked SSD computation.
        bias: Whether to use bias in linear layers.
        conv_bias: Whether to use bias in conv1d.
        dropout: Dropout rate.
        norm_epsilon: Epsilon for layer normalization.
    """

    d_model: int = 256
    d_state: int = 64
    n_heads: int = 8
    d_conv: int = 4
    expand: int = 2
    chunk_size: int = 64
    bias: bool = False
    conv_bias: bool = True
    dropout: float = 0.0
    norm_epsilon: float = 1e-5

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model


# ============================================================================
# Semiseparable Matrix Multiplication
# ============================================================================


def semiseparable_matrix(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """
    Construct the semiseparable matrix M for an SSM.

    For an SSM with diagonal A and input-dependent B, C:
        M[i, j] = C_i^T * (Π_{k=j+1}^{i} A_k) * B_j  for i >= j
        M[i, j] = 0                                      for i < j

    Args:
        A: Diagonal A factors, shape (L, N) or (N,).
        B: Input vectors B, shape (L, N).
        C: Output vectors C, shape (L, N).

    Returns:
        M: Lower-triangular semiseparable matrix, shape (L, L).
    """
    L, N = B.shape

    if A.dim() == 1:
        A = A.unsqueeze(0).expand(L, -1)  # (L, N)

    device = B.device

    # Cumulative log-products for efficient range products
    log_A = torch.log(A + 1e-8)  # (L, N)
    cum_log_A = torch.cumsum(log_A, dim=0)  # (L, N)

    M = torch.zeros(L, L, device=device)

    for i in range(L):
        for j in range(i + 1):
            if i == j:
                M[i, j] = torch.sum(C[i] * B[i])
            else:
                log_prod = cum_log_A[i] - cum_log_A[j]
                prod = torch.exp(log_prod)
                M[i, j] = torch.sum(C[i] * prod * B[j])

    return M


def semiseparable_multiply(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Multiply the semiseparable matrix M (defined by A, B, C) by vector x.

    Computes y = M * x using the SSM recurrence, avoiding explicit
    construction of the L×L matrix. O(L) complexity.

    For each channel:
        h_t = A_t * h_{t-1} + B_t * x_t   (state update)
        y_t = C_t * h_t                     (output)

    Args:
        A: Diagonal A factors, shape (B, L, N).
        B: Input vectors B, shape (B, L, N).
        C: Output vectors C, shape (B, L, N).
        x: Input signal, shape (B, L, D).

    Returns:
        y: Output signal, shape (B, L, D).
    """
    B_dim, L, D = x.shape
    N = A.shape[-1]
    device = x.device

    # For each channel, run the SSM recurrence
    y = torch.zeros(B_dim, L, D, device=device)

    for d in range(D):
        h = torch.zeros(B_dim, N, device=device)  # state
        for t in range(L):
            h = A[:, t, :] * h + B[:, t, :] * x[:, t, d:d + 1]
            y[:, t, d] = torch.sum(C[:, t, :] * h, dim=-1)

    return y


# ============================================================================
# SSD Kernel
# ============================================================================


def ssd_kernel(
    u: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    chunk_size: int = 64,
    dt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the SSD (Structured State Space Duality) kernel output.

    Implements the chunked SSD algorithm:
    1. Chunk the sequence into blocks of size chunk_size
    2. Within each block: compute via SSM recurrence
    3. Across blocks: pass SSM state

    Args:
        u: Input tensor, shape (B, L, H, P) or (B, L, D).
        A: State transition per head, shape (H, N) or (H,).
        B_mat: Input-dependent B vectors, shape (B, L, H, N) or (B, L, N).
        C_mat: Input-dependent C vectors, shape (B, L, H, N) or (B, L, N).
        chunk_size: Block size for chunked computation.
        dt: Optional step sizes, shape (B, L, H).

    Returns:
        y: Output tensor, same shape as u.
    """
    # Handle 3D input: (B, L, D) — single head, treat D as both head and feat dims
    if u.dim() == 3:
        u = u.unsqueeze(2)  # (B, L, 1, D)
        if B_mat.dim() == 3:
            B_mat = B_mat.unsqueeze(2)  # (B, L, 1, N)
        if C_mat.dim() == 3:
            C_mat = C_mat.unsqueeze(2)
        if A.dim() == 1:
            A = A.unsqueeze(0)  # (1, N)
        squeeze_out = True
    else:
        squeeze_out = False

    B_dim, L, n_heads, head_dim = u.shape
    N = B_mat.shape[-1]
    device = u.device

    # Ensure A has correct shape (H, N)
    if A.dim() == 1:
        A = A.unsqueeze(-1).expand(n_heads, N)
    elif A.dim() == 2 and A.shape[0] == 1:
        A = A.expand(n_heads, -1)

    # Apply dt if provided: A_d = exp(dt * A)
    if dt is not None:
        dt_exp = dt.unsqueeze(-1)  # (B, L, H, 1)
        A_exp = A.unsqueeze(0).unsqueeze(0)  # (1, 1, H, N)
        A_d_full = torch.exp(dt_exp * A_exp)  # (B, L, H, N)
    else:
        A_d_full = A.unsqueeze(0).unsqueeze(0).expand(B_dim, L, -1, -1)

    # Chunked SSD recurrence
    y = torch.zeros(B_dim, L, n_heads, head_dim, device=device)
    state = torch.zeros(B_dim, n_heads, N, device=device)  # (B, H, N)

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        clen = end - start

        A_chunk = A_d_full[:, start:end]  # (B, clen, H, N)
        B_chunk = B_mat[:, start:end]  # (B, clen, H, N)
        C_chunk = C_mat[:, start:end]  # (B, clen, H, N)
        u_chunk = u[:, start:end]  # (B, clen, H, head_dim)

        h = state  # (B, H, N) — carry state across chunks

        for t in range(clen):
            # Decay state: h = A[t] * h
            h = A_chunk[:, t, :, :] * h  # (B, H, N)

            # Input contribution: h += B[t] * u[t, p] for each head
            # B[t]: (B, H, N), u[t]: (B, H, head_dim)
            # For each head h_dim: h[:, h, :] += B[:, h, :] * u[:, h, p]
            # Average u over head_dim for efficient scan
            u_scalar = u_chunk[:, t, :, :].mean(dim=-1)  # (B, H)
            h = h + B_chunk[:, t, :, :] * u_scalar.unsqueeze(-1)

            # Output: y[t, p] = sum_n C[t, n] * h[n] for each head
            y_h = torch.sum(C_chunk[:, t, :, :] * h, dim=-1)  # (B, H)
            # Expand to head_dim dimensions
            y_t = y_h.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, H, head_dim)
            y[:, start + t] = y_t

    if squeeze_out:
        y = y.squeeze(2)  # (B, L, D)

    return y


# ============================================================================
# Mamba2 Block
# ============================================================================


class Mamba2Block(nn.Module):
    """
    Mamba-2 / SSD block using structured state space duality.

    Implements the dual view of SSMs as structured matrix multiplication.
    Uses multi-head SSMs similar to multi-head attention but with O(N)
    complexity through the semiseparable structure.

    Architecture:
        u → SiLU(Conv1d(Linear(u))) → SSM (multi-head) → gate → out_proj
    """

    def __init__(self, config: Mamba2Config) -> None:
        """
        Initialize Mamba-2 block.

        Args:
            config: Mamba2Config dataclass.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.n_heads = config.n_heads
        self.d_inner = config.d_inner
        self.d_conv = config.d_conv
        self.chunk_size = config.chunk_size

        # Input projection: d_model → 2 * d_inner (x for SSM, z for gate)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)

        # 1D depthwise convolution for local mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=config.conv_bias,
        )

        # A: state transition per head (log-space for positivity)
        self.A_log = nn.Parameter(torch.log(torch.ones(config.n_heads)))

        # Δ / dt projection: d_inner → n_heads (per-head step size)
        self.dt_proj = nn.Linear(self.d_inner, config.n_heads, bias=True)
        nn.init.constant_(self.dt_proj.bias, 1.0)

        # B projection: d_inner → n_heads * d_state
        self.B_proj = nn.Linear(self.d_inner, config.n_heads * config.d_state, bias=False)

        # C projection: d_inner → n_heads * d_state
        self.C_proj = nn.Linear(self.d_inner, config.n_heads * config.d_state, bias=False)

        # D: skip connection per head
        self.D = nn.Parameter(torch.ones(config.n_heads))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            u: Input tensor of shape (B, L, d_model).

        Returns:
            Output tensor of shape (B, L, d_model).
        """
        B, L, D = u.shape

        # Input projection
        xz = self.in_proj(u)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Causal convolution for local mixing
        x_conv = x.transpose(1, 2)  # (B, d_inner, L)
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))  # Causal padding
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv[..., :L]  # Remove extra padding
        x_act = F.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)

        # Get A (per head, fixed, negative for stability)
        A = -torch.exp(self.A_log)  # (n_heads,)

        # Get Δ (per head, input-dependent)
        dt = F.softplus(self.dt_proj(x_act))  # (B, L, n_heads)

        # Get B, C (input-dependent)
        B_ssm = self.B_proj(x_act).reshape(B, L, self.n_heads, self.d_state)
        C_ssm = self.C_proj(x_act).reshape(B, L, self.n_heads, self.d_state)

        # SSM computation using block-decomposition
        y_ssm = _ssd_chunked_scan(
            x_act, A, B_ssm, C_ssm, dt, self.D, self.chunk_size
        )  # (B, L, d_inner)

        # Gate with SiLU
        y = y_ssm * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y


def _ssd_chunked_scan(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Block-decomposed SSD scan.

    Uses the chunked algorithm:
    1. Within chunk: sequential SSM recurrence.
    2. Across chunks: SSM state passing.

    Args:
        u: Input activations, shape (B, L, d_inner).
        A: Per-head transition, shape (n_heads,).
        B: Per-head input vectors, shape (B, L, n_heads, d_state).
        C: Per-head output vectors, shape (B, L, n_heads, d_state).
        dt: Step sizes, shape (B, L, n_heads).
        D: Skip connection per head, shape (n_heads,).
        chunk_size: Block size.

    Returns:
        y: Output, shape (B, L, d_inner).
    """
    B_dim, L, D_inner = u.shape
    n_heads = A.shape[0]
    d_state = B.shape[-1]
    device = u.device

    y = torch.zeros(B_dim, L, D_inner, device=device)

    # Map d_inner to heads: each head covers roughly d_inner // n_heads channels.
    # Use ceil and pad so reshape always succeeds (handles non-divisible sizes).
    d_head = max(1, int(math.ceil(D_inner / n_heads)))
    padded_dim = n_heads * d_head
    pad_amount = padded_dim - D_inner

    # State per head per channel group: (B, n_heads, d_state)
    state = torch.zeros(B_dim, n_heads, d_state, device=device)

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        clen = end - start

        u_chunk = u[:, start:end]  # (B, clen, D_inner)
        B_chunk = B[:, start:end]  # (B, clen, n_heads, d_state)
        C_chunk = C[:, start:end]  # (B, clen, n_heads, d_state)
        dt_chunk = dt[:, start:end]  # (B, clen, n_heads)

        # Discretize A per timestep: A_d[t, h] = exp(dt[t,h] * A[h])
        A_d_chunk = torch.exp(dt_chunk * A.unsqueeze(0).unsqueeze(0))  # (B, clen, n_heads)

        for t in range(clen):
            # Decay state: h = A_d[t] * h
            A_d_t = A_d_chunk[:, t, :].unsqueeze(-1)  # (B, n_heads, 1)
            state = A_d_t * state  # (B, n_heads, d_state)

            # Input contribution: h += B[t, h, :] * u_proj[t, h]
            # Pad u_t so it can be reshaped to (n_heads, d_head) cleanly
            u_t = u_chunk[:, t]  # (B, D_inner)
            if pad_amount > 0:
                u_t = F.pad(u_t, (0, pad_amount))  # (B, padded_dim)
            u_head = u_t.reshape(B_dim, n_heads, d_head)
            u_scalar = u_head.mean(dim=-1)  # (B, n_heads)

            state = state + B_chunk[:, t, :, :] * u_scalar.unsqueeze(-1)

            # Output: y[t] = C[t] * h
            C_t = C_chunk[:, t, :, :]  # (B, n_heads, d_state)
            y_h = torch.sum(C_t * state, dim=-1)  # (B, n_heads)

            # Expand back to padded_dim, then truncate to D_inner
            y_t = y_h.unsqueeze(-1).expand(B_dim, n_heads, d_head)
            y[:, start + t] = y_t.reshape(B_dim, padded_dim)[:, :D_inner]

    # Skip connection
    y = y + u * D.mean()

    return y


# ============================================================================
# SSD Model (full stack)
# ============================================================================


class SSDModel(nn.Module):
    """
    Full Mamba-2 / SSD model as a stack of SSD blocks.

    Provides an efficient linear-complexity alternative to attention
    for long sequences.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 64,
        vocab_size: int | None = None,
        dropout: float = 0.1,
    ):
        """
        Initialize SSD model.

        Args:
            d_model: Hidden dimension.
            n_layers: Number of SSD blocks.
            n_heads: Number of SSM heads.
            d_state: State dimension per head.
            d_conv: Convolution kernel width.
            expand: Expansion factor.
            chunk_size: Block size for chunked computation.
            vocab_size: Optional vocabulary size for LM head.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model

        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = None

        config = Mamba2Config(
            d_model=d_model,
            d_state=d_state,
            n_heads=n_heads,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([
            Mamba2Block(config) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if vocab_size is not None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            if self.embedding is not None:
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
            inputs_embeds: Embeddings, shape (B, L, d_model).

        Returns:
            Hidden states or logits, shape (B, L, d_model) or (B, L, vocab_size).
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds required")
            x = self.embedding(input_ids)
        else:
            x = inputs_embeds

        for layer in self.layers:
            x = layer(x) + x
            x = self.dropout(x)

        x = self.norm(x)

        if self.lm_head is not None:
            x = self.lm_head(x)

        return x
