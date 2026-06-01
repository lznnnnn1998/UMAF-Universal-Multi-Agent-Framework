"""Griffin: Real-Gated Linear Recurrent Unit (RG-LRU).

Implements the RG-LRU from the Griffin paper — a linear recurrent layer
that uses input-dependent gating with real-valued (not complex) diagonal
recurrence.  Unlike attention or the WKV operator, the recurrence is
purely linear (no exponential / softmax over keys), giving it O(T·D)
complexity and constant memory.

Reference
---------
- Griffin: Mixing Gated Linear Recurrences with Local Attention
  for Efficient Language Models (De et al., DeepMind, 2024)
  https://arxiv.org/abs/2402.19427
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm


# ---------------------------------------------------------------------------
# RG-LRU core
# ---------------------------------------------------------------------------

class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit.

    The recurrence is diagonal (per-channel), making it O(T·D) in both
    time and memory::

        r_t = σ(W_a x_t + b_a)                    recurrence gate
        a_t = σ(Λ)^{c · r_t}                      input-dependent decay
        h_t = a_t ⊙ h_{t-1}  +  √(1 - a_t²) ⊙  (i_t ⊙ x̅_t)

    where *Λ* is a learnable per-channel log-decay, *c* is a scalar
    temperature, and *x̅_t* is the pre-processed input.

    Key properties:
    - Diagonal state matrix → efficient element-wise recurrence.
    - Input-dependent decay a_t → context-sensitive memory.
    - Real-valued → numerically simpler than S4 / S5.

    Args:
        dim: Feature dimension.
        expand: Expansion factor for the FFN-style input projection
            (default 2; actual expand ratio is ~2.5 as per paper).
    """

    def __init__(self, dim: int, expand: int = 2) -> None:
        super().__init__()
        self.dim = dim

        # Input projection (expand then contract)
        hidden_dim = int(dim * expand * 2.5)
        self.input_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=True)

        # Per-channel log-decay  Λ  (initialised for long-range bias)
        self.log_lambda = nn.Parameter(torch.linspace(
            math.log(0.9), math.log(0.999), dim
        ))
        # Temperature scalar for the recurrence gate
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        # Scale factor for the input gate
        self.a_param = nn.Parameter(torch.zeros(dim))
        self.b_param = nn.Parameter(torch.zeros(dim))

        self.norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RG-LRU recurrence over the sequence.

        Args:
            x: (B, T, D)

        Returns:
            (B, T, D) — processed sequence.
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        H = self.input_proj.out_features  # hidden_dim

        # Input and gate projections → hidden dim
        u = self.input_proj(x)                     # (B, T, H)
        g = self.gate_proj(x)                     # (B, T, H)

        # Recurrence gate: use the first D channels of the gate projection
        # to compute per-channel input-dependent decay.
        r = torch.sigmoid(g[..., :D])              # (B, T, D)

        # Temperature for gate modulation
        temp = torch.exp(self.log_temp)             # scalar

        # Input-dependent decay:  a_t = exp(-exp(Λ + temp * r_t))
        # where Λ is the per-channel log-decay parameter.
        log_decay = -torch.exp(
            self.log_lambda.unsqueeze(0).unsqueeze(0) + temp * r.float()
        )  # (B, T, D)
        a_t = torch.exp(log_decay).float()          # (B, T, D), in (0, 1)

        # Stabilisation term
        sqrt_term = torch.sqrt(1.0 - a_t ** 2 + 1e-8)

        # Recurrence input: first D channels of the projected input
        u_d = u[..., :D].float()                    # (B, T, D)

        # Diagonal recurrence  —  O(T·D) time, O(D) state
        state = torch.zeros(B, D, device=device, dtype=torch.float32)
        outputs_d: list[torch.Tensor] = []
        for t in range(T):
            state = a_t[:, t] * state + sqrt_term[:, t] * u_d[:, t]
            outputs_d.append(state.unsqueeze(1))

        out_d = torch.cat(outputs_d, dim=1).to(dtype)  # (B, T, D)

        # Concatenate with the non-recurrent hidden channels (direct pass-through)
        if H > D:
            out = torch.cat([out_d, u[..., D:].to(dtype)], dim=-1)   # (B, T, H)
        else:
            out = out_d[..., :H]   # truncate to hidden_dim channels

        # Output projection back to D
        out = self.out_proj(out)
        return out

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.constant_(self.log_temp, 0.0)
        self.norm.reset_parameters()


# ---------------------------------------------------------------------------
# A simpler, cleaner RG-LRU implementation
# ---------------------------------------------------------------------------

class SimpleRGLRU(nn.Module):
    """Simplified RG-LRU with diagonal recurrence in the original dimension.

    This is the clean mathematical formulation without dimension expansion::

        a_t = σ(W_a x_t + b_a)                   [gate activations]
        decay_t = exp(-exp(Λ + c · a_t))          [input-dependent decay]
        h_t = decay_t ⊙ h_{t-1}  +  √(1-decay_t²) ⊙  x_t

    where Λ is a learnable per-channel log-decay parameter.

    Args:
        dim: Feature dimension.
        temperature: Initial value for the temperature scalar c.
    """

    def __init__(self, dim: int, temperature: float = 8.0) -> None:
        super().__init__()
        self.dim = dim

        # Gate: maps input to per-channel sigmoid gate
        self.gate_proj = nn.Linear(dim, dim, bias=True)

        # Per-channel log-decay  Λ
        # Initialised to cover both short and long timescales
        self.log_lambda = nn.Parameter(torch.linspace(
            math.log(0.9), math.log(0.999), dim
        ))
        # Temperature c — controls how much the input modulates decay
        self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RG-LRU recurrence.

        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Gate:  a_t ∈ (0, 1) for each channel
        a_raw = self.gate_proj(x.float())          # (B, T, D)
        a = torch.sigmoid(a_raw)                   # (B, T, D) ∈ (0, 1)

        # Input-dependent decay
        # For each channel c:
        #   decay_t[c] = σ(exp(Λ[c]))^{c · a_t[c]}
        #              = exp(-exp(Λ[c] + log(c) + log(a_t[c])))
        #
        # Simplified:  decay_t = σ(λ)^a_t  where λ = exp(Λ)
        # In log-space:  log(decay_t) = a_t * log(σ(λ))
        #
        # More precisely (per the paper):
        #   λ_t = exp(-c · softplus(Λ) · a_t)
        # where c is the temperature.

        temp = torch.exp(self.log_temp)            # scalar
        # Λ in log-space:  softplus ensures positivity
        l = F.softplus(self.log_lambda)            # (D,), always > 0

        # Compute per-step, per-channel decay
        # decay_t[c] = exp(-temp * l[c] * a_t[c])
        log_decay = -temp * l.unsqueeze(0).unsqueeze(0) * a  # (B, T, D)
        decay = torch.exp(log_decay)               # (B, T, D) ∈ (0, 1)

        # Stability term:  sqrt(1 - decay^2)
        one_minus_decay_sq = 1.0 - decay ** 2
        one_minus_decay_sq = torch.clamp(one_minus_decay_sq, min=0.0)
        input_scale = torch.sqrt(one_minus_decay_sq + 1e-8)  # (B, T, D)

        # Run the recurrence
        state = torch.zeros(B, D, device=device, dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        for t in range(T):
            state = (decay[:, t].float() * state
                     + input_scale[:, t].float() * x[:, t].float())
            outputs.append(state.unsqueeze(1))

        return torch.cat(outputs, dim=1).to(dtype)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.constant_(self.log_temp, math.log(8.0))


# ---------------------------------------------------------------------------
# Griffin Block
# ---------------------------------------------------------------------------

class GriffinBlock(nn.Module):
    """A complete Griffin layer.

    Follows the pre-norm structure described in the paper::

        x = x + rg_lru(norm1(x))
        x = x + ffn(norm2(x))

    with an optional temporal-mixing MLP before the RG-LRU.

    Args:
        dim: Feature dimension.
        expand: FFN expansion factor (hidden = expand * dim).
        use_temporal_mixing: Whether to include the temporal-mixing MLP.
        temperature: Initial gate temperature for RG-LRU.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        use_temporal_mixing: bool = False,
        temperature: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_temporal_mixing = use_temporal_mixing

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        if use_temporal_mixing:
            self.norm_temporal = RMSNorm(dim)
            self.temporal_mlp = nn.Sequential(
                nn.Linear(dim, dim * expand, bias=False),
                nn.GELU(),
                nn.Linear(dim * expand, dim, bias=False),
            )

        self.rg_lru = SimpleRGLRU(dim, temperature=temperature)

        ffn_dim = dim * expand
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal mixing (optional)
        if self.use_temporal_mixing:
            x = x + self.dropout(self.temporal_mlp(self.norm_temporal(x)))

        # RG-LRU with pre-norm
        residual = x
        x = residual + self.dropout(self.rg_lru(self.norm1(x)))

        # FFN with pre-norm
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    def reset_parameters(self) -> None:
        self.rg_lru.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        if self.use_temporal_mixing:
            self.norm_temporal.reset_parameters()
            for module in self.temporal_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
