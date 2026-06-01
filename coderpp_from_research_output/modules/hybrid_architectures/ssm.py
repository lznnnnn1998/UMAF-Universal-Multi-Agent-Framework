"""State Space Model (SSM) core with diagonal parameterization.

Implements diagonal state space models (S4D-style) supporting:
- HiPPO-inspired initialization for long-range memory
- Bilinear (Tustin) discretization with learnable step sizes
- Convolution mode for parallel training
- Recurrent mode for autoregressive inference
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalSSM(nn.Module):
    """Diagonal State Space Model (S4D-style).

    Implements the linear time-invariant SSM:

        x'(t) = A x(t) + B u(t)
        y(t)  = C x(t) + D u(t)

    where A ∈ R^{N×N} is diagonal, B ∈ R^{N×1}, C ∈ R^{1×N}, D ∈ R.
    The SSM is applied independently per channel with a shared state dimension.

    Discretized via bilinear (Tustin) transform with learnable step size Δ:

        Ā = (I + ΔA/2)(I − ΔA/2)⁻¹
        B̄ = (I − ΔA/2)⁻¹ Δ B

    Supports two forward modes:
    - **convolution**: Compute the SSM kernel and apply via causal conv1d.
      Parallel over sequence length — suitable for training.
    - **recurrent**: Step through time sequentially.
      O(L) serial — suitable for autoregressive inference.

    Args:
        d_state: State dimension N (number of SSM state variables per channel).
        d_model: Number of independent SSM channels (typically model dim).
        dt_min: Minimum step size in absolute units (default 0.001).
        dt_max: Maximum step size in absolute units (default 0.1).
    """

    def __init__(
        self,
        d_state: int,
        d_model: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model

        # --- A matrix (diagonal) ---
        # Store as log(−A) to ensure A < 0 for stability.
        # HiPPO-LegS-inspired: eigenvalues spaced proportionally.
        A_init = torch.linspace(0.5, d_state - 0.5, d_state)
        self.A_log = nn.Parameter(torch.log(A_init))

        # --- B, C projections ---
        # One set of (B, C) per channel × per state dimension.
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        # --- Skip connection D ---
        self.D = nn.Parameter(torch.ones(d_model))

        # --- Step size Δ (per channel) ---
        dt_uniform = torch.rand(d_model) * (dt_max - dt_min) + dt_min
        self.log_dt = nn.Parameter(torch.log(dt_uniform))

    def _discretize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Discretize A and B via bilinear (Tustin) transform.

        Returns:
            A_bar: ``(d_model, d_state)`` discrete diagonal.
            B_bar: ``(d_model, d_state)`` discrete input vector.
        """
        A = -torch.exp(self.A_log)                     # (N,)   — A < 0
        dt = torch.exp(self.log_dt)                    # (D,)

        # Broadcast for element-wise operations:  (D,1) × (1,N) → (D,N)
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0)
        denom = 1.0 - dtA / 2.0

        A_bar = (1.0 + dtA / 2.0) / denom               # (D, N)
        B_bar = dt.unsqueeze(-1) * self.B / denom        # (D, N)
        return A_bar, B_bar

    def _compute_kernel(self, A_bar: torch.Tensor, B_bar: torch.Tensor, L: int) -> torch.Tensor:
        """Compute the SSM convolution kernel.

        K[d, ℓ] = Σₙ C[d,n] · A_bar[d,n]^ℓ · B_bar[d,n]   for ℓ = 0,…,L−1

        Args:
            A_bar: ``(d_model, d_state)`` discrete state matrix (diagonal).
            B_bar: ``(d_model, d_state)`` discrete input vector.
            L: Sequence length.

        Returns:
            K: ``(d_model, L)`` convolution kernel (causal).
        """
        # A_bar: (D, N)  →  expand to (D, N, L) by powering
        powers = torch.arange(L, device=A_bar.device, dtype=A_bar.dtype)  # (L,)
        A_pow = A_bar.unsqueeze(-1) ** powers                             # (D, N, L)

        # K[d, ℓ] = Σₙ C[d,n] · A_pow[d,n,ℓ] · B_bar[d,n]
        K = (self.C.unsqueeze(-1) * A_pow * B_bar.unsqueeze(-1)).sum(dim=1)  # (D, L)
        return K

    def forward(
        self,
        u: torch.Tensor,
        mode: str = "convolution",
        state: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply SSM to the input sequence.

        Args:
            u: ``(B, L, d_model)`` input tensor.
            mode: ``"convolution"`` (parallel training) or
                  ``"recurrent"`` (sequential inference).
            state: ``(B, d_model, d_state)`` initial state for recurrent mode.
                   If ``None``, zero-initialised.

        Returns:
            **convolution**: ``y`` — ``(B, L, d_model)`` output.
            **recurrent**: ``(y, final_state)`` where *y* is ``(B, L, d_model)``
            and *final_state* is ``(B, d_model, d_state)``.
        """
        A_bar, B_bar = self._discretize()

        if mode == "convolution":
            y = self._conv_forward(u, A_bar, B_bar)
            return y
        elif mode == "recurrent":
            y, final_state = self._recurrent_forward(u, A_bar, B_bar, state)
            return y, final_state
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'convolution' or 'recurrent'.")

    def _conv_forward(
        self, u: torch.Tensor, A_bar: torch.Tensor, B_bar: torch.Tensor
    ) -> torch.Tensor:
        """Convolution mode — parallel over the time dimension."""
        B, L, D = u.shape
        K = self._compute_kernel(A_bar, B_bar, L)          # (D, L)

        # Causal conv1d: input (B, D, L), kernel (D, 1, L), groups=D
        u_bdl = u.permute(0, 2, 1)                         # (B, D, L)
        K_d1l = K.unsqueeze(1)                             # (D, 1, L)
        y_bdl = F.conv1d(u_bdl, K_d1l, groups=D, padding=L - 1)[:, :, :L]
        y = y_bdl.permute(0, 2, 1)                         # (B, L, D)

        # Skip connection
        y = y + u * self.D.view(1, 1, -1)
        return y

    def _recurrent_forward(
        self,
        u: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent mode — step through time sequentially."""
        B, L, D = u.shape
        N = self.d_state

        if state is None:
            state = torch.zeros(B, D, N, device=u.device, dtype=u.dtype)

        outputs: list[torch.Tensor] = []
        A_bar_bd = A_bar.unsqueeze(0)    # (1, D, N)
        B_bar_bd = B_bar.unsqueeze(0)    # (1, D, N)
        C_bd = self.C.unsqueeze(0)       # (1, D, N)

        for t in range(L):
            # h_t = Ā ⊙ h_{t−1} + B̄ ⊙ u_t   (element-wise per channel/state)
            u_t = u[:, t, :].unsqueeze(-1)             # (B, D, 1)
            state = A_bar_bd * state + B_bar_bd * u_t   # (B, D, N)
            # y_t = Σₙ C_{d,n} · h_{t,d,n}
            y_t = (C_bd * state).sum(dim=-1)            # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)                  # (B, L, D)
        y = y + u * self.D.view(1, 1, -1)
        return y, state

    @torch.no_grad()
    def step(
        self, u_t: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single recurrent step — for autoregressive inference.

        Args:
            u_t: ``(B, d_model)`` input at the current time step.
            state: ``(B, d_model, d_state)`` previous SSM state.

        Returns:
            ``(y_t, new_state)`` where *y_t* is ``(B, d_model)`` and
            *new_state* is ``(B, d_model, d_state)``.
        """
        A_bar, B_bar = self._discretize()

        new_state = (
            A_bar.unsqueeze(0) * state
            + B_bar.unsqueeze(0) * u_t.unsqueeze(-1)
        )
        y_t = (self.C.unsqueeze(0) * new_state).sum(dim=-1) + self.D * u_t
        return y_t, new_state

    def reset_state(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Return a zero-initialised state tensor.

        Args:
            batch_size: Batch dimension size.
            device: Target device (defaults to parameter device).

        Returns:
            ``(batch_size, d_model, d_state)`` zero tensor.
        """
        return torch.zeros(batch_size, self.d_model, self.d_state,
                           device=device or self.A_log.device,
                           dtype=self.A_log.dtype)
