"""H3 (Hungry Hungry Hippos) layer.

H3 combines two flavours of state-space model with multiplicative (gated)
interactions.  The architecture, from Dao, Fu, Saab, Thomas, Rudra & Ré
(2023), interleaves a **shift SSM** (captures local patterns) with a
**diagonal SSM** (captures long-range dependencies) via element-wise gates.

Layer structure
---------------

::

    x → Proj → [Q, K, V]
         Q → ShiftSSM  ─┐
         K → DiagSSM   ─┤→ ⊙ → DiagSSM → OutProj → y
         V ─────────────┘
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import DiagonalSSM


class H3Layer(nn.Module):
    """H3 (Hungry Hungry Hippos) layer.

    Implements the SSM-gated mixing pattern:

    1. Project input into Q, K, V.
    2. Apply shift SSM to Q (local / short-range).
    3. Apply diagonal SSM to K (global / long-range).
    4. Element-wise gate: ``G = ShiftSSM(Q) ⊙ DiagSSM(K)``.
    5. Apply another diagonal SSM to ``G ⊙ V``.
    6. Output projection.

    Args:
        d_model: Model (hidden) dimension.
        d_state: SSM state dimension (default 64).
        dropout: Dropout probability after the output projection.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Shift SSM — small step size for local emphasis
        self.shift_ssm = DiagonalSSM(d_state, d_model, dt_min=0.0001, dt_max=0.01)

        # Diagonal SSM for K branch — larger step size for longer memory
        self.diag_ssm_k = DiagonalSSM(d_state, d_model, dt_min=0.01, dt_max=0.5)

        # Diagonal SSM for the gated-V branch
        self.diag_ssm_v = DiagonalSSM(d_state, d_model, dt_min=0.01, dt_max=0.5)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "convolution",
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, L, d_model)`` input.
            mode: SSM forward mode — ``"convolution"`` or ``"recurrent"``.

        Returns:
            ``(B, L, d_model)`` output.
        """
        # 1. Project to Q, K, V
        qkv = self.qkv_proj(x)                                       # (B, L, 3D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)                        # each (B, L, D)

        # 2. Shift SSM on Q → captures local structure
        q_ssm = self.shift_ssm(q, mode=mode)
        if isinstance(q_ssm, tuple):
            q_ssm = q_ssm[0]

        # 3. Diagonal SSM on K → captures long-range structure
        k_ssm = self.diag_ssm_k(k, mode=mode)
        if isinstance(k_ssm, tuple):
            k_ssm = k_ssm[0]

        # 4. Gating: element-wise multiply
        g = q_ssm * k_ssm                                            # (B, L, D)

        # 5. Gated V → second diagonal SSM
        gated_v = g * v                                              # (B, L, D)
        o = self.diag_ssm_v(gated_v, mode=mode)
        if isinstance(o, tuple):
            o = o[0]

        # 6. Output projection
        out = self.out_proj(o)
        return self.dropout(out)

    def step(
        self,
        x_t: torch.Tensor,
        state_q: torch.Tensor,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Single autoregressive step.

        Args:
            x_t: ``(B, d_model)`` input at current step.
            state_q: ``(B, d_model, d_state)`` shift SSM state.
            state_k: ``(B, d_model, d_state)`` diag SSM (K) state.
            state_v: ``(B, d_model, d_state)`` diag SSM (V) state.

        Returns:
            ``(y_t, (new_state_q, new_state_k, new_state_v))``.
        """
        qkv = self.qkv_proj(x_t)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q_out, state_q = self.shift_ssm.step(q, state_q)
        k_out, state_k = self.diag_ssm_k.step(k, state_k)

        g = q_out * k_out
        gated_v = g * v
        o, state_v = self.diag_ssm_v.step(gated_v, state_v)

        y = self.out_proj(o)
        return y, (state_q, state_k, state_v)

    def reset_states(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return zero-initialised states for all three SSMs."""
        return (
            self.shift_ssm.reset_state(batch_size, device),
            self.diag_ssm_k.reset_state(batch_size, device),
            self.diag_ssm_v.reset_state(batch_size, device),
        )
