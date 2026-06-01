"""xLSTM: Extended Long Short-Term Memory.

Implements the two cell variants from Beck et al. (2024):

* **mLSTM** — Matrix memory cell that stores key-value associations in
  a full matrix, enabling efficient retrieval via query-key matching.
* **sLSTM** — Scalar memory cell with exponential gating, similar to
  classical LSTM but with modern stabilisation techniques.

Reference
---------
- xLSTM: Extended Long Short-Term Memory (Beck et al., 2024)
  https://arxiv.org/abs/2405.04517
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm, LayerNorm


# ---------------------------------------------------------------------------
# mLSTM — Matrix Memory cell
# ---------------------------------------------------------------------------

class mLSTMCell(nn.Module):
    """Matrix LSTM cell with covariance-based memory storage.

    Each head maintains a **matrix** memory  C ∈ R^{d_qk × d_v}  that
    accumulates outer products  v_t k_t^T , scaled by exponential input
    and forget gates.  Queries retrieve information from memory via
    matrix-vector product  C·q_t , normalised by an accumulated key
    vector  n_t ∈ R^{d_qk}.

    **Key equations** (per head)::

        i_t = exp(W_i^T x_t)       input gate  (exponential)
        f_t = σ(W_f^T x_t)   or    exp(...)    forget gate
        o_t = σ(W_o x_t)           output gate

        C_t = f_t · C_{t-1}  +  i_t · v_t k_t^T       [matrix update]
        n_t = f_t · n_{t-1}  +  i_t · k_t             [normaliser]
        h_t = o_t ⊙ (C_t q_t  /  max(|n_t^T q_t|, 1)) [retrieval]

    Args:
        d_model:   Total model dimension (must be divisible by num_heads).
        d_qk:      Query / key dimension per head.
        d_v:       Value dimension per head  (default = d_qk).
        num_heads: Number of parallel memory heads.
        use_exp_forget: If True, use exponential forget gate (like input).
            If False, use sigmoid forget gate (bounded, may be more stable).
    """

    def __init__(
        self,
        d_model: int,
        d_qk: int = 64,
        d_v: int | None = None,
        num_heads: int = 4,
        use_exp_forget: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_qk = d_qk
        self.d_v = d_v if d_v is not None else d_qk
        self.use_exp_forget = use_exp_forget

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        # Per-head projections: all heads computed in parallel via reshaping.
        self.w_q = nn.Linear(d_model, num_heads * d_qk, bias=False)
        self.w_k = nn.Linear(d_model, num_heads * d_qk, bias=False)
        self.w_v = nn.Linear(d_model, num_heads * self.d_v, bias=False)

        # Scalar gates (1 value per head)
        self.w_i = nn.Linear(d_model, num_heads, bias=True)   # input gate
        self.w_f = nn.Linear(d_model, num_heads, bias=True)   # forget gate
        self.w_o = nn.Linear(d_model, num_heads * self.d_v, bias=False)  # output gate

        # Output projection: heads → d_model
        self.out_proj = nn.Linear(num_heads * self.d_v, d_model, bias=False)

        self.norm = LayerNorm(d_model)

    def _gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute input and forget gates.

        Returns:
            i_gate: (B, T, H) — exponential input gate.
            f_gate: (B, T, H) — forget gate (exp or sigmoid).
        """
        i_log = self.w_i(x)  # (B, T, H)
        f_log = self.w_f(x)  # (B, T, H)
        i_gate = torch.exp(i_log)  # always exponential
        if self.use_exp_forget:
            f_gate = torch.exp(f_log)
        else:
            f_gate = torch.sigmoid(f_log)
        return i_gate, f_gate

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass over a sequence.

        Args:
            x:     (B, T, d_model)  input sequence.
            state: Optional initial state  (C, n)  with shapes
                   C: (B, H, d_qk, d_v),  n: (B, H, d_qk).
                   If None, both are initialised to zero.

        Returns:
            output: (B, T, d_model)
            state:  Final  (C, n)  tuple.
        """
        B, T, _ = x.shape
        H = self.num_heads
        dk = self.d_qk
        dv = self.d_v
        device = x.device
        dtype = x.dtype

        # Pre-compute projections  (B, T, H·d)
        x_norm = self.norm(x)
        q = self.w_q(x_norm).view(B, T, H, dk)          # (B, T, H, d_qk)
        k = self.w_k(x_norm).view(B, T, H, dk)          # (B, T, H, d_qk)
        v = self.w_v(x_norm).view(B, T, H, dv)          # (B, T, H, d_v)
        o_gate = torch.sigmoid(self.w_o(x_norm))         # (B, T, H*d_v)
        i_gate, f_gate = self._gates(x)                  # (B, T, H)

        # Initialise or unpack state
        if state is None:
            C = torch.zeros(B, H, dk, dv, device=device, dtype=torch.float32)
            n = torch.zeros(B, H, dk, device=device, dtype=torch.float32)
        else:
            C, n = state
            C, n = C.float(), n.float()

        outputs: list[torch.Tensor] = []
        for t in range(T):
            # Gates at this timestep
            i_t = i_gate[:, t]   # (B, H)
            f_t = f_gate[:, t]   # (B, H)

            k_t = k[:, t].float()   # (B, H, d_qk)
            v_t = v[:, t].float()   # (B, H, d_v)
            q_t = q[:, t].float()   # (B, H, d_qk)

            # Memory update:  C_t = f_t·C_{t-1} + i_t·v_t·k_t^T
            #  outer product:  (B, H, d_v, 1) × (B, H, 1, d_qk) → (B, H, d_v, d_qk)
            #  ... but we store C as (B, H, d_qk, d_v)
            outer = torch.einsum("bhi,bhj->bhij", v_t, k_t)          # (B, H, d_v, d_qk)
            outer = outer.transpose(-2, -1)                          # (B, H, d_qk, d_v)
            C = (f_t.unsqueeze(-1).unsqueeze(-1) * C
                 + i_t.unsqueeze(-1).unsqueeze(-1) * outer)

            # Normaliser update:  n_t = f_t·n_{t-1} + i_t·k_t
            n = f_t.unsqueeze(-1) * n + i_t.unsqueeze(-1) * k_t   # (B, H, d_qk)

            # Retrieval:  h_t = C_t @ q_t   (C: B,H,d_qk,d_v  q: B,H,d_qk)
            h_raw = torch.einsum("bhij,bhi->bhj", C, q_t)           # (B, H, d_v)

            # Normalisation by |n^T q|
            n_dot_q = torch.einsum("bhi,bhi->bh", n, q_t)           # (B, H)
            norm_factor = torch.clamp(torch.abs(n_dot_q), min=1.0)  # (B, H)
            h_normalised = h_raw / norm_factor.unsqueeze(-1)        # (B, H, d_v)

            outputs.append(h_normalised)

        # Stack and project back to d_model
        h_seq = torch.stack(outputs, dim=1).to(dtype)               # (B, T, H, d_v)
        h_seq = h_seq.reshape(B, T, H * dv)
        out = self.out_proj(h_seq)                                  # (B, T, d_model)

        return out, (C, n)

    def reset_parameters(self) -> None:
        for module in [self.w_q, self.w_k, self.w_v, self.w_i,
                       self.w_f, self.w_o, self.out_proj]:
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a zero-initialised state for a given batch size."""
        C = torch.zeros(batch_size, self.num_heads, self.d_qk, self.d_v, device=device)
        n = torch.zeros(batch_size, self.num_heads, self.d_qk, device=device)
        return C, n


# ---------------------------------------------------------------------------
# sLSTM — Scalar memory cell
# ---------------------------------------------------------------------------

class sLSTMCell(nn.Module):
    """Scalar LSTM cell with exponential gating.

    Each cell maintains its own scalar hidden and cell state.  Multiple
    cells stack to form a full sLSTM block.  Exponential input/forget
    gates allow strong forgetting and rapid memory updates.

    **Key equations**::

        i_t = exp(W_i x_t)
        f_t = σ(W_f x_t)   or   exp(W_f x_t)
        z_t = tanh(W_z x_t)
        o_t = σ(W_o x_t)

        c_t = f_t ⊙ c_{t-1} + i_t ⊙ z_t
        n_t = f_t ⊙ n_{t-1} + i_t
        h_t = o_t ⊙ (c_t / n_t)

    Args:
        d_model: Model dimension (= number of cells).
        use_exp_forget: Whether to use exponential forget gate.
    """

    def __init__(self, d_model: int, use_exp_forget: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_exp_forget = use_exp_forget

        self.norm = LayerNorm(d_model)
        self.w_z = nn.Linear(d_model, d_model, bias=False)
        self.w_i = nn.Linear(d_model, d_model, bias=True)
        self.w_f = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass over a sequence.

        Args:
            x:     (B, T, d_model).
            state: Optional  (c, n)  initial cell / normaliser states,
                   each of shape (B, d_model).

        Returns:
            output: (B, T, d_model)
            state:  Final (c, n).
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        x_norm = self.norm(x)
        z_all = torch.tanh(self.w_z(x_norm))         # (B, T, D)
        o_all = torch.sigmoid(self.w_o(x_norm))      # (B, T, D)
        i_all = torch.exp(self.w_i(x_norm))          # (B, T, D)
        if self.use_exp_forget:
            f_all = torch.exp(self.w_f(x_norm))
        else:
            f_all = torch.sigmoid(self.w_f(x_norm))  # (B, T, D)

        if state is None:
            c = torch.zeros(B, D, device=device, dtype=torch.float32)
            n = torch.zeros(B, D, device=device, dtype=torch.float32)
        else:
            c, n = state
            c, n = c.float(), n.float()

        outputs: list[torch.Tensor] = []
        for t in range(T):
            i_t = i_all[:, t].float()
            f_t = f_all[:, t].float()
            z_t = z_all[:, t].float()
            o_t = o_all[:, t].float()

            c = f_t * c + i_t * z_t
            n = f_t * n + i_t
            h_t = o_t * (c / (n + 1e-8))
            outputs.append(h_t.unsqueeze(1))

        h_seq = torch.cat(outputs, dim=1).to(dtype)
        return h_seq, (c, n)

    def reset_parameters(self) -> None:
        for module in [self.w_z, self.w_i, self.w_f, self.w_o]:
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a zero-initialised state for a given batch size."""
        c = torch.zeros(batch_size, self.d_model, device=device)
        n = torch.zeros(batch_size, self.d_model, device=device)
        return c, n


# ---------------------------------------------------------------------------
# xLSTM Block (wraps mLSTM or sLSTM with residual + FFN)
# ---------------------------------------------------------------------------

class xLSTMBlock(nn.Module):
    """A full xLSTM layer: cell → residual → FFN.

    Supports both mLSTM and sLSTM cells.  The block follows the
    pre-norm Transformer convention:

        x = x + cell(norm1(x))
        x = x + ffn(norm2(x))

    Args:
        d_model: Model dimension.
        cell_type: 'mlstm' or 'slstm'.
        d_qk: Query/key dim per head (mLSTM only).
        d_v: Value dim per head (mLSTM only).
        num_heads: Number of heads (mLSTM only).
        use_exp_forget: Exponential forget gate.
        ffn_mult: FFN hidden = ffn_mult * d_model.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        cell_type: str = "mlstm",
        d_qk: int = 64,
        d_v: int | None = None,
        num_heads: int = 4,
        use_exp_forget: bool = False,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cell_type = cell_type

        if cell_type == "mlstm":
            self.cell = mLSTMCell(
                d_model=d_model, d_qk=d_qk, d_v=d_v,
                num_heads=num_heads, use_exp_forget=use_exp_forget,
            )
        elif cell_type == "slstm":
            self.cell = sLSTMCell(d_model=d_model, use_exp_forget=use_exp_forget)
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}. Use 'mlstm' or 'slstm'.")

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        ffn_dim = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple | None]:
        """Forward pass.

        Args:
            x: (B, T, d_model).
            state: Optional cell state.

        Returns:
            output: (B, T, d_model)
            state:  Updated cell state.
        """
        # xLSTM cell with pre-norm
        cell_out, new_state = self.cell(self.norm1(x), state)
        x = x + self.dropout(cell_out)

        # FFN with pre-norm
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, new_state

    def reset_parameters(self) -> None:
        self.cell.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
