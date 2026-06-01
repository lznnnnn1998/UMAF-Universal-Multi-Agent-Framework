# Linear Recurrent Module — Implementation Log

## Implementation Summary

This module implements four families of linear recurrent and gated architectures
as attention alternatives for sequence modeling. All implementations use pure
PyTorch with type hints and docstrings.

### Files

| File | Description | Lines |
|------|-------------|-------|
| `common.py` | Shared utilities: RMSNorm, LayerNorm, SquaredReLU, SwiGLU, get_activation | 140 |
| `rwkv.py` | RWKV-style token-mixing (WKV operator) and channel-mixing blocks | 250 |
| `xlstm.py` | xLSTM with matrix memory (mLSTM) and scalar gating (sLSTM) | 389 |
| `griffin.py` | Griffin RG-LRU (two variants) and GriffinBlock | 311 |
| `retnet.py` | Retention with parallel, recurrent, and chunkwise modes | 517 |
| `__init__.py` | Public API exports | 85 |
| `test_linear_recurrent.py` | 72 unit tests | 645 |

Total: ~2,337 lines across 7 files.

### Architecture Overview

#### 1. RWKV (`rwkv.py`)
- **WKVOperator**: Per-channel exponential decay recurrence.
  - Recurrence: `a_t = decay * a_{t-1} + exp(k_t) * v_t`, `b_t = decay * b_{t-1} + exp(k_t)`
  - Output: `wkv_t = r_t * (num / den)` with per-channel learnable decay and bonus.
  - O(T·D) time, O(D) state.
- **TokenShift**: Learned mixing of current and previous tokens via per-channel `mu`.
- **ChannelMixing**: Gated FFN with squared-ReLU (`max(0,k)^2`) activation.
- **RWKVBlock**: Pre-LayerNorm → TimeMix → ChannelMix, with residual connections.

#### 2. xLSTM (`xlstm.py`)
- **mLSTMCell**: Matrix memory — each head stores `C ∈ R^{d_qk × d_v}`.
  - Update: `C_t = f_t * C_{t-1} + i_t * (v_t ⊗ k_t)`
  - Normalizer: `n_t = f_t * n_{t-1} + i_t * k_t`
  - Retrieval: `h_t = o_t ⊙ (C_t q_t / max(|n_t^T q_t|, 1))`
  - Exponential input gate, sigmoid or exponential forget gate.
- **sLSTMCell**: Scalar LSTM with gating.
  - `c_t = f_t ⊙ c_{t-1} + i_t ⊙ z_t`, normalized by `n_t`.
  - Multi-head via parallel cells rather than attention heads.
- **xLSTMBlock**: Pre-norm cell → residual → FFN.

#### 3. Griffin (`griffin.py`)
- **SimpleRGLRU**: Clean diagonal recurrence with input-dependent decay.
  - Gate: `a_t = sigmoid(W_g * x_t)`
  - Decay: `λ_t = exp(-c * softplus(Λ) * a_t)` where Λ is per-channel log-decay.
  - Update: `h_t = λ_t ⊙ h_{t-1} + sqrt(1 - λ_t^2) ⊙ x_t`
  - Real-valued → numerically simpler than complex-valued SSMs (S4/S5).
- **RGLRU** (expanded variant): Projects to hidden dim, uses first D channels for recurrence.
- **GriffinBlock**: Pre-norm RG-LRU → FFN, with optional temporal-mixing MLP.

#### 4. RetNet (`retnet.py`)
- **Three compute modes**, mathematically equivalent:
  - **Parallel**: `Retention(X) = (Q K^T ⊙ D) V` where `D_{ij} = γ^{i-j}`. O(T²) for training.
  - **Recurrent**: `S_t = γ S_{t-1} + K_t^T V_t`, `o_t = Q_t S_t`. O(T·D²) for inference.
  - **Chunkwise**: Hybrid — parallel within chunks (O(C²) per chunk), recurrent state between chunks (O(D²)).
- **MultiScaleRetention**: Multi-head with per-head γ in [γ_min, γ_max]. GroupNorm + swish gate.
- **RetNetBlock**: Pre-norm retention → FFN.

## Design Decisions

### 1. Pure PyTorch with explicit recurrence loops
All recurrent computations use simple Python `for` loops over the time dimension
rather than CUDA kernels or compiled scans. This keeps the code readable and
portable at the cost of speed for very long sequences (>10K tokens). The loops
run in O(T·D) time which is fast enough for most testing and prototyping.

### 2. Float32 accumulation for numerical stability
All recurrence state accumulators (WKV numerator/denominator, mLSTM matrix memory,
RG-LRU state) are cast to `float32` internally, even when the input is `float16`
or `bfloat16`. This prevents gradient underflow in the decay terms. Outputs are
cast back to the input dtype.

### 3. Shared `common.py` module
Normalization layers (RMSNorm, LayerNorm) and activations are factored out into
a shared module that all four architectures import, avoiding duplication.

### 4. State-as-tuple pattern
Recurrent states are returned as tuples/dicts rather than being mutated in-place.
Each module's `forward()` can accept an optional `state` argument and returns
`(output, new_state)`. This is consistent across xLSTM and the RWKV/RetNet
recurrent variants.

### 5. Pre-norm architecture throughout
All blocks follow the pre-LayerNorm convention (norm → sublayer → residual add),
consistent with modern Transformer practice. This improves training stability.

### 6. Chunkwise retention off-by-one fix
The initial chunkwise implementation had an off-by-one error in the cross-chunk
decay computation. The recurrent state S represents the accumulator at the *last*
token of the previous chunk (position `c*C - 1`). For position `j` in the current
chunk, the correct decay from S is `γ^{j+1}`, not `γ^j`. Fixed during testing.

### 7. Broadcasting fix in chunkwise D_c
The within-chunk decay matrix used `g.squeeze(-1) ** dist_c` which fails when
`num_heads ≠ chunk_size`. Fixed by reshaping `g` to `(H, 1, 1)` and unsqueezing
`dist_c` for correct broadcasting.

## Known Issues

1. **No CUDA kernel optimization**: The Python `for`-loop scan over the sequence
   is O(T·D) but not optimized for GPU. For production use with sequences > 10K,
   a Triton/CUDA kernel or `torch.compile` would be needed.

2. **mLSTM memory overhead**: The matrix memory C has shape `(B, H, d_qk, d_v)`,
   which is `B * H * d_qk * d_v` elements per layer. With typical values
   (B=8, H=8, d_qk=64, d_v=64), this is ~1M floats per layer — workable but heavy.

3. **float16 not fully supported**: Linear layer weights are stored in float32,
   so full float16 inference is not tested. The recurrent computations use float32
   internally for stability, which means mixed-precision training would need
   careful autocast configuration.

4. **RWKV WKV operator is v4-style**: Does not include the token-shifting time-mix
   formulation of v5/v6 (Eagle/Finch). The WKV operator here computes `exp(k)`
   rather than using the more recent `wkv` CUDA kernel from the RWKV project.

5. **No weight tying or initialization schemes**: The implementations use default
   Xavier uniform initialization. The original papers often use specific
   initialization (e.g., RWKV's special init for time_decay) which may affect
   convergence on large-scale training.

6. **Chunkwise padding**: The chunkwise retention pads sequences to be divisible
   by `chunk_size`, which adds minor overhead for non-aligned lengths.

7. **Griffin temporal mixing is optional**: The full Griffin architecture uses
   a mix of RG-LRU blocks and local attention blocks. Only the RG-LRU + MLP
   variant is implemented here; local attention is not included.

## Test Results

### Summary
- **71 passed**, 1 skipped, 0 failed
- Test duration: ~0.43s
- Python: 3.11.15, PyTorch: 2.12.0

### Test Coverage by Module

| Module | Tests | Status |
|--------|-------|--------|
| Common (RMSNorm, LayerNorm, SquaredReLU, SwiGLU, get_activation) | 12 | All pass |
| RWKV (token_shift, WKVOperator, TimeMixBlock, ChannelMixBlock, RWKVBlock) | 11 | All pass |
| xLSTM (mLSTMCell, sLSTMCell, xLSTMBlock) | 11 | All pass |
| Griffin (RGLRU, SimpleRGLRU, GriffinBlock) | 6 | All pass |
| RetNet (decay matrix, parallel, recurrent, chunkwise, MultiScaleRetention, RetNetBlock) | 25 | All pass |
| Stacking (multiple blocks) | 4 | All pass |
| Edge cases (single token, small dim, float16) | 3 | 2 pass, 1 skipped |

### Key Tests

- **Causality**: RWKV WKV and SimpleRGLRU verify that position `t` output does
  not depend on positions `> t` (truncation test).
- **Equivalence**: RetNet parallel ↔ recurrent, recurrent ↔ chunkwise equivalence
  verified within `atol=1e-4`.
- **State management**: xLSTM and RetNet recurrent modes tested with both
  zero-initialized and user-provided initial states.
- **Edge cases**: Single-token sequences (T=1), small model dimensions (D=16),
  non-divisible chunk sizes all handled correctly.

## How to Run Tests

```bash
cd /Users/zhinan/universal_multi_agent_framework/coderpp_from_research_output
PYTHONPATH=modules python -m pytest modules/linear_recurrent/ -v
```

Or from any directory with absolute paths:

```bash
PYTHONPATH=/Users/zhinan/universal_multi_agent_framework/coderpp_from_research_output/modules \
  /opt/homebrew/opt/python@3.11/bin/python3.11 \
  -m pytest modules/linear_recurrent/ -v
```
