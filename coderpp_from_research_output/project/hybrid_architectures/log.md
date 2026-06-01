# Hybrid Architectures — Implementation Log

## Implementation Summary

The `hybrid_architectures` module implements architectures that strategically combine
attention mechanisms with sub-quadratic sequence models for efficient long-range
sequence processing. Six submodules are provided:

| Submodule | Description | Key Reference |
|-----------|-------------|---------------|
| `ssm.py` | Diagonal State Space Models (S4D-style) with HiPPO initialization, bilinear discretization, convolution/recurrent dual modes | Gu et al. (2022) |
| `attention.py` | Sliding window attention (Mistral/Longformer-style) + linear attention with kernel trick | Katharopoulos et al. (2020) |
| `h3.py` | H3 layer: shift SSM + diagonal SSM with multiplicative gating | Dao, Fu et al. (2023) |
| `gla.py` | Gated Linear Attention with chunkwise parallel form and hardware-efficient tiling | Yang et al. (2024) |
| `mega.py` | Mega-style EMA-gated attention with exponential moving average components | Ma et al. (2023) |
| `mixer.py` | Unified sequence mixer abstraction supporting attention/recurrent/hybrid modes with configurable kernel fusion paths | — |

### Architecture Overview

```
mixer.py ── SequenceMixer (ABC) ──┬── AttentionMixer (SlidingWindowAttention)
                                  ├── SSMMixer (DiagonalSSM)
                                  └── HybridMixer (fuses both, 3 fusion paths)

gla.py    ── GatedLinearAttention (multi-head) + gla_chunkwise / gla_recurrent

h3.py     ── H3Layer (ShiftSSM → gate → DiagSSM × 2)

mega.py   ── MegaLayer (EMA sub-layer → Gated Attention sub-layer)

ssm.py    ── DiagonalSSM (S4D-style, convolution + recurrent + step)
```

## Design Decisions

### 1. Diagonal SSM parameterization (S4D-style)
- **A matrix stored as `log(-A)`** to guarantee stability (A < 0 ⇒ eigenvalues in the left half-plane).
- **Bilinear (Tustin) discretization** with learnable per-channel step size Δ provides better accuracy than Euler discretization.
- **Three forward modes**: convolution (parallel training), recurrent (sequential inference), and `step()` (single-token autoregressive generation).

### 2. GLA chunkwise numerical stability
- The original chunkwise implementation used `cumprod` in float32, which underflows
  to ~1e-13 after ~15 steps of gate values in (0, 1). Dividing by these tiny values
  then amplifies rounding errors, causing the chunkwise output to diverge from the
  recurrent reference.
- **Fix**: All intra-chunk cumprod/cumsum/division operations use **float64** precision
  internally, with a clamp at 1e-30 to prevent extreme underflow. Results are cast
  back to the input dtype at the end of each chunk. This matches the chunkwise output
  to the recurrent output within rtol=1e-4.

### 3. H3 dual-SSM design
- **Shift SSM** uses small step sizes (dt_min=1e-4, dt_max=0.01) → emphasizes local patterns.
- **Diagonal SSM** uses larger step sizes (dt_min=0.01, dt_max=0.5) → captures long-range dependencies.
- Q and K branches are gated element-wise: `G = SSM(Q) ⊙ SSM(K)`, then `SSM(G ⊙ V)`.

### 4. Mega EMA + Gated Attention
- EMA serves as a **positional encoding substitute**, providing local smoothing.
- Single-head gated attention with **input gate η**: `y = η ⊙ Attn(x) + (1−η) ⊙ x`.
- Follows **pre-norm** Transformer pattern: `Norm → SubLayer → + residual`.

### 5. Unified mixer abstraction
- `SequenceMixer` ABC with `forward_attention()`, `forward_recurrent()`, `forward_hybrid()`.
- Three kernel fusion paths:
  - **Serial**: Attention → SSM (higher accuracy, sequential latency)
  - **Parallel**: α·Attention + β·SSM (lower latency, learnable blending)
  - **Interleaved**: [Attention → SSM] × N blocks (MambaFormer-style)

## Known Issues

1. **GLA numerical precision**: The chunkwise form uses internal float64 for cumprod/cumsum.
   This increases memory usage ~2× during intra-chunk computation. For very long sequences
   (L > 4096) with many chunks, this may be noticeable but is typically dwarfed by the
   attention memory savings.

2. **GLA chunk_size=0 edge case**: When `chunk_size=0` is set on the GLA module, the
   chunkwise forward path will fail (division by zero in chunk size calculation). Use
   `mode="recurrent"` instead.

3. **LinearAttention serial loop**: The current implementation uses a Python for-loop over
   the sequence length, which is slow for long sequences. A parallel scan implementation
   would be more efficient.

4. **H3 SSM convolution padding**: The causal conv1d implementation pads with `L-1` zeros
   on the left, which wastes some computation. An FFT-based convolution would be more
   efficient for long sequences but adds complexity.

5. **Softmax attention with all-`-inf` mask**: If an attention mask masks all positions
   for a query, `softmax([-inf, -inf, ...])` produces NaN. Users should ensure at least
   one position is unmasked per query.

## Test Results

**131 tests pass, 0 failures, 0 skipped.**

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_ssm.py` | 17 | Construction, convolution, recurrent, step, discretization, kernel shapes, gradients, errors |
| `test_attention.py` | 20 | Causal mask (6), sliding window attention (7), linear attention (7) |
| `test_h3.py` | 14 | Construction, convolution/recurrent modes, step, reset, gradients, dropout, variable length |
| `test_gla.py` | 24 | Recurrent API (4), chunkwise API (5 including matching), multi-head module (12), edge cases |
| `test_mega.py` | 27 | EMA (10), MegaGatedAttention (9), MegaLayer (8) |
| `test_mixer.py` | 29 | SSMMixer (5), AttentionMixer (4), enums (5), HybridMixer all fusion paths (15) |

### Test Categories Covered
- **Shape correctness**: All modules verified to produce expected output shapes.
- **Numerical correctness**: Recurrent vs convolution matching (SSM), chunkwise vs recurrent matching (GLA), step-by-step vs full-sequence (H3/SSM).
- **Causality**: Verified via gradient isolation — future tokens don't influence past outputs.
- **Gradient flow**: All modules support end-to-end autograd.
- **Edge cases**: Sequence length 1, dropout in eval mode, invalid modes, invalid parameter combinations.
- **Multi-length**: Parametrized tests for L ∈ {1, 3, 8, 16, 32, 33}.

## How to Run Tests

```bash
PYTHONPATH=modules python -m pytest modules/hybrid_architectures/ -v

# Run a specific test file:
PYTHONPATH=modules python -m pytest modules/hybrid_architectures/test_gla.py -v

# Run a specific test:
PYTHONPATH=modules python -m pytest modules/hybrid_architectures/test_gla.py::TestGlaChunkwise::test_matches_recurrent -v
```

Requirements: PyTorch ≥ 2.0.0, einops ≥ 0.7.0, pytest ≥ 7.0.0.
