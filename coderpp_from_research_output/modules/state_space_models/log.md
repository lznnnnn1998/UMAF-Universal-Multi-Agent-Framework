# State Space Models — Implementation Log

## Implementation Summary

This module implements state space model (SSM) architectures from S4 through Mamba-2, providing a complete reference implementation across 8 Python files:

| File | Lines | Description |
|------|-------|-------------|
| `core.py` | 250 | `SSMConfig` dataclass, `StateSpaceModel` ABC, matrix-based discretization (ZOH, bilinear), step size initializer |
| `hippo.py` | 207 | HiPPO matrix initialization (LegS, LegT, FouD, LagM), DPLR conversion, state reconstruction |
| `scan.py` | 415 | Blelloch scan, parallel associative scan (3 variants), sequential reference scan, selective scan for S6 |
| `s4.py` | 454 | DPLR-to-diag conversion via Woodbury, Cauchy kernel, S4 kernel (FFT + direct time-domain), `S4Kernel`/`S4Layer` |
| `s4d.py` | 337 | S4D diagonal kernels (FFT + time-domain), HiPPO eigenvalue extraction, `S4DKernel`/`S4DLayer` with 4 init modes |
| `s6.py` | 357 | Pedagogical Mamba/S6: loop-based selective scan, `MambaBlock`/`MambaModel` stack |
| `mamba.py` | 365 | Production Mamba: `MambaConfig`, `RMSNorm`, parallel scan-based `MambaBlock`/`MambaModel`, autoregressive `step()` |
| `mamba2.py` | 562 | Mamba-2/SSD: `Mamba2Config`, semiseparable matrices, chunked SSD scan, `Mamba2Block`/`SSDModel` |
| `ssm.py` | 313 | Diagonal SSM utilities, ZOH/bilinear discretization (diagonal A), conv kernel gen, `DiagonalSSM` module |
| `__init__.py` | 108 | Public API exports with name disambiguation (e.g., s4d's `S4DKernel` → `S4DDKernel`) |

### Architecture

```
__init__.py (public API)
├── hippo.py      — HiPPO matrices → A initialization
├── scan.py       — Associative scan (GPU primitive)
├── core.py       — Base classes, config, discretization
├── ssm.py        — Diagonal SSM utilities
├── s4.py         — S4 with DPLR (Cauchy kernel)
├── s4d.py        — S4D diagonal variant
├── s6.py         — Mamba/S6 selective SSM
├── mamba.py      — Full Mamba with parallel scan
└── mamba2.py     — Mamba-2 with SSD, semiseparable matrices
```

### Key Algorithms

1. **HiPPO-LegS matrix** (`hippo.py`): Lower-triangular matrix with A[n,k] = -√((2n+1)(2k+1)) for n>k, A[n,n] = -(n+1). Used to initialize state transition matrices for optimal online function approximation.

2. **DPLR parameterization** (`s4.py`): Diagonal Plus Low-Rank decomposition A = diag(Λ) - PQ^T. The Cauchy kernel formulation enables O(N+L) kernel computation via the Woodbury identity.

3. **Parallel associative scan** (`scan.py`): O(log L) parallel prefix sum for the recurrence x_t = a_t·x_{t-1} + b_t. Three variants: element-wise diagonal (`parallel_scan`), batched diagonal (`associative_scan`), and full matrix (`associative_scan_matrix`). All pad to power-of-2, use tree-based up/down sweeps.

4. **Selective SSM (Mamba/S6)** (`mamba.py`, `s6.py`): B(x), C(x), Δ(x) are learned functions of input x. Discretization: Ā = exp(Δ·A), B̄ = Δ·B. The input-dependence enables content-aware reasoning.

5. **SSD (Mamba-2)** (`mamba2.py`): Structured State Space Duality — SSM computation equivalent to semiseparable matrix multiplication. Multi-head SSMs process each head with shared A. Chunked algorithm: sequential within chunks, SSM state passing between chunks.

## Design Decisions

1. **Two Mamba implementations**: `s6.py` provides a pedagogical S6 implementation with loop-based selective scan; `mamba.py` provides a production-style Mamba using `parallel_scan` from `scan.py`. The test suite exercises `mamba.py`.

2. **Name disambiguation in __init__.py**: `s4d.py` exports `S4DKernel` which conflicts with `s4.py`'s `S4DKernel`. Resolved via aliasing: `from .s4d import S4DKernel as S4DDKernel`.

3. **FFT-based convolution**: All kernel computations use frequency-domain methods (rfft/irfft) for O(L log L) complexity rather than O(N·L) time-domain construction.

4. **Pad-to-power-of-2**: Both `parallel_scan` and `associative_scan` pad sequences to the next power of 2, enabling the tree-based parallel scan. Padding uses identity elements (1 for multiplicative, 0 for additive).

5. **Numeric stability**: Softplus-clamped Δ (dt_min=0.001, dt_max=0.1), clamped exponents (max=50.0), and safe division with epsilons throughout.

6. **No CUDA dependency**: All implementations use pure PyTorch and run on CPU. The parallel scan is GPU-friendly but not CUDA-optimized.

7. **Type hints**: Modern Python 3.10+ syntax (`X | None`, no `Optional[X]`) used throughout.

## Known Issues

- **No CUDA kernel fusion**: The parallel scan is implemented in pure PyTorch. Production Mamba uses custom CUDA kernels (selective_scan_fwd) for ~3× speedup. Not implemented here.
- **S4Layer.step()** has numerical drift vs conv mode due to per-feature kernel approximation — the step mode uses per-kernel Λ while conv uses the trained kernel directly.
- **s6.py MambaBlock** uses a sequential loop per inner dimension, making it O(L·D·N) rather than the O(L·N) of a fused scan.
- **ssd_kernel()** averages u over head_dim dimensions for the scan scalar — an approximation. The true SSD maps each head dimension through B independently.
- **associative_scan()** returns a tuple (x, final_state) but many callers use `x, _ = associative_scan(...)`. The `_selective_scan_loop` in `s6.py` correctly handles this.
- **HiPPO-FouD** requires even N (assertion-based check).
- **No support for complex B, C vectors** in scan — all scans work with real-valued tensors.
- **Mamba2 chunked scan** is sequential within chunks. The production algorithm uses matrix multiplication within chunks for parallelism.
- **Two `discretize_zoh` functions**: `core.py` has a matrix-valued version (uses `torch.matrix_exp`/`torch.linalg.solve` for full A matrices); `ssm.py` has a diagonal version (element-wise exp for diagonal A). Both are functionally correct for their respective use cases but share the same name — import carefully.
- **`_causal_conv1d` duplication**: Defined in both `s4.py` and `s4d.py` (identical implementations). Could be consolidated into `core.py` for a future cleanup pass.

## Test Results

### Summary: **76 passed, 0 failed** (0.41s, Python 3.11.15, pytorch 2.12.0)

| Test Class | Tests | Status |
|-----------|-------|--------|
| TestHiPPOLegS | 7 | All pass |
| TestHiPPOLegT | 3 | All pass |
| TestHiPPOFouD | 4 | All pass |
| TestBinaryOperator | 2 | All pass |
| TestParallelScan | 6 | All pass |
| TestDPLRConversion | 1 | All pass |
| TestS4Kernel | 3 | All pass |
| TestS4Layer | 2 | All pass |
| TestApplySSMConvolution | 2 | All pass |
| TestS4DKernel | 4 | All pass |
| TestS4DLayer | 3 | All pass |
| TestApplyS4DConvolution | 2 | All pass |
| TestDiscretization | 6 | All pass |
| TestSSMConvolution | 4 | All pass |
| TestDiagonalSSM | 4 | All pass |
| TestMambaDiscretization | 1 | All pass |
| TestMambaBlock | 6 | All pass |
| TestMamba2Config | 1 | All pass |
| TestSemiseparableMultiply | 2 | All pass |
| TestSSDKernel | 1 | All pass |
| TestMamba2Block | 4 | All pass |
| TestEndToEndSSM | 4 | All pass |
| TestNumericalStability | 4 | All pass |

### Coverage Notes
- HiPPO: All matrix types (LegS, LegT, FouD, LagM), DPLR conversion, state reconstruction
- Scan: diagonal binary operator, parallel scan, sequential reference, reverse scan, various lengths, matrix-valued scan
- S4: DPLR conversion, Cauchy kernel, FFT kernel, direct kernel, layer forward + step mode
- S4D: All four init modes (legs, inv, lin, real), kernel module, layer, convolution
- Discretization: ZOH (scalar, batched, accuracy), bilinear (scalar, comparison)
- Mamba: config, forward, step mode, gradient flow, causality
- Mamba-2: semiseparable multiply vs reference, SSD kernel, block, gradient flow
- Integration: end-to-end pipelines combining HiPPO → kernel → convolution
- Stability: long sequences (256-512), near-zero eigenvalues, zero input

## How to Run Tests

```bash
# From the working directory:
PYTHONPATH=modules python -m pytest modules/state_space_models/ -v

# Run a specific test class:
PYTHONPATH=modules python -m pytest modules/state_space_models/test_state_space_models.py::TestParallelScan -v

# Run with verbose output for failures:
PYTHONPATH=modules python -m pytest modules/state_space_models/ -v --tb=long

# Verify module imports:
PYTHONPATH=modules python -c "from state_space_models import *; print('Import OK')"
```
