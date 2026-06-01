# Build Log — Attention Mechanisms Benchmarking Framework v1.0.0

**Date**: 2026-06-01  
**Pipeline**: coderpp_from_research_output  
**Source**: research_output/research_proposal.tex (Transformer Attention Mechanisms Survey)

---

## Project Overview

A pure-PyTorch benchmarking and reference implementation framework covering **six families** of modern attention mechanisms, derived from the UMAF research pipeline's survey of recent advances in Transformer attention.

The research proposal identified three critical dimensions: hardware-aware attention optimization (FlashAttention), rotary position embeddings with length extrapolation, and attention alternatives (SSMs, linear RNNs, hybrid designs). This project provides reference implementations and benchmarking tools for all three.

---

## Module Layout

```
project/
├── main.py                        # Unified CLI entry point & demo runner
├── setup.py                       # Package configuration
├── requirements.txt               # Third-party dependencies
├── README.md                      # Installation/usage documentation
├── BUILD_LOG.md                   # This file
│
├── flash_attention/               # FlashAttention v1-v3 (+ tiling, FP8 quant)
│   ├── __init__.py                # Public API: 14 exports
│   ├── core.py                    # V1/V2/V3 forward+backward + FLOPs/mem analysis
│   ├── _tiling.py                 # Block partitioning, online softmax, recomputation
│   ├── _quantization.py           # FP8 E4M3 quantization config + matmul
│   └── test_flash_attention.py    # 68 tests (all passing)
│
├── rope_extrapolation/            # RoPE + 5 extrapolation strategies
│   ├── __init__.py                # Public API: 20+ exports
│   ├── rope.py                    # Base RoPE (dual half-dim + full-dim API)
│   ├── pi.py                      # Position Interpolation scaler
│   ├── ntk.py                     # NTK-aware scaling
│   ├── yarn.py                    # YaRN (NTK + ramp + temperature)
│   ├── dpe.py                     # Dual Chunk Position Encoding
│   ├── extrapolation.py           # Unified entry point + angle comparison
│   └── test_rope.py               # 93 tests (all passing)
│
├── state_space_models/            # SSM lineage: S4 → Mamba-2
│   ├── __init__.py                # Public API: 30+ exports (name disambiguation)
│   ├── core.py                    # SSMConfig, discretization (ZOH, bilinear)
│   ├── hippo.py                   # HiPPO matrix initializations
│   ├── scan.py                    # 3 parallel associative scan variants
│   ├── s4.py                      # S4 with DPLR/Cauchy kernel
│   ├── s4d.py                     # S4D diagonal (4 init modes)
│   ├── s6.py                      # S6/Mamba pedagogical (loop-based)
│   ├── mamba.py                   # Mamba production (parallel scan)
│   ├── mamba2.py                  # Mamba-2/SSD (semiseparable + chunked SSD)
│   ├── ssm.py                     # DiagonalSSM + apply_ssm_convolution
│   └── test_state_space_models.py # 76 tests (all passing)
│
├── linear_recurrent/              # Attention alternatives
│   ├── __init__.py                # Public API: 12+ exports
│   ├── common.py                  # RMSNorm, Swish, get_activation
│   ├── rwkv.py                    # RWKV v4-style (WKV, token-shift, TimeMix)
│   ├── xlstm.py                   # xLSTM (mLSTM matrix memory, sLSTM gating)
│   ├── griffin.py                 # Griffin (RG-LRU real-gated recurrent)
│   ├── retnet.py                  # RetNet (3 compute modes: parallel/recurrent/chunkwise)
│   └── test_linear_recurrent.py   # 71 tests + 1 skip (all passing)
│
├── hybrid_architectures/          # Attention + SSM combinations
│   ├── __init__.py                # Public API: 10+ exports
│   ├── attention.py               # SlidingWindowAttention + MultiHeadAttention
│   ├── ssm.py                     # DiagonalSSM (hybrid variant)
│   ├── h3.py                      # Hungry Hungry Hippos layer
│   ├── gla.py                     # Gated Linear Attention
│   ├── mega.py                    # Moving Average Equipped Gated Attention
│   ├── mixer.py                   # HybridMixer (3 fusion paths: serial/parallel/interleaved)
│   ├── test_attention.py          # Attention tests
│   ├── test_ssm.py                # SSM tests
│   ├── test_h3.py                 # H3 tests
│   ├── test_gla.py                # GLA tests
│   ├── test_mega.py               # Mega tests
│   └── test_mixer.py              # Mixer tests
│
└── evaluation/                    # Benchmarking harness
    ├── __init__.py                 # Public API: 12+ exports
    ├── throughput.py               # ThroughputBenchmark
    ├── memory.py                   # MemoryProfiler
    ├── perplexity.py               # PerplexityBenchmark
    ├── entropy.py                  # AttentionEntropyAnalyzer
    ├── length_extrapolation.py     # LengthExtrapolationEval
    ├── comparison.py               # ComparisonTable (LaTeX/CSV export)
    ├── roofline.py                 # RooflinePlot (GPU model database)
    └── test_evaluation.py          # Evaluation tests
```

---

## Integration Decisions & Changes

### 1. Module Independence (Design Principle)
All six modules are **deliberately standalone** — zero cross-module code dependencies. This reflects the research pipeline's independent-worker architecture. Each module self-contains its utilities (e.g., `RMSNorm` exists in both `linear_recurrent/common.py` and conceptually in `state_space_models/mamba.py`).

**Rationale**: The modules are research reference implementations. Forcing shared code would create coupling that complicates versioning, testing, and independent evolution. Users can mix and match modules freely.

### 2. Natural Integration Points (Documented, Not Forced)
While standalone, these integration points are noted:
- `rope_extrapolation` ↔ `flash_attention`: Apply RoPE to Q/K before FlashAttention (FlashAttention's docstring explicitly says "No ALiBi or RoPE — apply before calling")
- `state_space_models` ↔ `linear_recurrent`: Both implement `RMSNorm` and activation helpers
- `hybrid_architectures` depends conceptually on all three attention-style modules

### 3. Public API Design
Each module has a clean `__init__.py` exporting only the public API. Internal implementation details (private functions, helper classes) are hidden via `__all__` lists.

### 4. State Space Models Name Disambiguation
`state_space_models/__init__.py` performs explicit re-export shadowing to resolve naming conflicts:
- `MambaConfig` (from s6.py) vs `Mamba2Config` (from mamba2.py)
- `MambaBlock` → production Mamba (mamba.py), pedagogical version as `S6Block` (s6.py)
- `discretize_zoh` from core.py (matrix version) takes precedence over ssm.py (diagonal version)

### 5. Test Infrastructure
- All tests use `torch.float64` for numerical precision
- Tests run on CPU (no CUDA requirement)
- Complete in < 2 seconds for all 526 tests
- 1 intentional skip: float16 edge case in `linear_recurrent`

### 6. Python Version Compatibility
- Python >= 3.11 required (**`X | None`** syntax throughout)
- `from __future__ import annotations` for deferred evaluation
- Dataclasses for configuration (Mamba2Config, DPEConfig, etc.)

---

## Final Test Results

```
======================== 526 passed, 1 skipped in 1.47s ========================
```

### Per-Module Breakdown

| Module | Tests | Status | Notes |
|--------|-------|--------|-------|
| flash_attention | 68 | ✅ All pass | V1/V2/V3 forward/backward, FP8, FLOPs/memory |
| rope_extrapolation | 93 | ✅ All pass | All 5 methods, angle comparison, end-to-end |
| state_space_models | 76 | ✅ All pass | HiPPO, scan, S4, S4D, Mamba, Mamba-2, stability |
| linear_recurrent | 71+1 | ✅ 71 pass, 1 skip | Float16 skip (intentional) |
| hybrid_architectures | ~120 | ✅ All pass | H3, GLA, Mega, Mixer, sliding window, SSM |
| evaluation | ~98 | ✅ All pass | Throughput, memory, perplexity, entropy, comp, roofline |
| **TOTAL** | **526** | **526 pass, 1 skip** | |

### Demo Verification (main.py --quick)
All 6 module demos run successfully:
- ✅ FlashAttention v1/v2/v3: correct output shapes
- ✅ RoPE (base, PI, NTK, YaRN, DPE): correct rotation
- ✅ State Space Models (S4, S4D, Mamba, Mamba-2): correct output
- ✅ Linear Recurrent (RWKV, xLSTM, Griffin, RetNet): correct output
- ✅ Hybrid Architectures (H3, GLA, Mega, HybridMixer): all 3 fusion paths work
- ✅ Evaluation Harness: throughput, memory, perplexity, entropy, comparison, roofline

---

## How to Run

```bash
# Install
cd project
pip install -e .

# Quick smoke test
python main.py --quick

# All demos
python main.py

# Single module
python main.py --module flash
python main.py --module ssm
python main.py --module hybrid

# Full benchmark
python main.py --benchmark

# All tests
python -m pytest -v

# Specific module tests
python -m pytest flash_attention/ -v
python -m pytest rope_extrapolation/ -v
python -m pytest state_space_models/ -v
python -m pytest linear_recurrent/ -v
python -m pytest hybrid_architectures/ -v
python -m pytest evaluation/ -v
```

---

## Known Limitations

1. **Pure PyTorch — no CUDA kernels**: All implementations are for educational/reference use. Production deployment requires Triton/CUDA kernels or `torch.compile`.
2. **Float16 limited support**: Some modules (linear_recurrent) have limited float16 compatibility.
3. **Internal code duplication**: `RMSNorm` exists in both `linear_recurrent` and `state_space_models`; `_causal_conv1d` is duplicated in `s4.py` and `s4d.py`; `discretize_zoh` has implementations in `core.py` and `ssm.py`. This is deliberate for module independence.
4. **No GQA/MQA support** in FlashAttention implementation.
5. **RWKV is v4-style** — not upgraded to v5/v6 Eagle/Finch token-shift patterns.
6. **Griffin lacks local attention** component — only the RG-LRU recurrent core is implemented.

---

## Dependencies

```
numpy>=1.24.0
torch>=2.0.0
einops>=0.7.0
matplotlib>=3.7.0    # For roofline plots only
pytest>=7.0.0        # For testing only
```

---

## Conclusion

All 6 modules from the research pipeline have been successfully integrated into a single, coherent project. The framework is complete, all 526 tests pass, all demos run correctly, and the project is ready for use as a research reference and benchmarking tool.
