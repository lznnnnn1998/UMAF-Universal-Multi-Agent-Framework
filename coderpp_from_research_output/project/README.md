# Attention Mechanisms Benchmarking Framework

Pure PyTorch implementations of modern attention mechanisms, state space models, linear recurrent architectures, and hybrid designs — with a comprehensive evaluation harness for benchmarking.

## Overview

This project provides research-grade, pure-PyTorch implementations of the key attention mechanism families that have driven NLP architecture advances from 2020–2024. Every implementation prioritizes algorithmic fidelity and educational clarity over production performance (no CUDA kernels).

### Modules

| Module | Description | Key Algorithms |
|--------|-------------|---------------|
| **flash_attention** | IO-aware tiled attention | FlashAttention v1/v2/v3, online softmax, FP8 quantization |
| **rope_extrapolation** | Rotary Position Embeddings & length generalization | RoPE, PI, NTK-aware, YaRN, DPE |
| **state_space_models** | Continuous-time → discrete SSMs | S4, S4D, Mamba/S6, Mamba-2/SSD |
| **linear_recurrent** | Sub-quadratic attention alternatives | RWKV, xLSTM, Griffin, RetNet |
| **hybrid_architectures** | Attention + SSM combinations | H3, GLA, Mega, HybridMixer |
| **evaluation** | Comprehensive benchmarking harness | Throughput, memory, perplexity, entropy, roofline |

### Test Coverage

**308+ tests** across all modules with 0 failures.

## Installation

```bash
cd project
pip install -e .
# Or for development with test dependencies:
pip install -e ".[test]"
```

### Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- einops >= 0.7.0
- matplotlib >= 3.7.0 (for roofline plots)
- pytest >= 7.0.0 (for running tests)

## Quick Start

```bash
# Run all demos
python main.py

# Quick smoke test
python main.py --quick

# Demo a single module
python main.py --module flash
python main.py --module ssm

# Run comprehensive benchmark
python main.py --benchmark

# Run all tests
python main.py --test
# Or directly:
python -m pytest -v
```

## Usage Examples

### FlashAttention

```python
from flash_attention import FlashAttentionV1, flash_attention_v1
import torch

q = torch.randn(2, 8, 512, 64)  # (batch, heads, seq, head_dim)
k = torch.randn(2, 8, 512, 64)
v = torch.randn(2, 8, 512, 64)

attn = FlashAttentionV1(Br=64, Bc=64, causal=True)
output = attn(q, k, v)
```

### RoPE with Length Extrapolation

```python
from rope_extrapolation import RoPE, YaRNRoPE, compare_angles

rope = RoPE(dim=64, max_seq_len=2048)
yarn = YaRNRoPE(dim=64, max_seq_len=2048, scale_factor=4.0)

x = torch.randn(2, 2048, 64)
rotated = rope(x)
extrapolated = yarn(x)
```

### State Space Models

```python
from state_space_models import MambaBlock, Mamba2Block

x = torch.randn(2, 256, 512)  # (batch, seq_len, d_model)
mamba = MambaBlock(d_model=512, d_state=16)
output = mamba(x)
```

### Linear Recurrent Architectures

```python
from linear_recurrent import RWKVBlock, RetNetBlock, xLSTMBlock

x = torch.randn(2, 128, 256)
rwkv = RWKVBlock(d_model=256)
retnet = RetNetBlock(d_model=256)
xlstm = xLSTMBlock(d_model=256)
```

### Hybrid Architectures

```python
from hybrid_architectures import HybridMixer, KernelFusionPath

mixer = HybridMixer(d_model=512, n_heads=8, d_state=64,
                    fusion_path=KernelFusionPath.INTERLEAVED)
output = mixer(x, mode="hybrid")
```

### Evaluation

```python
from evaluation import (
    ThroughputBenchmark, MemoryProfiler,
    ComparisonTable, RooflinePlot,
)

# Throughput
bench = ThroughputBenchmark(gpu_model="H100")
result = bench.benchmark(attention_fn, seq_len=2048, d_model=1024)

# Roofline analysis
plotter = RooflinePlot(gpu_model="A100")
plotter.add_measurement("FlashAttention v3", 10.0, 250.0)
plotter.save("roofline.pdf")

# Comparison table
table = ComparisonTable()
table.add_row("FlashAttention v3", complexity="O(N)", memory="O(N)")
table.add_row("Mamba-2", complexity="O(N log N)", memory="O(N)")
print(table.to_latex())
```

## Project Structure

```
project/
  flash_attention/          # FlashAttention v1-v3 + tiling + FP8
  rope_extrapolation/        # RoPE + PI + NTK + YaRN + DPE
  state_space_models/        # S4, S4D, Mamba/S6, Mamba-2/SSD
  linear_recurrent/          # RWKV, xLSTM, Griffin, RetNet
  hybrid_architectures/      # H3, GLA, Mega, HybridMixer
  evaluation/                # Benchmarking harness
  main.py                   # Entry point & demo runner
  setup.py                  # Package configuration
  requirements.txt          # Dependencies
  BUILD_LOG.md              # Integration build log
```

## Running Tests

```bash
cd project
python -m pytest -v                          # All tests
python -m pytest flash_attention/ -v         # Single module
python -m pytest -v -k "test_forward"        # Filter by name
```

## Design Decisions

- **Pure PyTorch**: No CUDA kernels — prioritizes clarity and portability over raw performance
- **Python >= 3.11**: Uses `X | None` syntax throughout (no `Optional[X]`)
- **Standalone modules**: Each module is self-contained with zero cross-module code dependencies
- **Comprehensive tests**: 308+ tests covering algorithmic correctness, edge cases, and numerical stability
- **Educational focus**: Every algorithm is implemented with pedagogical clarity — reading the code teaches the algorithm

## License

MIT License
