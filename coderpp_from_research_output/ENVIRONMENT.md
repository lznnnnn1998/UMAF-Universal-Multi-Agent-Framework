# Project Environment

## Python
- Path: `/opt/homebrew/opt/python@3.11/bin/python3.11`
- Alias: `python3` → `python3.11`
- Version: Python 3.11.15

## Conda
- Environment: none (conda not available)
- Using system Python via Homebrew

## Pip
- Path: `/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/bin/pip3`
- Version: pip 26.0.1

## Working Directory
- Path: `/Users/zhinan/universal_multi_agent_framework/coderpp_from_research_output`

## Required Packages (requirements.txt)
```
numpy>=1.24.0
torch>=2.0.0
einops>=0.7.0
matplotlib>=3.7.0
psutil>=5.9.0
pytest>=7.0.0
tqdm>=4.65.0
scipy>=1.10.0
```

## Pre-installed Packages (snapshot: 2026-06-01)
| Package           | Version   |
|-------------------|-----------|
| torch             | 2.12.0    |
| numpy             | 2.4.6     |
| scipy             | 1.17.1    |
| einops            | 0.8.2     |
| matplotlib        | 3.10.9    |
| psutil            | 7.2.2     |
| pytest            | 9.0.3     |
| tqdm              | 4.67.3    |
| transformers      | 5.9.0     |
| huggingface_hub   | 1.17.0    |
| fsspec            | 2026.4.0  |
| filelock          | 3.29.0    |
| safetensors       | 0.7.0     |
| sympy             | 1.14.0    |
| packaging         | 26.2      |
| networkx          | 3.6.1     |
| anyio             | 4.13.0    |
| httpx             | 0.28.1    |
| certifi           | 2026.5.20 |
| typing-extensions | 4.15.0    |

## Installation Results (2026-06-01)

Ran `python3 -m pip install -r requirements.txt` — all packages already satisfied.

**All 8 required packages are available:**
- numpy 2.4.6 (>=1.24.0 ✓)
- torch 2.12.0 (>=2.0.0 ✓)
- einops 0.8.2 (>=0.7.0 ✓)
- matplotlib 3.10.9 (>=3.7.0 ✓)
- psutil 7.2.2 (>=5.9.0 ✓)
- pytest 9.0.3 (>=7.0.0 ✓)
- tqdm 4.67.3 (>=4.65.0 ✓)
- scipy 1.17.1 (>=1.10.0 ✓)

**Key transitive dependencies:**
- sympy 1.14.0 (torch dependency)
- networkx 3.6.1 (torch dependency)
- packaging 26.2 (matplotlib dependency)
- pillow 12.2.0 (matplotlib dependency)

## Module Structure
```
modules/
├── flash_attention/       # Module 1: FlashAttention v1–v4
├── rope_extrapolation/    # Module 2: RoPE + Length Extrapolation
├── state_space_models/    # Module 3: S4 through Mamba-2
├── linear_recurrent/      # Module 4: RWKV, xLSTM, Griffin, RetNet
├── hybrid_architectures/  # Module 5: Hybrid Attention+SSM Designs
└── evaluation/            # Module 6: Benchmarking Harness (to be created)
```

## Running Tests
```bash
cd /Users/zhinan/universal_multi_agent_framework/coderpp_from_research_output
python3 -m pytest modules/ -v
```
