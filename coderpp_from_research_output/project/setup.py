"""Setup script for the Attention Mechanisms Benchmarking Framework.

Provides a collection of pure PyTorch implementations of modern attention
mechanisms, state space models, linear recurrent architectures, hybrid
architectures, and a comprehensive evaluation harness.
"""

from setuptools import setup, find_packages

setup(
    name="attn-bench",
    version="1.0.0",
    description="Attention Mechanisms Benchmarking Framework — FlashAttention, RoPE, SSMs, Linear RNNs, Hybrids, Evaluation",
    author="UMAF Research Pipeline",
    python_requires=">=3.11",
    packages=[
        "flash_attention",
        "rope_extrapolation",
        "state_space_models",
        "linear_recurrent",
        "hybrid_architectures",
        "evaluation",
    ],
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "einops>=0.7.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
