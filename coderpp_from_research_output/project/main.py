#!/usr/bin/env python3
"""Attention Mechanisms Benchmarking Framework — Main Entry Point.

Demonstrates and benchmarks all implemented attention mechanisms:

  - FlashAttention v1/v2/v3 (block-wise tiled attention)
  - RoPE extrapolation (PI, NTK, YaRN, DPE)
  - State Space Models (S4, S4D, Mamba/S6, Mamba-2/SSD)
  - Linear Recurrent Architectures (RWKV, xLSTM, Griffin, RetNet)
  - Hybrid Architectures (H3, GLA, Mega, configurable hybrid mixers)
  - Evaluation Harness (throughput, memory, perplexity, entropy, comparison)

Usage:
    cd project && python main.py                    # Run all demos
    cd project && python main.py --quick             # Quick smoke-test
    cd project && python main.py --module flash      # Single module demo
    cd project && python main.py --benchmark         # Run full benchmark
"""

from __future__ import annotations

import argparse
import math
import time
import sys

import torch


# ─────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    """Print a section banner."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ─────────────────────────────────────────────────────────────────────────
# Module demos
# ─────────────────────────────────────────────────────────────────────────

def demo_flash_attention() -> None:
    """Demonstrate FlashAttention v1/v2/v3."""
    banner("FlashAttention v1/v2/v3")
    from flash_attention import (
        FlashAttentionV1, FlashAttentionV2, FlashAttentionV3,
        compute_flops_saved, compute_memory_saved,
    )

    B, H, L, D = 2, 4, 128, 64
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    for version, cls in [("v1", FlashAttentionV1), ("v2", FlashAttentionV2), ("v3", FlashAttentionV3)]:
        attn = cls(Br=32, Bc=32, causal=True)
        with torch.no_grad():
            out = attn(q, k, v)
        ok = out.shape == (B, H, L, D)
        print(f"  FlashAttention {version}: shape={tuple(out.shape)} {'✓' if ok else '✗'}")

    flops = compute_flops_saved(L, D)
    mem = compute_memory_saved(L, D)
    print(f"  FLOPs: standard={flops.get('standard_gflops', 0):.1f} GFLOPs, "
          f"tiled={flops.get('tiled_gflops', 0):.1f} GFLOPs")
    print(f"  Memory: standard={mem.get('standard_mb', 0):.1f} MB, "
          f"tiled={mem.get('tiled_mb', 0):.1f} MB (O(N) vs O(N²))")


def demo_rope_extrapolation() -> None:
    """Demonstrate RoPE and length extrapolation methods."""
    banner("RoPE & Length Extrapolation")
    from rope_extrapolation import (
        RoPE, PIRoPE, NTKAwareRoPE, YaRNRoPE, DPERoPE,
        compare_angles,
    )

    dim, seq_len = 64, 128
    # RoPE expects 4D: (batch, seq_len, num_heads, head_dim)
    q = torch.randn(2, seq_len, 4, dim)
    k = torch.randn(2, seq_len, 4, dim)

    # Base RoPE — forward takes (q, k); YaRN returns (q, k, temperature); DPE returns (q, k, mask)
    for name, rope_cls, init_kwargs in [
        ("RoPE (base)", RoPE, {"dim": dim, "max_seq_len": seq_len}),
        ("PI (2x)", PIRoPE, {"dim": dim, "original_max_len": 64, "extended_max_len": 128}),
        ("NTK-aware", NTKAwareRoPE, {"dim": dim, "scale_factor": 2.0}),
        ("YaRN", YaRNRoPE, {"dim": dim, "scale_factor": 2.0}),
        ("DPE", DPERoPE, {"dim": dim, "chunk_size": 64, "num_global_tokens": 4}),
    ]:
        try:
            rope = rope_cls(**init_kwargs)
            result = rope(q, k)
            q_rot, k_rot = result[0], result[1]
            extra = f" (+{len(result)-2} extra)" if len(result) > 2 else ""
            ok = q_rot.shape == q.shape and k_rot.shape == k.shape
            print(f"  {name}: q={tuple(q_rot.shape)}, k={tuple(k_rot.shape)}{extra} {'✓' if ok else '✗'}")
        except Exception as e:
            print(f"  {name}: ✗ ({e})")

    # Compare angle distributions across methods
    print()
    print("  Angle comparison across methods:")
    try:
        comparison = compare_angles(dim=dim, max_pos=128, scale=2.0)
        for method, angles in comparison.items():
            if isinstance(angles, torch.Tensor):
                print(f"    {method}: shape={tuple(angles.shape)}, range=[{angles.min().item():.3f}, {angles.max().item():.3f}]")
    except Exception as e:
        print(f"    ✗ ({e})")


def demo_state_space_models() -> None:
    """Demonstrate State Space Models (S4 through Mamba-2)."""
    banner("State Space Models: S4 → Mamba-2")
    from state_space_models import (
        hippo_legs_matrix,
        associative_scan,
        S4Layer, S4DLayer,
        MambaBlock as MambaProdBlock,  # production Mamba
        Mamba2Block,
    )

    B, L, D = 2, 64, 32
    x = torch.randn(B, L, D)

    # HiPPO
    A = hippo_legs_matrix(16)
    print(f"  HiPPO-LegS (16×16): shape={tuple(A.shape)}, neg_eigvals={bool((A.diag() < 0).all())}")

    # Associative scan
    scan_a = torch.ones(B, L)
    scan_b = torch.randn(B, L)
    X, final = associative_scan(scan_a, scan_b)
    print(f"  Associative Scan: X={tuple(X.shape)}, final={tuple(final.shape)}")

    # S4 / S4D — use `N` for state dim, `d_model` for model dim
    for name, layer_cls in [("S4", S4Layer), ("S4D", S4DLayer)]:
        try:
            layer = layer_cls(d_model=D, N=16)
            out = layer(x)
            ok = out.shape == x.shape
            print(f"  {name}: shape={tuple(out.shape)} {'✓' if ok else '✗'}")
        except Exception as e:
            print(f"  {name}: ✗ ({e})")

    # Mamba (production)
    try:
        mamba = MambaProdBlock(d_model=D, d_state=16)
        out = mamba(x)
        print(f"  Mamba/S6 (prod): shape={tuple(out.shape)} {'✓' if out.shape == x.shape else '✗'}")
    except Exception as e:
        print(f"  Mamba/S6 (prod): ✗ ({e})")

    # Mamba-2 (uses Mamba2Config dataclass)
    try:
        from state_space_models import Mamba2Config
        mamba2 = Mamba2Block(Mamba2Config(d_model=D, d_state=64))
        out = mamba2(x)
        print(f"  Mamba-2/SSD: shape={tuple(out.shape)} {'✓' if out.shape == x.shape else '✗'}")
    except Exception as e:
        print(f"  Mamba-2/SSD: ✗ ({e})")


def demo_linear_recurrent() -> None:
    """Demonstrate Linear Recurrent Architectures."""
    banner("Linear Recurrent: RWKV, xLSTM, Griffin, RetNet")
    from linear_recurrent import (
        RWKVBlock, xLSTMBlock, GriffinBlock, RetNetBlock,
        RMSNorm,
    )

    B, L, D = 2, 64, 32
    x = torch.randn(B, L, D)

    # RMSNorm
    norm = RMSNorm(D)
    out = norm(x)
    print(f"  RMSNorm: shape={tuple(out.shape)} ✓")

    # Each block uses `dim` parameter (not `d_model`)
    for name, block_cls, kwargs in [
        ("RWKV", RWKVBlock, {"dim": D}),
        ("xLSTM", xLSTMBlock, {"d_model": D}),
        ("Griffin", GriffinBlock, {"dim": D}),
        ("RetNet", RetNetBlock, {"dim": D}),
    ]:
        try:
            block = block_cls(**kwargs)
            out = block(x)
            # xLSTM returns (output, state) tuple
            if isinstance(out, tuple):
                out = out[0]
            ok = out.shape == x.shape
            print(f"  {name}: shape={tuple(out.shape)} {'✓' if ok else '✗'}")
        except Exception as e:
            print(f"  {name}: ✗ ({e})")


def demo_hybrid_architectures() -> None:
    """Demonstrate Hybrid Architectures."""
    banner("Hybrid Architectures: H3, GLA, Mega, HybridMixer")
    from hybrid_architectures import (
        H3Layer, GatedLinearAttention, MegaLayer, HybridMixer,
        SlidingWindowAttention, DiagonalSSM,
        KernelFusionPath,
    )

    B, L, D = 2, 64, 32
    x = torch.randn(B, L, D)

    # Sliding window attention
    swa = SlidingWindowAttention(d_model=D, n_heads=4, window_size=16)
    out = swa(x)
    print(f"  SlidingWindowAttention: shape={tuple(out.shape)} {'✓' if out.shape == x.shape else '✗'}")

    # Diagonal SSM
    ssm = DiagonalSSM(d_state=16, d_model=D)
    out = ssm(x, mode="convolution")
    print(f"  DiagonalSSM: shape={tuple(out.shape)} {'✓' if out.shape == x.shape else '✗'}")

    # H3, GLA, Mega
    for name, init_fn in [
        ("H3", lambda: H3Layer(d_model=D, d_state=16)),
        ("GLA", lambda: GatedLinearAttention(d_model=D, n_heads=4)),
        ("Mega", lambda: MegaLayer(d_model=D)),
    ]:
        try:
            layer = init_fn()
            out = layer(x)
            if isinstance(out, tuple):
                out = out[0]
            ok = out.shape == x.shape
            print(f"  {name}: shape={tuple(out.shape)} {'✓' if ok else '✗'}")
        except Exception as e:
            print(f"  {name}: ✗ ({e})")

    # Hybrid mixer — all three fusion paths
    for path in [KernelFusionPath.SERIAL, KernelFusionPath.PARALLEL, KernelFusionPath.INTERLEAVED]:
        try:
            mixer = HybridMixer(d_model=D, n_heads=4, d_state=16, fusion_path=path)
            out = mixer(x, mode="hybrid")
            print(f"  HybridMixer ({path.value}): shape={tuple(out.shape)} {'✓' if out.shape == x.shape else '✗'}")
        except Exception as e:
            print(f"  HybridMixer ({path.value}): ✗ ({e})")


def demo_evaluation() -> None:
    """Demonstrate the Evaluation Harness."""
    banner("Evaluation Harness")
    from evaluation import (
        ThroughputBenchmark, MemoryProfiler, PerplexityBenchmark,
        LengthExtrapolationEval, AttentionEntropyAnalyzer,
        ComparisonTable, RooflinePlot,
    )

    def dummy_attn(q, k, v, **kwargs):
        d_head = q.shape[-1]
        return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head), dim=-1), v)

    # Throughput
    bench = ThroughputBenchmark(warmup_iters=2, bench_iters=3)
    result = bench.benchmark(dummy_attn, seq_len=64, d_model=256, batch_size=2)
    print(f"  Throughput: {result.tokens_per_second:.0f} tok/s ({result.compute_utilization_pct:.1f}% util)")

    # Memory
    profiler = MemoryProfiler()
    mem = profiler.profile(dummy_attn, seq_len=64, d_model=256, batch_size=2)
    print(f"  Memory: {mem.peak_memory_gb:.4f} GB peak ({mem.memory_utilization_pct:.1f}% util)")

    # Perplexity
    ppl = PerplexityBenchmark()
    ppl_result = ppl.evaluate(dummy_attn, max_seq_len=64, d_model=256, num_sequences=3)
    print(f"  Perplexity: {ppl_result.perplexity:.2f}")

    # Length extrapolation
    ext = LengthExtrapolationEval(model_name="dummy")
    ext_result = ext.evaluate(dummy_attn, lengths=[64, 128], d_model=256)
    print(f"  Length Extrapolation: pk_acc={[f'{a:.2f}' for a in ext_result.passkey_accuracies]}")

    # Entropy
    analyzer = AttentionEntropyAnalyzer()
    attn_mats = [torch.softmax(torch.randn(1, 4, 16, 16), dim=-1) for _ in range(2)]
    diagnosis = analyzer.analyze(attn_mats)
    print(f"  Entropy: {diagnosis.health_verdict} ({diagnosis.healthy_heads}/{diagnosis.total_heads} healthy heads)")

    # Comparison table
    table = ComparisonTable()
    table.add_row("FlashAttention v3", complexity="O(N)", memory="O(N)", fp8_support="Yes")
    table.add_row("Mamba-2", complexity="O(N log N)", memory="O(N)", fp8_support="No")
    table.add_row("RWKV-7", complexity="O(N)", memory="O(1)", fp8_support="No")
    print(f"  Comparison Table: {len(table)} rows")
    print(f"  LaTeX table: {len(table.to_latex())} chars, CSV: {len(table.to_csv())} chars")

    # Roofline plot
    plotter = RooflinePlot(gpu_model="H100")
    plotter.add_measurement("FA v3", 10.0, 250.0)
    plotter.add_measurement("Mamba-2", 50.0, 400.0)
    print(f"  Roofline Plot: {len(plotter.measurements)} measurements, ridge={plotter.ridge_point:.1f}")


# ─────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────

def run_benchmark() -> None:
    """Run a naive attention speed benchmark."""
    banner("COMPREHENSIVE BENCHMARK")

    def naive_attn(q, k, v, **kwargs):
        d_head = q.shape[-1]
        return torch.matmul(
            torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head), dim=-1), v)

    configs = [
        ("Small", 128, 256, 4),
        ("Medium", 512, 512, 4),
        ("Large", 1024, 768, 2),
    ]

    print(f"\n  {'Config':<10} {'Seq':>6} {'Dim':>6} {'Batch':>6} {'Time (ms)':>10} {'Tok/s':>12}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*12}")

    for name, seq, dim, batch in configs:
        B, H, L, D = batch, 8, seq, dim
        q = torch.randn(B, H, L, D // H)
        k = torch.randn(B, H, L, D // H)
        v = torch.randn(B, H, L, D // H)

        start = time.perf_counter()
        for _ in range(10):
            _ = naive_attn(q, k, v)
        elapsed_ms = (time.perf_counter() - start) / 10 * 1000
        tok_per_sec = (B * L) / (elapsed_ms / 1000)
        print(f"  {name:<10} {seq:>6} {dim:>6} {batch:>6} {elapsed_ms:>10.2f} {tok_per_sec:>12.0f}")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attention Mechanisms Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Run all demos
  python main.py --quick          Quick smoke test
  python main.py --module flash   Demo a single module
  python main.py --benchmark      Run comprehensive benchmark
  python main.py --test           Run all tests via pytest

Available modules: flash, rope, ssm, linear, hybrid, eval
        """,
    )
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--benchmark", action="store_true", help="Full benchmark")
    parser.add_argument("--module", type=str, default="all",
                        choices=["all", "flash", "rope", "ssm", "linear", "hybrid", "eval"],
                        help="Single module to demo")
    parser.add_argument("--test", action="store_true", help="Run all tests via pytest")
    args = parser.parse_args()

    print("=" * 70)
    print("  Attention Mechanisms Benchmarking Framework  v1.0.0")
    print("  FlashAttention · RoPE · SSMs · Linear RNNs · Hybrids · Evaluation")
    print("=" * 70)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {'Available' if torch.cuda.is_available() else 'Not available (CPU mode)'}")
    print(f"  Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if args.test:
        print("\nRunning all tests via pytest...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", "-v", "--tb=short"], check=False)
        return

    if args.benchmark:
        run_benchmark()
        return

    module_map = {
        "flash": demo_flash_attention,
        "rope": demo_rope_extrapolation,
        "ssm": demo_state_space_models,
        "linear": demo_linear_recurrent,
        "hybrid": demo_hybrid_architectures,
        "eval": demo_evaluation,
    }

    if args.module != "all":
        module_map[args.module]()
    else:
        for name, demo_fn in module_map.items():
            try:
                demo_fn()
            except Exception as e:
                import traceback
                print(f"\n  [{name}] ERROR: {e}")
                if not args.quick:
                    traceback.print_exc()

    print()
    print("=" * 70)
    print("  All demos complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
