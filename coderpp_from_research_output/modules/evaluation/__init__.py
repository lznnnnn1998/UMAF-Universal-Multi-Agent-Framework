"""
Evaluation Module — Comprehensive Benchmarking Harness.

Benchmarks and compares attention mechanism implementations across seven
dimensions:

1. **Throughput**: tokens/sec, achieved TFLOPS, compute utilization vs GPUs.
2. **Memory**: peak HBM usage, memory bandwidth utilization via roofline model.
3. **Perplexity**: WikiText-2, PG-19, degradation curves beyond training length.
4. **Length Extrapolation**: perplexity-vs-length, passkey retrieval, RULER, InfiniteBench.
5. **Attention Entropy**: per-layer/per-head distributions, attention dilution diagnosis.
6. **Comparison Tables**: tabular data across architectures with standard dimensions.
7. **Roofline Plots**: arithmetic intensity vs FLOP/s, publication-quality figures.

Quickstart::

    from evaluation import ThroughputBenchmark, MemoryProfiler, PerplexityBenchmark
    from evaluation import LengthExtrapolationEval, AttentionEntropyAnalyzer
    from evaluation import ComparisonTable, RooflinePlot

    # Throughput benchmark
    bench = ThroughputBenchmark(gpu_model="H100")
    result = bench.benchmark(attention_fn, seq_len=2048, d_model=1024, batch_size=8)

    # Memory analysis
    profiler = MemoryProfiler(hbm_capacity_gb=80, bandwidth_gb_s=3350)
    mem_result = profiler.profile(attention_fn, seq_len=2048, d_model=1024)

    # Perplexity
    ppl = PerplexityBenchmark(wikitext_path="/path/to/wikitext.pt")
    ppl_result = ppl.evaluate(attention_fn, max_seq_len=2048)

    # Length extrapolation
    ext = LengthExtrapolationEval()
    ext_result = ext.evaluate(attention_fn, lengths=[512, 1024, 2048, 4096, 8192])

    # Entropy
    entropy_analyzer = AttentionEntropyAnalyzer()
    entropy_results = entropy_analyzer.analyze(attention_matrices)

    # Comparison table
    table = ComparisonTable()
    table.add_row("FlashAttention v3", complexity="O(N)", memory="O(N)")
    latex = table.to_latex()

    # Roofline plot
    plotter = RooflinePlot(gpu_model="A100")
    plotter.add_measurement("FlashAttention v3", arithmetic_intensity=10.5, flops_per_sec=250)
    plotter.save("roofline.pdf")
"""

from .throughput import (
    ThroughputBenchmark,
    ThroughputResult,
    GPU_SPECS,
)
from .memory import (
    MemoryProfiler,
    MemoryProfileResult,
    RooflineModel,
    RooflinePoint,
    ROOFLINE_GPU_SPECS,
)
from .perplexity import (
    PerplexityBenchmark,
    PerplexityResult,
    PerplexityDegradationCurve,
    WIKITEXT2_FILENAME,
    PG19_FILENAME,
)
from .length_extrapolation import (
    LengthExtrapolationEval,
    ExtrapolationResult,
    PasskeyRetrievalTest,
    RulerBenchmark,
    InfiniteBenchBenchmark,
    PASSKEY_DEFAULT_LENGTHS,
)
from .entropy import (
    AttentionEntropyAnalyzer,
    LayerEntropyProfile,
    HeadEntropyDistribution,
    EntropyDiagnosis,
)
from .comparison import (
    ComparisonTable,
    ComparisonRow,
    COMPARISON_DIMENSIONS,
)
from .roofline import (
    RooflinePlot,
    RooflineMeasurement,
    GPU_ROOFLINE_SPECS,
)

__all__ = [
    # Throughput
    "ThroughputBenchmark",
    "ThroughputResult",
    "GPU_SPECS",
    # Memory
    "MemoryProfiler",
    "MemoryProfileResult",
    "RooflineModel",
    "RooflinePoint",
    "ROOFLINE_GPU_SPECS",
    # Perplexity
    "PerplexityBenchmark",
    "PerplexityResult",
    "PerplexityDegradationCurve",
    "WIKITEXT2_FILENAME",
    "PG19_FILENAME",
    # Length Extrapolation
    "LengthExtrapolationEval",
    "ExtrapolationResult",
    "PasskeyRetrievalTest",
    "RulerBenchmark",
    "InfiniteBenchBenchmark",
    "PASSKEY_DEFAULT_LENGTHS",
    # Entropy
    "AttentionEntropyAnalyzer",
    "LayerEntropyProfile",
    "HeadEntropyDistribution",
    "EntropyDiagnosis",
    # Comparison
    "ComparisonTable",
    "ComparisonRow",
    "COMPARISON_DIMENSIONS",
    # Roofline
    "RooflinePlot",
    "RooflineMeasurement",
    "GPU_ROOFLINE_SPECS",
]
