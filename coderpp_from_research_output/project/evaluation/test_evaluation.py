"""
Unit tests for the evaluation module.

Covers:
  - Import validation (all public symbols available)
  - Throughput Benchmark: GPU_SPECS, ThroughputBenchmark, ThroughputResult
  - Memory Profiler: MemoryProfiler, RooflineModel, RooflinePoint
  - Perplexity: PerplexityBenchmark, PerplexityDegradationCurve
  - Length Extrapolation: LengthExtrapolationEval, PasskeyRetrievalTest
  - Entropy: AttentionEntropyAnalyzer, LayerEntropyProfile
  - Comparison: ComparisonTable, ComparisonRow, to_latex, to_csv, to_markdown
  - Roofline: RooflinePlot, RooflineMeasurement, GPU_ROOFLINE_SPECS

Run with:
    PYTHONPATH=modules python -m pytest modules/evaluation/ -v
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest
import torch


# ===================================================================
# Fixtures
# ===================================================================

def _dummy_attention_fn(q, k, v, **kwargs):
    """Simple scaled dot-product attention for benchmarking tests."""
    d_head = q.shape[-1]
    scale = 1.0 / math.sqrt(d_head)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, v)


@pytest.fixture
def dummy_attn():
    return _dummy_attention_fn


# ===================================================================
# Tests: Import validation
# ===================================================================


class TestImports:
    """Verify all public symbols are importable."""

    def test_top_level_imports(self):
        import evaluation
        assert hasattr(evaluation, "ThroughputBenchmark")
        assert hasattr(evaluation, "ThroughputResult")
        assert hasattr(evaluation, "GPU_SPECS")
        assert hasattr(evaluation, "MemoryProfiler")
        assert hasattr(evaluation, "MemoryProfileResult")
        assert hasattr(evaluation, "RooflineModel")
        assert hasattr(evaluation, "RooflinePoint")
        assert hasattr(evaluation, "ROOFLINE_GPU_SPECS")
        assert hasattr(evaluation, "PerplexityBenchmark")
        assert hasattr(evaluation, "PerplexityResult")
        assert hasattr(evaluation, "PerplexityDegradationCurve")
        assert hasattr(evaluation, "LengthExtrapolationEval")
        assert hasattr(evaluation, "ExtrapolationResult")
        assert hasattr(evaluation, "PasskeyRetrievalTest")
        assert hasattr(evaluation, "AttentionEntropyAnalyzer")
        assert hasattr(evaluation, "LayerEntropyProfile")
        assert hasattr(evaluation, "ComparisonTable")
        assert hasattr(evaluation, "ComparisonRow")
        assert hasattr(evaluation, "RooflinePlot")
        assert hasattr(evaluation, "RooflineMeasurement")

    def test___all___complete(self):
        import evaluation
        expected_symbols = [
            "ThroughputBenchmark", "ThroughputResult", "GPU_SPECS",
            "MemoryProfiler", "MemoryProfileResult", "RooflineModel",
            "RooflinePoint", "ROOFLINE_GPU_SPECS",
            "PerplexityBenchmark", "PerplexityResult",
            "PerplexityDegradationCurve", "WIKITEXT2_FILENAME", "PG19_FILENAME",
            "LengthExtrapolationEval", "ExtrapolationResult",
            "PasskeyRetrievalTest", "RulerBenchmark", "InfiniteBenchBenchmark",
            "PASSKEY_DEFAULT_LENGTHS",
            "AttentionEntropyAnalyzer", "LayerEntropyProfile",
            "HeadEntropyDistribution", "EntropyDiagnosis",
            "ComparisonTable", "ComparisonRow", "COMPARISON_DIMENSIONS",
            "RooflinePlot", "RooflineMeasurement", "GPU_ROOFLINE_SPECS",
        ]
        for sym in expected_symbols:
            assert hasattr(evaluation, sym), f"Missing from evaluation: {sym}"


# ===================================================================
# Tests: throughput.py
# ===================================================================


class TestGPU_SPECS:
    """Tests for GPU_SPECS constant."""

    def test_contains_h100(self):
        from evaluation.throughput import GPU_SPECS
        assert "H100" in GPU_SPECS
        assert GPU_SPECS["H100"]["peak_tflops_fp16"] == 989.0

    def test_contains_a100(self):
        from evaluation.throughput import GPU_SPECS
        assert "A100" in GPU_SPECS
        assert GPU_SPECS["A100"]["hbm_capacity_gb"] == 80.0

    def test_all_gpus_have_required_keys(self):
        from evaluation.throughput import GPU_SPECS
        required = {"peak_tflops_fp16", "peak_tflops_bf16", "peak_tflops_fp32",
                     "memory_bandwidth_gb_s", "hbm_capacity_gb"}
        for gpu, specs in GPU_SPECS.items():
            missing = required - set(specs.keys())
            assert not missing, f"{gpu} missing keys: {missing}"


class TestThroughputBenchmark:
    """Tests for ThroughputBenchmark."""

    def test_init_default(self):
        from evaluation.throughput import ThroughputBenchmark
        bench = ThroughputBenchmark()
        assert bench.gpu_model == "H100"
        assert bench.warmup_iters == 5
        assert bench.bench_iters == 20

    def test_init_invalid_gpu(self):
        from evaluation.throughput import ThroughputBenchmark
        with pytest.raises(ValueError, match="Unknown GPU"):
            ThroughputBenchmark(gpu_model="FakeGPU")

    def test_init_invalid_precision(self):
        from evaluation.throughput import ThroughputBenchmark
        with pytest.raises(ValueError):
            ThroughputBenchmark(precision="int8")

    def test_get_peak_tflops(self):
        from evaluation.throughput import ThroughputBenchmark
        bench = ThroughputBenchmark(gpu_model="A100", precision="fp32")
        assert bench.get_peak_tflops() == 19.5

    def test_get_memory_bandwidth(self):
        from evaluation.throughput import ThroughputBenchmark
        bench = ThroughputBenchmark(gpu_model="H100")
        assert bench.get_memory_bandwidth() == 3350.0

    def test_benchmark_runs(self, dummy_attn):
        from evaluation.throughput import ThroughputBenchmark
        bench = ThroughputBenchmark(warmup_iters=2, bench_iters=3)
        result = bench.benchmark(dummy_attn, seq_len=64, d_model=256, batch_size=2)
        assert result.gpu_model == "H100"
        assert result.seq_len == 64
        assert result.tokens_per_second > 0
        assert result.compute_utilization_pct >= 0
        assert result.elapsed_seconds > 0

    def test_throughput_result_repr(self):
        from evaluation.throughput import ThroughputResult
        r = ThroughputResult(
            gpu_model="H100", seq_len=2048, d_model=1024, batch_size=8,
            tokens_per_second=1000.0, achieved_tflops=50.0, peak_tflops=989.0,
            compute_utilization_pct=5.0, elapsed_seconds=0.5, total_flops=1e12,
            memory_bytes=1024 * 1024 * 1024, precision="fp16",
        )
        s = repr(r)
        assert "H100" in s
        assert "2048" in s


class TestThroughputResult:
    """Tests for ThroughputResult dataclass."""

    def test_construction(self):
        from evaluation.throughput import ThroughputResult
        r = ThroughputResult(
            gpu_model="H100", seq_len=128, d_model=512, batch_size=4,
            tokens_per_second=500.0, achieved_tflops=10.0, peak_tflops=989.0,
            compute_utilization_pct=1.01, elapsed_seconds=0.1, total_flops=1e10,
            memory_bytes=2**30,
        )
        assert r.tokens_per_second == 500.0
        assert r.precision == "fp16"  # default


# ===================================================================
# Tests: memory.py
# ===================================================================


class TestRooflineModel:
    """Tests for RooflineModel."""

    def test_init_default(self):
        from evaluation.memory import RooflineModel
        model = RooflineModel(gpu_model="H100")
        assert model.peak_tflops == 989.0
        assert model.bandwidth_gb_s == 3350.0

    def test_ridge_point(self):
        from evaluation.memory import RooflineModel
        model = RooflineModel(gpu_model="A100")
        # ridge = peak / (bw / 1000) = 312 / (2039/1000) ≈ 153.0
        expected = 312.0 / (2039.0 / 1000.0)
        assert math.isclose(model.ridge_point, expected, rel_tol=1e-9)

    def test_evaluate_bandwidth_bound(self):
        from evaluation.memory import RooflineModel
        model = RooflineModel(gpu_model="H100")
        # Low arithmetic intensity → bandwidth bound
        point = model.evaluate(1.0)
        assert point.bandwidth_bound
        assert not point.compute_bound
        # achievable = 1.0 * 3.35 = 3.35 TFLOPS
        assert point.achievable_tflops == pytest.approx(3.35, rel=0.01)

    def test_evaluate_compute_bound(self):
        from evaluation.memory import RooflineModel
        model = RooflineModel(gpu_model="H100")
        # High arithmetic intensity → compute bound
        point = model.evaluate(1000.0)
        assert not point.bandwidth_bound
        assert point.compute_bound
        assert point.achievable_tflops == model.peak_tflops

    def test_invalid_gpu(self):
        from evaluation.memory import RooflineModel
        with pytest.raises(ValueError, match="Unknown GPU"):
            RooflineModel(gpu_model="NonexistentGPU")


class TestMemoryProfiler:
    """Tests for MemoryProfiler."""

    def test_init_default(self):
        from evaluation.memory import MemoryProfiler
        profiler = MemoryProfiler()
        assert profiler.hbm_capacity_gb == 80.0
        assert profiler.bandwidth_gb_s == 3350.0

    def test_profile_runs(self, dummy_attn):
        from evaluation.memory import MemoryProfiler
        profiler = MemoryProfiler()
        result = profiler.profile(dummy_attn, seq_len=64, d_model=256, batch_size=2)
        assert result.gpu_model == "H100"
        assert result.peak_memory_gb > 0
        assert result.memory_utilization_pct >= 0
        assert result.seq_len == 64

    def test_profile_tensor_shapes(self):
        from evaluation.memory import MemoryProfiler
        profiler = MemoryProfiler()
        mem = profiler.profile_tensor_shapes((2, 8, 2048, 128), dtype_size=2)
        expected = 2 * 8 * 2048 * 128 * 2
        assert mem == expected

    def test_get_roofline_model(self):
        from evaluation.memory import MemoryProfiler, RooflineModel
        profiler = MemoryProfiler(gpu_model="A100")
        rm = profiler.get_roofline_model()
        assert isinstance(rm, RooflineModel)
        assert rm.gpu_model == "A100"

    def test_memory_profile_result_repr(self):
        from evaluation.memory import MemoryProfileResult, RooflinePoint
        point = RooflinePoint(
            arithmetic_intensity=10.0, achievable_tflops=250.0,
            bandwidth_bound=True, compute_bound=False,
        )
        result = MemoryProfileResult(
            gpu_model="H100", hbm_capacity_gb=80.0, bandwidth_gb_s=3350.0,
            peak_memory_bytes=2**30, peak_memory_gb=1.0,
            memory_utilization_pct=1.25, bandwidth_utilization_pct=25.0,
            roofline_point=point, seq_len=2048, d_model=1024, batch_size=8,
            flops=1e12, bytes_read=2**30, bytes_written=2**29,
        )
        s = repr(result)
        assert "H100" in s


class TestROOFLINE_GPU_SPECS:
    """Tests for ROOFLINE_GPU_SPECS."""

    def test_contains_gpus(self):
        from evaluation.memory import ROOFLINE_GPU_SPECS
        assert "H100" in ROOFLINE_GPU_SPECS
        assert "A100" in ROOFLINE_GPU_SPECS


# ===================================================================
# Tests: perplexity.py
# ===================================================================


class TestPerplexityBenchmark:
    """Tests for PerplexityBenchmark."""

    def test_init_default(self):
        from evaluation.perplexity import PerplexityBenchmark
        ppl = PerplexityBenchmark()
        assert ppl.wikitext_path is None
        assert ppl.vocab_size == 50257

    def test_evaluate_runs(self, dummy_attn):
        from evaluation.perplexity import PerplexityBenchmark
        ppl = PerplexityBenchmark()
        result = ppl.evaluate(dummy_attn, max_seq_len=128, d_model=256, num_sequences=3)
        assert result.dataset == "synthetic"
        assert result.perplexity > 0
        assert result.tokens_evaluated > 0

    def test_evaluate_degradation(self, dummy_attn):
        from evaluation.perplexity import PerplexityBenchmark
        ppl = PerplexityBenchmark()
        curve = ppl.evaluate_degradation(dummy_attn, lengths=[64, 128], d_model=256)
        assert len(curve.lengths) == 2
        assert len(curve.perplexities) == 2

    def test_load_wikitext_none_path(self):
        from evaluation.perplexity import PerplexityBenchmark
        ppl = PerplexityBenchmark()
        assert ppl.load_wikitext() is None

    def test_load_pg19_none_path(self):
        from evaluation.perplexity import PerplexityBenchmark
        ppl = PerplexityBenchmark()
        assert ppl.load_pg19() is None

    def test_constants(self):
        from evaluation.perplexity import WIKITEXT2_FILENAME, PG19_FILENAME
        assert "wikitext" in WIKITEXT2_FILENAME.lower()
        assert "pg19" in PG19_FILENAME.lower()


class TestPerplexityDegradationCurve:
    """Tests for PerplexityDegradationCurve."""

    def test_add_point(self):
        from evaluation.perplexity import PerplexityDegradationCurve
        curve = PerplexityDegradationCurve(label="test")
        curve.add_point(512, 10.0)
        curve.add_point(1024, 12.0)
        assert len(curve.lengths) == 2

    def test_get_degradation_slope(self):
        from evaluation.perplexity import PerplexityDegradationCurve
        curve = PerplexityDegradationCurve()
        curve.add_point(512, 10.0)
        curve.add_point(1024, 12.0)
        slope = curve.get_degradation_slope()
        assert slope is not None
        assert slope == pytest.approx(2.0 / 512, rel=0.01)

    def test_get_degradation_slope_insufficient_data(self):
        from evaluation.perplexity import PerplexityDegradationCurve
        curve = PerplexityDegradationCurve()
        assert curve.get_degradation_slope() is None
        curve.add_point(512, 10.0)
        assert curve.get_degradation_slope() is None

    def test_to_dict(self):
        from evaluation.perplexity import PerplexityDegradationCurve
        curve = PerplexityDegradationCurve()
        curve.add_point(512, 10.0)
        curve.add_point(1024, 12.0)
        d = curve.to_dict()
        assert "lengths" in d
        assert "perplexities" in d
        assert len(d["lengths"]) == 2


# ===================================================================
# Tests: length_extrapolation.py
# ===================================================================


class TestPasskeyRetrievalTest:
    """Tests for PasskeyRetrievalTest."""

    def test_evaluate_runs(self, dummy_attn):
        from evaluation.length_extrapolation import PasskeyRetrievalTest
        pk = PasskeyRetrievalTest(num_trials=5)
        acc = pk.evaluate(dummy_attn, seq_len=64, d_model=256)
        assert 0.0 <= acc <= 1.0

    def test_perfect_retrieval_theoretical(self):
        """With a known passkey, verify we can detect it."""
        from evaluation.length_extrapolation import PasskeyRetrievalTest
        pk = PasskeyRetrievalTest(num_trials=1)

        def perfect_attn(q, k, v, **kwargs):
            return v  # Identity: output = v, so passkey is exactly at position

        acc = pk.evaluate(perfect_attn, seq_len=32, d_model=256)
        assert acc == 1.0


class TestLengthExtrapolationEval:
    """Tests for LengthExtrapolationEval."""

    def test_evaluate_runs(self, dummy_attn):
        from evaluation.length_extrapolation import LengthExtrapolationEval
        ev = LengthExtrapolationEval(model_name="test-attn")
        result = ev.evaluate(dummy_attn, lengths=[64, 128], d_model=256)
        assert result.model_name == "test-attn"
        assert len(result.lengths) == 2
        assert len(result.passkey_accuracies) == 2
        assert result.degradation_slope is not None

    def test_summary(self, dummy_attn):
        from evaluation.length_extrapolation import LengthExtrapolationEval
        ev = LengthExtrapolationEval(model_name="test")
        result = ev.evaluate(dummy_attn, lengths=[64, 128], d_model=256)
        summary = result.summary()
        assert "test" in summary


class TestRulerBenchmark:
    """Tests for RulerBenchmark."""

    def test_evaluate_runs(self, dummy_attn):
        from evaluation.length_extrapolation import RulerBenchmark
        ruler = RulerBenchmark(max_seq_len=128)
        scores = ruler.evaluate(dummy_attn)
        assert isinstance(scores, dict)
        assert len(scores) == 4
        for task in ["needle_in_haystack", "variable_tracking", "common_words", "multi_hop_qa"]:
            assert task in scores

    def test_composite_score(self):
        from evaluation.length_extrapolation import RulerBenchmark
        ruler = RulerBenchmark()
        assert ruler.composite_score({}) == 0.0
        assert ruler.composite_score({"a": 0.5, "b": 0.7}) == 0.6


class TestInfiniteBenchBenchmark:
    """Tests for InfiniteBenchBenchmark."""

    def test_evaluate_runs(self, dummy_attn):
        from evaluation.length_extrapolation import InfiniteBenchBenchmark
        ib = InfiniteBenchBenchmark(max_seq_len=128)
        scores = ib.evaluate(dummy_attn)
        assert isinstance(scores, dict)
        assert len(scores) == 4

    def test_composite_score(self):
        from evaluation.length_extrapolation import InfiniteBenchBenchmark
        ib = InfiniteBenchBenchmark()
        assert ib.composite_score({}) == 0.0
        assert ib.composite_score({"a": 100.0, "b": 50.0}) == 75.0


class TestPASSKEY_DEFAULT_LENGTHS:
    """Tests for PASSKEY_DEFAULT_LENGTHS."""

    def test_is_list_of_int(self):
        from evaluation.length_extrapolation import PASSKEY_DEFAULT_LENGTHS
        assert isinstance(PASSKEY_DEFAULT_LENGTHS, list)
        assert all(isinstance(x, int) for x in PASSKEY_DEFAULT_LENGTHS)
        assert PASSKEY_DEFAULT_LENGTHS == sorted(PASSKEY_DEFAULT_LENGTHS)


# ===================================================================
# Tests: entropy.py
# ===================================================================


class TestAttentionEntropyAnalyzer:
    """Tests for AttentionEntropyAnalyzer."""

    def test_init_default(self):
        from evaluation.entropy import AttentionEntropyAnalyzer
        analyzer = AttentionEntropyAnalyzer()
        assert analyzer.dilution_threshold == 0.85
        assert analyzer.collapse_threshold == 0.15

    def test_init_invalid_thresholds(self):
        from evaluation.entropy import AttentionEntropyAnalyzer
        with pytest.raises(ValueError):
            AttentionEntropyAnalyzer(dilution_threshold=0.1, collapse_threshold=0.5)

    def test_compute_entropy_uniform(self):
        from evaluation.entropy import AttentionEntropyAnalyzer
        analyzer = AttentionEntropyAnalyzer()
        # Uniform distribution over 4 tokens → H = log(4) ≈ 1.386
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        ent = analyzer.compute_entropy(probs)
        expected = math.log(4)
        assert torch.allclose(ent, torch.tensor(expected), atol=1e-5)

    def test_compute_entropy_deterministic(self):
        from evaluation.entropy import AttentionEntropyAnalyzer
        analyzer = AttentionEntropyAnalyzer()
        # One-hot → H = 0 (no uncertainty)
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        ent = analyzer.compute_entropy(probs)
        assert torch.allclose(ent, torch.tensor(0.0), atol=1e-5)

    def test_analyze_runs(self):
        from evaluation.entropy import AttentionEntropyAnalyzer
        analyzer = AttentionEntropyAnalyzer()
        # Create attention matrices: 2 layers, batch=1, heads=4, seq=16
        attn1 = torch.softmax(torch.randn(1, 4, 16, 16), dim=-1)
        attn2 = torch.softmax(torch.randn(1, 4, 16, 16), dim=-1)
        diagnosis = analyzer.analyze([attn1, attn2])
        assert diagnosis.num_layers == 2
        assert diagnosis.total_heads == 8
        assert diagnosis.health_verdict in ("HEALTHY", "WARNING", "UNHEALTHY")

    def test_analyze_tensor_input(self):
        from evaluation.entropy import AttentionEntropyAnalyzer
        analyzer = AttentionEntropyAnalyzer()
        # [n_layers=2, batch=1, heads=4, seq_q=8, seq_k=8]
        attn = torch.softmax(torch.randn(2, 1, 4, 8, 8), dim=-1)
        diagnosis = analyzer.analyze(attn)
        assert diagnosis.num_layers == 2

    def test_analyze_head_runs(self):
        from evaluation.entropy import AttentionEntropyAnalyzer
        analyzer = AttentionEntropyAnalyzer()
        attn = torch.softmax(torch.randn(1, 4, 16, 16), dim=-1)
        result = analyzer.analyze_head(attn, head_index=0)
        assert result.head_index == 0
        assert result.max_possible_entropy > 0

    def test_analyze_small_sequence(self):
        from evaluation.entropy import AttentionEntropyAnalyzer
        analyzer = AttentionEntropyAnalyzer(min_head_size=4)
        # Sequence shorter than min_head_size
        attn = torch.softmax(torch.randn(1, 2, 2, 2), dim=-1)
        diagnosis = analyzer.analyze([attn])
        assert diagnosis.num_layers == 1


class TestLayerEntropyProfile:
    """Tests for LayerEntropyProfile."""

    def test_health_ratio(self):
        from evaluation.entropy import LayerEntropyProfile
        profile = LayerEntropyProfile(
            layer_index=0, num_heads=4,
            healthy_heads=3, heads_collapsed=1, heads_diluted=0,
        )
        assert profile.health_ratio == 0.75

    def test_health_ratio_zero_heads(self):
        from evaluation.entropy import LayerEntropyProfile
        profile = LayerEntropyProfile(layer_index=0, num_heads=0)
        assert profile.health_ratio == 0.0


class TestEntropyDiagnosis:
    """Tests for EntropyDiagnosis."""

    def test_repr(self):
        from evaluation.entropy import EntropyDiagnosis
        dx = EntropyDiagnosis(
            num_layers=2, total_heads=8, heads_collapsed=1,
            heads_diluted=1, healthy_heads=6,
            global_mean_entropy=0.5, health_verdict="HEALTHY",
        )
        s = repr(dx)
        assert "HEALTHY" in s
        assert "layers=2" in s


# ===================================================================
# Tests: comparison.py
# ===================================================================


class TestComparisonTable:
    """Tests for ComparisonTable."""

    def test_add_row(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable()
        row = table.add_row("FlashAttention v3", complexity="O(N)", memory="O(N)")
        assert row.name == "FlashAttention v3"
        assert len(table) == 1

    def test_add_row_obj(self):
        from evaluation.comparison import ComparisonTable, ComparisonRow
        table = ComparisonTable()
        row = ComparisonRow(name="Test", complexity="O(1)")
        table.add_row_obj(row)
        assert len(table) == 1

    def test_get_row_found(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable()
        table.add_row("FlashAttention v1")
        row = table.get_row("FlashAttention v1")
        assert row is not None
        assert row.name == "FlashAttention v1"

    def test_get_row_not_found(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable()
        assert table.get_row("nonexistent") is None

    def test_to_string(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable()
        table.add_row("FA v3", complexity="O(N)")
        s = table.to_string()
        assert "FA v3" in s

    def test_to_string_empty(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable()
        assert "(empty table)" in table.to_string()

    def test_to_latex(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable()
        table.add_row("FlashAttention v3", complexity="O(N)", memory="O(N)")
        latex = table.to_latex()
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "FlashAttention v3" in latex

    def test_to_latex_empty(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable()
        latex = table.to_latex()
        assert "empty table" in latex

    def test_to_csv(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable(dimensions=["name", "complexity", "memory"])
        table.add_row("FA", complexity="O(N)", memory="O(N)")
        csv = table.to_csv()
        assert "name,complexity,memory" in csv
        assert "FA,O(N),O(N)" in csv

    def test_to_markdown(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable()
        table.add_row("FA v3", complexity="O(N)", memory="O(N)")
        md = table.to_markdown()
        assert "| name |" in md
        assert "FA v3" in md

    def test_to_dict(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable()
        table.add_row("FA v3", complexity="O(N)")
        data = table.to_dict()
        assert len(data) == 1
        assert data[0]["name"] == "FA v3"

    def test_init_custom_dimensions(self):
        from evaluation.comparison import ComparisonTable
        table = ComparisonTable(dimensions=["name", "complexity"])
        assert table.dimensions == ["name", "complexity"]

    def test_comparison_dimensions_constant(self):
        from evaluation.comparison import COMPARISON_DIMENSIONS
        assert "complexity" in COMPARISON_DIMENSIONS
        assert "memory" in COMPARISON_DIMENSIONS
        assert isinstance(COMPARISON_DIMENSIONS, list)


class TestComparisonRow:
    """Tests for ComparisonRow."""

    def test_to_dict(self):
        from evaluation.comparison import ComparisonRow
        row = ComparisonRow(name="FA v3", complexity="O(N)")
        d = row.to_dict()
        assert d["name"] == "FA v3"
        assert d["complexity"] == "O(N)"

    def test_defaults(self):
        from evaluation.comparison import ComparisonRow
        row = ComparisonRow(name="test")
        assert row.complexity == "O(N²)"
        assert row.memory == "O(N²)"
        assert row.causal == "Yes"

    def test_repr(self):
        from evaluation.comparison import ComparisonRow
        row = ComparisonRow(name="test", algorithm="v1")
        s = repr(row)
        assert "test" in s
        assert "v1" in s


# ===================================================================
# Tests: roofline.py
# ===================================================================


class TestRooflinePlot:
    """Tests for RooflinePlot."""

    def test_init_default(self):
        from evaluation.roofline import RooflinePlot
        plotter = RooflinePlot()
        assert plotter.gpu_model == "H100"
        assert plotter.precision == "fp16"
        assert plotter.peak_tflops > 0
        assert plotter.bandwidth_gb_s > 0

    def test_init_invalid_gpu(self):
        from evaluation.roofline import RooflinePlot
        with pytest.raises(ValueError, match="Unknown GPU"):
            RooflinePlot(gpu_model="FakeGPU")

    def test_init_invalid_precision(self):
        from evaluation.roofline import RooflinePlot
        with pytest.raises(ValueError):
            RooflinePlot(precision="int4")

    def test_add_measurement(self):
        from evaluation.roofline import RooflinePlot
        plotter = RooflinePlot()
        m = plotter.add_measurement("FA v3", arithmetic_intensity=10.0, flops_per_sec=250.0)
        assert m.label == "FA v3"
        assert len(plotter.measurements) == 1

    def test_add_measurement_from_attention(self):
        from evaluation.roofline import RooflinePlot
        plotter = RooflinePlot()
        m = plotter.add_measurement_from_attention(
            "Test", seq_len=2048, d_model=1024, elapsed_seconds=0.01,
        )
        assert m.label == "Test"
        assert m.arithmetic_intensity > 0

    def test_classify(self):
        from evaluation.roofline import RooflinePlot, RooflineMeasurement
        plotter = RooflinePlot(gpu_model="H100")
        # Low AI → bandwidth bound
        m_low = RooflineMeasurement("low", arithmetic_intensity=1.0, flops_per_sec=3.0)
        assert plotter.classify(m_low) == "bandwidth_bound"
        # High AI → compute bound
        m_high = RooflineMeasurement("high", arithmetic_intensity=1000.0, flops_per_sec=989.0)
        assert plotter.classify(m_high) == "compute_bound"

    def test_get_roofline_data(self):
        from evaluation.roofline import RooflinePlot
        plotter = RooflinePlot()
        data = plotter.get_roofline_data(num_points=10)
        assert len(data["arithmetic_intensity"]) == 10
        assert len(data["achievable_tflops"]) == 10
        assert data["gpu_model"] == "H100"

    def test_get_measurements_data(self):
        from evaluation.roofline import RooflinePlot
        plotter = RooflinePlot()
        plotter.add_measurement("FA v3", 10.0, 250.0)
        data = plotter.get_measurements_data()
        assert len(data) == 1
        assert "label" in data[0]

    def test_repr(self):
        from evaluation.roofline import RooflinePlot
        plotter = RooflinePlot(gpu_model="A100")
        s = repr(plotter)
        assert "A100" in s

    def test_save_plot(self, dummy_attn):
        """Save a real roofline plot to a temp file."""
        from evaluation.roofline import RooflinePlot
        plotter = RooflinePlot(gpu_model="H100")
        plotter.add_measurement("FA v3", 10.0, 250.0)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = plotter.save(path)
            assert os.path.exists(result)
        finally:
            os.unlink(path)


class TestRooflineMeasurement:
    """Tests for RooflineMeasurement."""

    def test_construction(self):
        from evaluation.roofline import RooflineMeasurement
        m = RooflineMeasurement("test", arithmetic_intensity=5.0, flops_per_sec=100.0)
        assert m.label == "test"
        assert m.achieved_tflops == 100.0
        assert m.color == "blue"
        assert m.marker == "o"

    def test_auto_cycle_colors(self):
        from evaluation.roofline import RooflinePlot
        plotter = RooflinePlot()
        m1 = plotter.add_measurement("a", 1.0, 100.0)
        m2 = plotter.add_measurement("b", 2.0, 200.0)
        assert m1.color != m2.color


class TestGPU_ROOFLINE_SPECS:
    """Tests for GPU_ROOFLINE_SPECS."""

    def test_contains_h100(self):
        from evaluation.roofline import GPU_ROOFLINE_SPECS
        assert "H100" in GPU_ROOFLINE_SPECS
        assert "peak_tflops_tc" in GPU_ROOFLINE_SPECS["H100"]

    def test_all_gpus_have_required_keys(self):
        from evaluation.roofline import GPU_ROOFLINE_SPECS
        required = {"peak_tflops_fp16", "bandwidth_gb_s", "hbm_gb"}
        for gpu, specs in GPU_ROOFLINE_SPECS.items():
            missing = required - set(specs.keys())
            assert not missing, f"{gpu} missing: {missing}"


# ===================================================================
# Tests: Cross-module consistency
# ===================================================================


class TestCrossModuleConsistency:
    """Integration/consistency tests across submodules."""

    def test_gpu_specs_consistent(self):
        """GPU_SPECS and ROOFLINE_GPU_SPECS should agree on common GPUs."""
        from evaluation.throughput import GPU_SPECS as TGS
        from evaluation.memory import ROOFLINE_GPU_SPECS as RMGS
        from evaluation.roofline import GPU_ROOFLINE_SPECS as GRS
        common = set(TGS.keys()) & set(RMGS.keys()) & set(GRS.keys())
        assert len(common) >= 3  # At least H100, A100, L40S
        for gpu in common:
            assert TGS[gpu]["peak_tflops_fp16"] == RMGS[gpu]["peak_tflops_fp16"], (
                f"{gpu} fp16 mismatch: {TGS[gpu]['peak_tflops_fp16']} vs {RMGS[gpu]['peak_tflops_fp16']}"
            )

    def test_no_circular_imports(self):
        """Ensure evaluation can be fully imported without circular imports."""
        import evaluation
        assert evaluation.__all__ is not None

    def test_roofline_model_consistent(self):
        """RooflineModel from memory and RooflinePlot from roofline should agree."""
        from evaluation.memory import RooflineModel
        from evaluation.roofline import RooflinePlot
        rm = RooflineModel(gpu_model="H100")
        rp = RooflinePlot(gpu_model="H100")
        assert rm.peak_tflops == rp.peak_tflops
        assert rm.bandwidth_gb_s == rp.bandwidth_gb_s
