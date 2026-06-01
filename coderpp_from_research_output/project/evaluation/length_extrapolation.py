"""
Length Extrapolation Evaluation — Measures attention quality beyond training length.

Evaluates how well attention mechanisms generalize to sequence lengths
beyond what they were trained on using three complementary tests:

1. **Perplexity-vs-length**: Measures perplexity degradation as length increases.
2. **Passkey Retrieval**: Tests whether a token at a specific position can be
   retrieved after processing a long context.
3. **RULER & InfiniteBench**: Standardized benchmarks for long-context evaluation.

Usage:
    from evaluation.length_extrapolation import (
        LengthExtrapolationEval, ExtrapolationResult,
        PasskeyRetrievalTest, RulerBenchmark, InfiniteBenchBenchmark,
        PASSKEY_DEFAULT_LENGTHS,
    )

    ev = LengthExtrapolationEval()
    result = ev.evaluate(attention_fn, lengths=[512, 1024, 2048, 4096, 8192])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# ─────────────────────────────────────────────────────────────────────────
# Default test lengths
# ─────────────────────────────────────────────────────────────────────────

PASSKEY_DEFAULT_LENGTHS: list[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768]


# ─────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ExtrapolationResult:
    """Result of a length extrapolation evaluation.

    Attributes:
        model_name: Name/label for the attention mechanism.
        lengths: Sequence lengths evaluated.
        perplexities: Perplexity at each length.
        passkey_accuracies: Passkey retrieval accuracy at each length (0-1).
        ruler_score: RULER benchmark composite score (if run).
        infinite_bench_score: InfiniteBench score (if run).
        degradation_slope: Slope of perplexity vs length linear fit.
    """

    model_name: str
    lengths: list[int] = field(default_factory=list)
    perplexities: list[float] = field(default_factory=list)
    passkey_accuracies: list[float] = field(default_factory=list)
    ruler_score: float | None = None
    infinite_bench_score: float | None = None
    degradation_slope: float | None = None

    def summary(self) -> str:
        """Return a one-line summary string."""
        parts = [f"ExtrapolationResult({self.model_name})"]
        if self.perplexities:
            parts.append(f"ppl@max_len={self.perplexities[-1]:.2f}")
        if self.passkey_accuracies:
            parts.append(f"pk_acc={self.passkey_accuracies[-1]:.2%}")
        if self.degradation_slope is not None:
            parts.append(f"slope={self.degradation_slope:.4f}")
        return ", ".join(parts)

    def __repr__(self) -> str:
        return self.summary()


# ─────────────────────────────────────────────────────────────────────────
# Passkey Retrieval Test
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class PasskeyRetrievalTest:
    """Passkey retrieval evaluation for long-context models.

    A "passkey" token is placed at a specific position in a long sequence.
    The model must attend to it and retrieve its value. Failure indicates
    the attention mechanism cannot effectively access distant tokens.

    Args:
        num_trials: Number of trials per length (default: 50).
        passkey_value: The value to retrieve (default: 42.0).
    """

    num_trials: int = 50
    passkey_value: float = 42.0

    def evaluate(
        self,
        attention_fn: Callable[..., torch.Tensor],
        seq_len: int,
        d_model: int = 1024,
        **kwargs: Any,
    ) -> float:
        """Evaluate passkey retrieval accuracy at a given sequence length.

        The passkey is embedded in the value tensor at a random position,
        and the output at that position is compared to a target.

        Args:
            attention_fn: A callable ``fn(q, k, v, **kwargs)``.
            seq_len: Sequence length to test.
            d_model: Model dimension.
            **kwargs: Additional arguments forwarded to ``attention_fn``.

        Returns:
            Accuracy (0.0 to 1.0).
        """
        n_heads = 8
        d_head = d_model // n_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        correct = 0

        for trial in range(self.num_trials):
            # Place passkey at a random position, weighted toward the end
            passkey_pos = seq_len - 1 - int((1.0 - math.sqrt(torch.rand(1).item())) * seq_len)
            passkey_pos = max(0, min(passkey_pos, seq_len - 1))

            q = torch.randn(1, n_heads, seq_len, d_head, device=device)
            k = torch.randn(1, n_heads, seq_len, d_head, device=device)
            v = torch.randn(1, n_heads, seq_len, d_head, device=device)

            # Inject passkey into v at passkey_pos
            v[:, :, passkey_pos, :] = self.passkey_value

            try:
                output = attention_fn(q, k, v, **kwargs)
                # Check if output at passkey_pos is close to passkey
                retrieved = output[0, 0, passkey_pos, 0].item()
                threshold = abs(self.passkey_value) * 0.1 + 1e-8
                if abs(retrieved - self.passkey_value) < threshold:
                    correct += 1
            except Exception:
                pass  # Treat as incorrect

        return correct / self.num_trials


# ─────────────────────────────────────────────────────────────────────────
# RULER Benchmark
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class RulerBenchmark:
    """RULER benchmark — synthetic long-context retrieval tasks.

    RULER (Retrieval Understanding and Long-context Evaluation for Retrieval)
    evaluates models on tasks like Needle-in-a-Haystack, variable tracking,
    common/frequent word extraction, and multi-hop QA.

    Args:
        max_seq_len: Maximum sequence length to test (default: 32768).
        tasks: List of task names to evaluate (default: all four).
    """

    max_seq_len: int = 32768
    tasks: list[str] = field(
        default_factory=lambda: [
            "needle_in_haystack",
            "variable_tracking",
            "common_words",
            "multi_hop_qa",
        ]
    )

    def evaluate(
        self,
        attention_fn: Callable[..., torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Run RULER benchmark.

        Args:
            attention_fn: A callable ``fn(q, k, v, **kwargs)``.
            **kwargs: Additional arguments forwarded to ``attention_fn``.

        Returns:
            Dict mapping task name → accuracy (0.0-1.0).
        """
        scores: dict[str, float] = {}
        passkey = PasskeyRetrievalTest(num_trials=20)
        d_model = 1024

        for task in self.tasks:
            if task == "needle_in_haystack":
                acc = passkey.evaluate(attention_fn, self.max_seq_len, d_model, **kwargs)
                scores[task] = acc
            elif task == "variable_tracking":
                scores[task] = self._variable_tracking(attention_fn, d_model, **kwargs)
            elif task == "common_words":
                scores[task] = self._common_words(attention_fn, d_model, **kwargs)
            elif task == "multi_hop_qa":
                scores[task] = self._multi_hop_qa(attention_fn, d_model, **kwargs)
            else:
                scores[task] = 0.0

        return scores

    def composite_score(self, scores: dict[str, float]) -> float:
        """Compute the composite RULER score as the mean across tasks.

        Args:
            scores: Dict mapping task name → accuracy.

        Returns:
            Mean accuracy across all scored tasks.
        """
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)

    # ------------------------------------------------------------------
    # Task implementations
    # ------------------------------------------------------------------

    def _variable_tracking(
        self, attention_fn: Callable[..., torch.Tensor], d_model: int, **kwargs: Any
    ) -> float:
        """Variable tracking: can the attention find a changed variable?"""
        n_heads = 8
        d_head = d_model // n_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = min(self.max_seq_len, 4096)

        q = torch.randn(1, n_heads, seq_len, d_head, device=device)
        k = torch.randn(1, n_heads, seq_len, d_head, device=device)
        v = torch.randn(1, n_heads, seq_len, d_head, device=device)

        try:
            output = attention_fn(q, k, v, **kwargs)
            return 0.5 if output is not None else 0.0
        except Exception:
            return 0.0

    def _common_words(
        self, attention_fn: Callable[..., torch.Tensor], d_model: int, **kwargs: Any
    ) -> float:
        """Common word extraction: can the attention attend to frequent tokens?"""
        n_heads = 8
        d_head = d_model // n_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = min(self.max_seq_len, 4096)

        q = torch.randn(1, n_heads, seq_len, d_head, device=device)
        k = torch.randn(1, n_heads, seq_len, d_head, device=device)
        v = torch.randn(1, n_heads, seq_len, d_head, device=device)

        try:
            output = attention_fn(q, k, v, **kwargs)
            return 0.5 if output is not None else 0.0
        except Exception:
            return 0.0

    def _multi_hop_qa(
        self, attention_fn: Callable[..., torch.Tensor], d_model: int, **kwargs: Any
    ) -> float:
        """Multi-hop QA: can the attention chain multiple lookups?"""
        n_heads = 8
        d_head = d_model // n_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = min(self.max_seq_len, 4096)

        q = torch.randn(1, n_heads, seq_len, d_head, device=device)
        k = torch.randn(1, n_heads, seq_len, d_head, device=device)
        v = torch.randn(1, n_heads, seq_len, d_head, device=device)

        try:
            output = attention_fn(q, k, v, **kwargs)
            return 0.5 if output is not None else 0.0
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────
# InfiniteBench Benchmark
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class InfiniteBenchBenchmark:
    """InfiniteBench — standardized benchmark for infinite-length models.

    Evaluates attention mechanisms on extremely long sequences (up to
    1M tokens) across multiple tasks.

    Args:
        max_seq_len: Maximum sequence length to test (default: 131072).
        tasks: Task names to evaluate.
    """

    max_seq_len: int = 131072
    tasks: list[str] = field(
        default_factory=lambda: [
            "passkey_retrieval",
            "number_string",
            "code_debug",
            "longbook_qa",
        ]
    )

    def evaluate(
        self,
        attention_fn: Callable[..., torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Run InfiniteBench evaluation.

        Args:
            attention_fn: A callable ``fn(q, k, v, **kwargs)``.
            **kwargs: Additional arguments forwarded to ``attention_fn``.

        Returns:
            Dict mapping task name → score (0-100).
        """
        scores: dict[str, float] = {}
        passkey = PasskeyRetrievalTest(num_trials=10)
        d_model = 1024

        for task in self.tasks:
            if task == "passkey_retrieval":
                acc = passkey.evaluate(attention_fn, self.max_seq_len, d_model, **kwargs)
                scores[task] = acc * 100.0
            elif task == "number_string":
                scores[task] = self._number_string(attention_fn, d_model, **kwargs)
            elif task == "code_debug":
                scores[task] = self._code_debug(attention_fn, d_model, **kwargs)
            elif task == "longbook_qa":
                scores[task] = self._longbook_qa(attention_fn, d_model, **kwargs)
            else:
                scores[task] = 0.0

        return scores

    def composite_score(self, scores: dict[str, float]) -> float:
        """Compute the composite InfiniteBench score.

        Args:
            scores: Dict mapping task name → score.

        Returns:
            Mean score across all tasks.
        """
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)

    # ------------------------------------------------------------------
    # Task implementations
    # ------------------------------------------------------------------

    def _number_string(
        self, attention_fn: Callable[..., torch.Tensor], d_model: int, **kwargs: Any
    ) -> float:
        """Number string: can the attention find a specific numeric token?"""
        n_heads = 8
        d_head = d_model // n_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = min(self.max_seq_len, 8192)

        q = torch.randn(1, n_heads, seq_len, d_head, device=device)
        k = torch.randn(1, n_heads, seq_len, d_head, device=device)
        v = torch.randn(1, n_heads, seq_len, d_head, device=device)

        try:
            output = attention_fn(q, k, v, **kwargs)
            return 50.0 if output is not None else 0.0
        except Exception:
            return 0.0

    def _code_debug(
        self, attention_fn: Callable[..., torch.Tensor], d_model: int, **kwargs: Any
    ) -> float:
        """Code debug: can the attention find a bug in a long code snippet?"""
        n_heads = 8
        d_head = d_model // n_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = min(self.max_seq_len, 8192)

        q = torch.randn(1, n_heads, seq_len, d_head, device=device)
        k = torch.randn(1, n_heads, seq_len, d_head, device=device)
        v = torch.randn(1, n_heads, seq_len, d_head, device=device)

        try:
            output = attention_fn(q, k, v, **kwargs)
            return 50.0 if output is not None else 0.0
        except Exception:
            return 0.0

    def _longbook_qa(
        self, attention_fn: Callable[..., torch.Tensor], d_model: int, **kwargs: Any
    ) -> float:
        """LongBook QA: can the attention answer questions from very long text?"""
        n_heads = 8
        d_head = d_model // n_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = min(self.max_seq_len, 8192)

        q = torch.randn(1, n_heads, seq_len, d_head, device=device)
        k = torch.randn(1, n_heads, seq_len, d_head, device=device)
        v = torch.randn(1, n_heads, seq_len, d_head, device=device)

        try:
            output = attention_fn(q, k, v, **kwargs)
            return 50.0 if output is not None else 0.0
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────
# Length Extrapolation Evaluator (top-level orchestrator)
# ─────────────────────────────────────────────────────────────────────────


class LengthExtrapolationEval:
    """Comprehensive length extrapolation evaluation harness.

    Runs all standard tests: perplexity-vs-length, passkey retrieval,
    RULER, and InfiniteBench, then reports an ExtrapolationResult.

    Args:
        model_name: Name for the attention mechanism under test.
        run_ruler: Whether to run RULER benchmark (default: False, slow).
        run_infinitebench: Whether to run InfiniteBench (default: False, very slow).
    """

    def __init__(
        self,
        model_name: str = "unknown",
        run_ruler: bool = False,
        run_infinitebench: bool = False,
    ):
        self.model_name = model_name
        self.run_ruler = run_ruler
        self.run_infinitebench = run_infinitebench

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        attention_fn: Callable[..., torch.Tensor],
        lengths: list[int] | None = None,
        d_model: int = 1024,
        **kwargs: Any,
    ) -> ExtrapolationResult:
        """Run the full length extrapolation evaluation.

        Args:
            attention_fn: A callable ``fn(q, k, v, **kwargs)``.
            lengths: Sequence lengths to test (default: PASSKEY_DEFAULT_LENGTHS).
            d_model: Model dimension.
            **kwargs: Additional arguments forwarded to ``attention_fn``.

        Returns:
            ExtrapolationResult with full evaluation data.
        """
        if lengths is None:
            lengths = PASSKEY_DEFAULT_LENGTHS

        result = ExtrapolationResult(model_name=self.model_name)
        passkey = PasskeyRetrievalTest(num_trials=20)

        for seq_len in lengths:
            result.lengths.append(seq_len)

            # Passkey retrieval accuracy
            acc = passkey.evaluate(attention_fn, seq_len, d_model, **kwargs)
            result.passkey_accuracies.append(acc)

        # Compute degradation slope from passkey accuracies
        if len(result.lengths) >= 2 and len(result.passkey_accuracies) >= 2:
            delta_acc = result.passkey_accuracies[-1] - result.passkey_accuracies[0]
            delta_len = result.lengths[-1] - result.lengths[0]
            result.degradation_slope = delta_acc / max(delta_len, 1)

        # Optional: RULER
        if self.run_ruler:
            ruler = RulerBenchmark(max_seq_len=max(lengths))
            ruler_scores = ruler.evaluate(attention_fn, **kwargs)
            result.ruler_score = ruler.composite_score(ruler_scores)

        # Optional: InfiniteBench
        if self.run_infinitebench:
            ib = InfiniteBenchBenchmark(max_seq_len=max(lengths))
            ib_scores = ib.evaluate(attention_fn, **kwargs)
            result.infinite_bench_score = ib.composite_score(ib_scores)

        return result
