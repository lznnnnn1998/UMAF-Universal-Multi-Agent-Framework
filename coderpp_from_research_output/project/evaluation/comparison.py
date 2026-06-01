"""
Comparison Table — Tabular comparison of attention architectures.

Generates comparison tables across attention mechanism implementations
with standard dimensions and optional LaTeX output.

Usage:
    from evaluation.comparison import ComparisonTable, ComparisonRow, COMPARISON_DIMENSIONS

    table = ComparisonTable()
    table.add_row("FlashAttention v3", complexity="O(N)", memory="O(N)")
    latex = table.to_latex()
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────
# Standard comparison dimensions
# ─────────────────────────────────────────────────────────────────────────

COMPARISON_DIMENSIONS: list[str] = [
    "name",
    "algorithm",
    "complexity",
    "memory",
    "causal",
    "block_size_q",
    "block_size_kv",
    "fp8_support",
    "backward",
    "gpu_support",
    "max_seq_len",
]


# ─────────────────────────────────────────────────────────────────────────
# Comparison Row
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ComparisonRow:
    """A single row in a comparison table.

    Attributes:
        name: Name of the attention mechanism.
        algorithm: Algorithm variant (e.g. "FlashAttention v2").
        complexity: Time complexity (e.g. "O(N²)", "O(N log N)").
        memory: Memory complexity.
        causal: Whether causal masking is supported.
        block_size_q: Default Q block size.
        block_size_kv: Default KV block size.
        fp8_support: Whether FP8 computation is supported.
        backward: Whether backward pass is supported.
        gpu_support: Supported GPU architectures.
        max_seq_len: Maximum sequence length tested.
    """

    name: str
    algorithm: str = ""
    complexity: str = "O(N²)"
    memory: str = "O(N²)"
    causal: str = "Yes"
    block_size_q: str = "32"
    block_size_kv: str = "32"
    fp8_support: str = "No"
    backward: str = "Yes"
    gpu_support: str = "Ampere+"
    max_seq_len: str = "N/A"

    def to_dict(self) -> dict[str, str]:
        """Convert to a flat dict for serialization."""
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "complexity": self.complexity,
            "memory": self.memory,
            "causal": self.causal,
            "block_size_q": self.block_size_q,
            "block_size_kv": self.block_size_kv,
            "fp8_support": self.fp8_support,
            "backward": self.backward,
            "gpu_support": self.gpu_support,
            "max_seq_len": self.max_seq_len,
        }

    def __repr__(self) -> str:
        return (
            f"ComparisonRow({self.name}, algorithm={self.algorithm}, "
            f"complexity={self.complexity}, memory={self.memory})"
        )


# ─────────────────────────────────────────────────────────────────────────
# Comparison Table
# ─────────────────────────────────────────────────────────────────────────


class ComparisonTable:
    """A comparison table for attention mechanism implementations.

    Collects ComparisonRow entries and renders them as a formatted table,
    CSV, or LaTeX document.

    Args:
        caption: Table caption (used in LaTeX output).
        label: LaTeX label for cross-referencing.
        dimensions: List of column names to include (default: all).
    """

    def __init__(
        self,
        caption: str = "Comparison of Attention Mechanism Implementations",
        label: str = "tab:attention-comparison",
        dimensions: list[str] | None = None,
    ):
        self.caption = caption
        self.label = label
        self.dimensions = dimensions or COMPARISON_DIMENSIONS
        self.rows: list[ComparisonRow] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_row(
        self,
        name: str,
        algorithm: str = "",
        complexity: str = "O(N²)",
        memory: str = "O(N²)",
        causal: str = "Yes",
        block_size_q: str = "32",
        block_size_kv: str = "32",
        fp8_support: str = "No",
        backward: str = "Yes",
        gpu_support: str = "Ampere+",
        max_seq_len: str = "N/A",
    ) -> ComparisonRow:
        """Add a row to the comparison table.

        Args:
            name: Name of the attention mechanism.
            algorithm: Algorithm variant.
            complexity: Time complexity.
            memory: Memory complexity.
            causal: Causal masking support.
            block_size_q: Default Q block size.
            block_size_kv: Default KV block size.
            fp8_support: FP8 support.
            backward: Backward pass support.
            gpu_support: Supported GPU architectures.
            max_seq_len: Maximum sequence length.

        Returns:
            The added ComparisonRow.
        """
        row = ComparisonRow(
            name=name,
            algorithm=algorithm,
            complexity=complexity,
            memory=memory,
            causal=causal,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
            fp8_support=fp8_support,
            backward=backward,
            gpu_support=gpu_support,
            max_seq_len=max_seq_len,
        )
        self.rows.append(row)
        return row

    def add_row_obj(self, row: ComparisonRow) -> None:
        """Add an existing ComparisonRow to the table."""
        self.rows.append(row)

    def get_row(self, name: str) -> ComparisonRow | None:
        """Get a row by name.

        Args:
            name: Name of the attention mechanism.

        Returns:
            ComparisonRow or None if not found.
        """
        for row in self.rows:
            if row.name == name:
                return row
        return None

    def to_dict(self) -> list[dict[str, str]]:
        """Export all rows as a list of dicts."""
        return [row.to_dict() for row in self.rows]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_string(self, columns: list[str] | None = None) -> str:
        """Format the table as a plain-text aligned string.

        Args:
            columns: Columns to include (default: name, complexity, memory).

        Returns:
            Formatted string.
        """
        if columns is None:
            columns = ["name", "complexity", "memory", "causal", "fp8_support"]

        if not self.rows:
            return "(empty table)"

        # Compute column widths
        widths: dict[str, int] = {}
        for col in columns:
            widths[col] = len(col)
            for row in self.rows:
                val = str(getattr(row, col, ""))
                widths[col] = max(widths[col], len(val))

        # Header
        header = " | ".join(col.ljust(widths[col]) for col in columns)
        sep = "-+-".join("-" * widths[col] for col in columns)
        parts = [header, sep]

        # Rows
        for row in self.rows:
            line = " | ".join(
                str(getattr(row, col, "")).ljust(widths[col]) for col in columns
            )
            parts.append(line)

        return "\n".join(parts)

    def to_latex(
        self, columns: list[str] | None = None, use_booktabs: bool = True
    ) -> str:
        """Generate a LaTeX tabular environment.

        Args:
            columns: Columns to include (default: all COMPARISON_DIMENSIONS).
            use_booktabs: Whether to use booktabs rules (default: True).

        Returns:
            LaTeX source string.
        """
        if columns is None:
            columns = self.dimensions

        if not self.rows:
            return "% empty table"

        n_cols = len(columns)
        col_spec = "l" + "c" * (n_cols - 1)

        lines: list[str] = []
        lines.append("\\begin{table}[htbp]")
        lines.append("  \\centering")
        lines.append(f"  \\caption{{{self._latex_escape(self.caption)}}}")
        lines.append(f"  \\label{{{self.label}}}")
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")

        if use_booktabs:
            lines.append("    \\toprule")

        # Header
        header = "    " + " & ".join(
            self._format_latex_header(col) for col in columns
        ) + " \\\\"
        lines.append(header)

        if use_booktabs:
            lines.append("    \\midrule")
        else:
            lines.append("    \\hline")

        # Body
        for row in self.rows:
            vals = [str(getattr(row, col, "")) for col in columns]
            escaped = [self._latex_escape(v) for v in vals]
            line = "    " + " & ".join(escaped) + " \\\\"
            lines.append(line)

        if use_booktabs:
            lines.append("    \\bottomrule")
        else:
            lines.append("    \\hline")

        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def to_csv(self, columns: list[str] | None = None) -> str:
        """Export as CSV string.

        Args:
            columns: Columns to include (default: all COMPARISON_DIMENSIONS).

        Returns:
            CSV string.
        """
        if columns is None:
            columns = self.dimensions

        lines: list[str] = []
        # Header
        lines.append(",".join(columns))
        # Body
        for row in self.rows:
            vals = [str(getattr(row, col, "")) for col in columns]
            # Quote fields with commas
            quoted = [f'"{v}"' if "," in v else v for v in vals]
            lines.append(",".join(quoted))

        return "\n".join(lines)

    def to_markdown(self, columns: list[str] | None = None) -> str:
        """Export as a GitHub-flavored Markdown table.

        Args:
            columns: Columns to include (default: name, complexity, memory).

        Returns:
            Markdown string.
        """
        if columns is None:
            columns = ["name", "complexity", "memory", "causal", "fp8_support"]

        if not self.rows:
            return "_(empty table)_"

        lines: list[str] = []

        # Header
        header = "| " + " | ".join(columns) + " |"
        lines.append(header)

        # Separator
        sep = "|" + "|".join(" --- " for _ in columns) + "|"
        lines.append(sep)

        # Body
        for row in self.rows:
            vals = [str(getattr(row, col, "")) for col in columns]
            line = "| " + " | ".join(vals) + " |"
            lines.append(line)

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"ComparisonTable(rows={len(self.rows)})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _latex_escape(text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            "\\": "\\textbackslash ",
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "~": "\\textasciitilde ",
            "^": "\\textasciicircum ",
        }
        result = text
        for char, repl in replacements.items():
            result = result.replace(char, repl)
        return result

    @staticmethod
    def _format_latex_header(col: str) -> str:
        """Format a column name for LaTeX with underscores escaped."""
        return col.replace("_", "\\_")
