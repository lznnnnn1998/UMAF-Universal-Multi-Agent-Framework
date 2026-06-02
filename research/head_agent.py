import re
from typing import Any

from agent import AgentResult, BaseDecomposerRole
from tools import ToolRegistry


class ResearchDecomposerRole(BaseDecomposerRole):
    """Decompose a research topic into 2-8 sub-topics, scaled to its complexity."""

    agent_name = "head_decompose"
    max_steps = 25

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        return ToolRegistry.to_dicts(ToolRegistry.research_decomposer_tools(backend))

    # -- Template overrides ---------------------------------------------------

    def _role_prompt(self, input_spec: str, **context: Any) -> str:
        return (
            "You are a research coordinator. Analyze the complexity of this "
            "research topic and decompose it into an appropriate number of "
            "sub-topics — at least 2 and at most 12.\n\n"
            f"Research topic: {input_spec}"
        )

    def _sizing_guide(self) -> str:
        return (
            "- **Narrow / specific topic** (e.g. a single method or technique): 3-5 sub-topics\n"
            "- **Moderate topic** (e.g. a family of techniques or one research area): 5-8 sub-topics\n"
            "- **Broad / complex topic** (e.g. a whole field or comparing multiple paradigms): 8-12 sub-topics"
        )

    def _sub_unit_requirements(self) -> str:
        return (
            "- Be specific and self-contained with clear scope boundaries\n"
            "- Include concrete research questions or areas to explore\n"
            "- Cover different angles of the main topic\n"
            "- Declare dependencies on other sub-topics (by id) when findings from that sub-topic are required input\n"
            "- Be suitable for a 25-minute focused research session"
        )

    @staticmethod
    def _json_template() -> str:
        return """[
  {
    "id": 1,
    "title": "Short descriptive title",
    "description": "Detailed description of what to research, including specific questions to answer and angles to explore.",
    "dependencies": []
  },
  ...
]"""

    def _backend_instructions(self, backend: str) -> str:
        if backend == "claude_cli":
            return (
                "Use your own knowledge to decompose the topic — do NOT search "
                "the web. Output ONLY the JSON array, nothing else before or after. "
                "Then write TASK_COMPLETE."
            )
        return (
            "Output ONLY the JSON array, nothing else before or after. "
            "Then write TASK_COMPLETE."
        )

    @staticmethod
    def _fallback_decompose(input_spec: str) -> list[dict[str, Any]]:
        keywords = [s.strip() for s in re.split(r',| and | vs |;', input_spec) if len(s.strip()) >= 2]
        if not keywords:
            keywords = [input_spec]

        templates: list[dict[str, Any]] = []
        kw_limit = min(len(keywords), 12)
        for i in range(kw_limit):
            kw = keywords[i].strip().rstrip('.')
            templates.append({
                "id": i + 1,
                "title": f"{kw}: Mechanisms, Methods, and Key Results",
                "description": (
                    f"Deep-dive into '{kw}' within the context of {input_spec}. "
                    f"Investigate the underlying mechanisms, established methods, "
                    f"representative results, and identify the most influential papers "
                    f"and benchmarks."
                ),
            })

        base_id = len(templates)
        templates.append({
            "id": base_id + 1,
            "title": f"Comparative Analysis of Approaches in {input_spec}",
            "description": (
                f"Compare and contrast the major approaches within {input_spec}. "
                f"Analyze trade-offs in accuracy, efficiency, implementation complexity, "
                f"and applicability. Identify which methods work best under which conditions."
            ),
        })
        templates.append({
            "id": base_id + 2,
            "title": f"Open Problems and Emerging Directions in {input_spec}",
            "description": (
                f"Identify open research questions, recent breakthroughs, and promising "
                f"future directions in {input_spec}. Survey papers from the last 1-2 years "
                f"for emerging trends and unresolved challenges."
            ),
        })

        if len(templates) < 2:
            templates.append({
                "id": len(templates) + 1,
                "title": f"Overview and Key Techniques in {input_spec}",
                "description": (
                    f"Provide a comprehensive overview of {input_spec}, covering the "
                    f"foundational concepts, key techniques, and major milestones in the field."
                ),
            })

        return templates[:8]


def decompose_topic(topic: str, working_dir: str, backend: str = "deepseek") -> list[dict[str, Any]]:
    """Decompose a research topic into 2-8 sub-topics, scaled to its complexity.

    Returns a list of dicts with keys: id, title, description.
    """
    role = ResearchDecomposerRole()
    return role.execute(working_dir=working_dir, backend=backend, input_spec=topic)


# Backward-compatible module-level alias
_fallback_decompose = ResearchDecomposerRole._fallback_decompose
