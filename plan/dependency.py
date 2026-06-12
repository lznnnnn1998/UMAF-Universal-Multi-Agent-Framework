"""PlanDependencyAnalyzerRole — dependency graph analysis agent."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from tools import ToolRegistry
from agent import AgentResult, AgentRole
from utils import extract_json_object, safe_read


class PlanDependencyAnalyzerRole(AgentRole):
    """Analyzes the task tree to produce a comprehensive dependency graph.

    Supports four edge types: blocks, informs, enables, is_subtask_of.
    Computes the critical path through the tree, identifies parallelizable
    sub-trees (tasks sharing no dependency edges), and validates the graph
    is acyclic.

    Produces:
    - dependency_graph.json: machine-readable graph with nodes, edges, critical path
    - critical_path.md: human-readable summary of longest dependency chain
    """

    agent_name: str = "dependency_analyzer"
    max_steps: int = 15

    def tools_for_backend(self, backend: str) -> list[dict[str, Any]]:
        """Return Read-only tool specs."""
        if hasattr(ToolRegistry, "plan_dependency_analyzer_tools"):
            return ToolRegistry.to_dicts(ToolRegistry.plan_dependency_analyzer_tools())
        return ToolRegistry.to_dicts([ToolRegistry.READ_FILE])

    def build_task(self, backend: str, working_dir: str = ".",
                   task_tree: dict[str, Any] | None = None, **context: Any) -> str:
        """Build the dependency analysis prompt."""
        tree_summary = self._summarize_tree(task_tree)

        common = (
            f"You are a dependency analysis expert. Your job is to analyze a "
            f"hierarchical task tree and produce a comprehensive dependency graph "
            f"with critical path analysis.\n\n"
            f"## Task Tree Summary\n{tree_summary}\n\n"
            f"## Instructions\n"
            f"1. Read task_tree.json from the working directory for full details\n"
            f"2. For every pair of nodes where one depends on another, classify "
            f"the edge type:\n"
            f"   - **blocks**: B cannot start until A completes (hard dependency)\n"
            f"   - **informs**: A's output influences B's design but B can start earlier\n"
            f"   - **enables**: A provides infrastructure/capability B requires\n"
            f"   - **is_subtask_of**: B is a child/subtask of A\n"
            f"3. Compute the critical path: the longest chain of blocking dependencies "
            f"from any root node to any leaf\n"
            f"4. Identify parallelizable sub-trees: groups of tasks that share no "
            f"dependency edges with each other\n"
            f"5. Validate the graph is acyclic (no circular dependencies)\n\n"
            f"## Output Format\n"
            f'Write "dependency_graph.json" with this structure:\n'
            f"```json\n"
            f'{{\n'
            f'  "nodes": [\n'
            f'    {{"id": 1, "title": "Task name", "type": "goal"}}\n'
            f'  ],\n'
            f'  "edges": [\n'
            f'    {{\n'
            f'      "from": 2,\n'
            f'      "to": 5,\n'
            f'      "type": "blocks",\n'
            f'      "description": "Task 5 requires task 2 output"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "critical_path": {{\n'
            f'    "nodes": [1, 2, 5],\n'
            f'    "total_complexity": 15,\n'
            f'    "description": "Longest blocking chain through the tree"\n'
            f'  }},\n'
            f'  "parallelizable_groups": [\n'
            f'    [3, 4, 6]\n'
            f'  ],\n'
            f'  "is_acyclic": true,\n'
            f'  "cycle_details": [],\n'
            f'  "generated_at": "<ISO timestamp>"\n'
            f'}}\n'
            f"```\n\n"
            f"Also write a human-readable critical_path.md summarizing:\n"
            f"- The critical path nodes in order\n"
            f"- Total complexity along the critical path\n"
            f"- Key synchronization points where parallel branches must merge\n"
            f"- Estimated minimum timeline if all parallel work is fully utilized\n\n"
            f"The JSON object MUST appear INLINE in your response text.\n"
            f"Working directory: {working_dir}"
        )

        if backend == "claude_cli":
            backend_note = (
                "\n\nUse your own knowledge — do NOT search the web. "
                "Read task_tree.json first. Write dependency_graph.json and "
                "critical_path.md, then output TASK_COMPLETE."
            )
        else:
            backend_note = (
                "\n\nRead task_tree.json first. Write dependency_graph.json "
                "and critical_path.md, then output TASK_COMPLETE."
            )

        return common + backend_note

    def parse_result(self, result: AgentResult, working_dir: str = ".",
                     task_tree: dict[str, Any] | None = None, **context: Any) -> dict[str, Any]:
        """Extract dependency graph from agent response, disk file, or fallback."""
        dep_graph: dict[str, Any] = {}

        # 1. Try agent messages
        for msg in reversed(result.messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            json_str = extract_json_object(content)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    if "nodes" in parsed and "edges" in parsed:
                        dep_graph = parsed
                        break
                except json.JSONDecodeError:
                    continue

        # 2. Try disk
        if not dep_graph:
            path = os.path.join(working_dir, "dependency_graph.json")
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        dep_graph = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

        # 3. Fallback
        if not dep_graph:
            dep_graph = self._fallback_dependency(task_tree, working_dir)
            out_path = os.path.join(working_dir, "dependency_graph.json")
            try:
                with open(out_path, "w") as f:
                    json.dump(dep_graph, f, indent=2)
            except OSError:
                pass

        return dep_graph

    @staticmethod
    def _summarize_tree(task_tree: dict[str, Any] | None) -> str:
        """Create a short summary of the task tree for the prompt."""
        if not task_tree or not task_tree.get("tree"):
            return "(No task tree provided. Read task_tree.json from the working directory.)"

        def _count_nodes(nodes: list) -> int:
            count = len(nodes)
            for n in nodes:
                count += _count_nodes(n.get("children", []))
            return count

        tree = task_tree.get("tree", [])
        total = _count_nodes(tree)
        complexity = task_tree.get("complexity_level", "unknown")
        leaves = sum(1 for n in tree if not n.get("children"))

        def _fmt_node(n):
            nid = n.get("id", "?")
            ntitle = n.get("title", "?")
            return f"#{nid} {ntitle}"

        lines = [
            f"- Complexity: {complexity}",
            f"- Total nodes: {total}",
            f"- Root goals: {len(tree)}",
            f"- Roots: {', '.join(_fmt_node(n) for n in tree[:5])}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _fallback_dependency(task_tree: dict[str, Any] | None,
                             working_dir: str = ".") -> dict[str, Any]:
        """Build a simple dependency graph from the task tree structure."""
        if not task_tree or not task_tree.get("tree"):
            return {
                "_fallback": True,
                "nodes": [],
                "edges": [],
                "critical_path": {"nodes": [], "total_complexity": 0,
                                  "description": "No task tree available."},
                "parallelizable_groups": [],
                "is_acyclic": True,
                "cycle_details": [],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        # Extract all nodes and edges from the tree
        tree = task_tree.get("tree", [])
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        def _extract(node_list: list[dict[str, Any]]):
            for node in node_list:
                nodes.append({
                    "id": node.get("id"),
                    "title": node.get("title", ""),
                    "type": node.get("type", "task"),
                    "complexity": node.get("complexity", 1),
                })
                # Add parent-child edges
                for child in node.get("children", []):
                    edges.append({
                        "from": node.get("id"),
                        "to": child.get("id"),
                        "type": "is_subtask_of",
                        "description": f"'{child.get('title', '')}' "
                                       f"is a subtask of '{node.get('title', '')}'",
                    })
                # Add explicit dependency edges
                for dep_id in node.get("dependencies", []):
                    edges.append({
                        "from": dep_id,
                        "to": node.get("id"),
                        "type": "blocks",
                        "description": f"'{node.get('title', '')}' "
                                       f"depends on task {dep_id}",
                    })
                _extract(node.get("children", []))

        _extract(tree)

        # Simple critical path: find the longest dependency chain
        # Build adjacency for topological ordering
        adj: dict[int, list[int]] = {}
        in_degree: dict[int, int] = {n["id"]: 0 for n in nodes}
        for n in nodes:
            adj.setdefault(n["id"], [])
        for e in edges:
            if e["type"] == "blocks":
                adj.setdefault(e["from"], []).append(e["to"])
                in_degree[e["to"]] = in_degree.get(e["to"], 0) + 1

        # Topological sort for longest path
        topo: list[int] = []
        q = [nid for nid, deg in in_degree.items() if deg == 0]
        while q:
            node_id = q.pop(0)
            topo.append(node_id)
            for neighbor in adj.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    q.append(neighbor)

        # Longest path in DAG
        id_to_complexity = {n["id"]: n.get("complexity", 1) for n in nodes}
        dist: dict[int, int] = {n["id"]: 0 for n in nodes}
        parent: dict[int, int | None] = {n["id"]: None for n in nodes}
        for nid in topo:
            d = dist[nid] + id_to_complexity.get(nid, 1)
            for neighbor in adj.get(nid, []):
                if d > dist.get(neighbor, 0):
                    dist[neighbor] = d
                    parent[neighbor] = nid

        if dist:
            end = max(dist, key=lambda k: dist[k])
            critical = []
            cur: int | None = end
            total_complexity = 0
            while cur is not None:
                critical.insert(0, cur)
                total_complexity += id_to_complexity.get(cur, 1)
                cur = parent.get(cur)
        else:
            critical = []
            total_complexity = 0

        return {
            "_fallback": True,
            "nodes": nodes,
            "edges": edges,
            "critical_path": {
                "nodes": critical,
                "total_complexity": total_complexity,
                "description": "Fallback critical path — computed from task tree dependencies.",
            },
            "parallelizable_groups": [],
            "is_acyclic": True,
            "cycle_details": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
