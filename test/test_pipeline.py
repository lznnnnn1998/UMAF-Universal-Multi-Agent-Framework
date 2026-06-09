"""Tests for BasePipeline — shared infrastructure for all 6 pipelines."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import test.conftest  # noqa: F401 — loads tools_config.json, tmpdir fixture

from pipeline.base import BasePipeline


# ═══════════════════════════════════════════════════════════════════════════
# Instantiation
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiation:
    def test_default_working_dir_uses_repo_root(self):
        p = BasePipeline()
        assert p.name == "base"
        assert "output" in p.working_dir
        assert p.backend == "deepseek"

    def test_custom_working_dir(self):
        p = BasePipeline(working_dir="/tmp/test_infra")
        assert p.working_dir == "/tmp/test_infra"

    def test_all_flags(self):
        p = BasePipeline(backend="claude_cli", clean=True, resume=True, yes=True)
        assert p.backend == "claude_cli"
        assert p.clean is True
        assert p.resume is True
        assert p.yes is True

    def test_default_flags(self):
        p = BasePipeline()
        assert p.clean is False
        assert p.resume is False
        assert p.yes is False


# ═══════════════════════════════════════════════════════════════════════════
# Output directory management
# ═══════════════════════════════════════════════════════════════════════════

class TestManageOutputDir:
    def test_creates_dir_if_missing(self, tmpdir):
        wd = os.path.join(tmpdir, "new_dir")
        p = BasePipeline(working_dir=wd)
        p.manage_output_dir()
        assert os.path.isdir(wd)

    def test_clean_removes_existing(self, tmpdir):
        with open(os.path.join(tmpdir, "old_file.txt"), "w") as f:
            f.write("data")
        p = BasePipeline(working_dir=tmpdir, clean=True)
        p.manage_output_dir()
        assert os.path.isdir(tmpdir)
        assert not os.path.exists(os.path.join(tmpdir, "old_file.txt"))


# ═══════════════════════════════════════════════════════════════════════════
# Decomposition display and editing
# ═══════════════════════════════════════════════════════════════════════════

class TestDisplayDecomposition:
    def test_display_with_module_names(self, capsys):
        p = BasePipeline()
        sub_tasks = [
            {"id": 1, "module_name": "core", "description": "Core logic module"},
            {"id": 2, "module_name": "utils", "description": "Shared utilities"},
        ]
        p._display_decomposition(sub_tasks)
        out = capsys.readouterr().out
        assert "core" in out
        assert "utils" in out
        assert "Core logic" in out

    def test_display_falls_back_to_title(self, capsys):
        p = BasePipeline()
        sub_tasks = [{"id": 1, "title": "Research Topic", "description": "Deep dive"}]
        p._display_decomposition(sub_tasks)
        out = capsys.readouterr().out
        assert "Research Topic" in out


# ═══════════════════════════════════════════════════════════════════════════
# Topological level computation
# ═══════════════════════════════════════════════════════════════════════════

class TestTopologicalLevels:
    def test_no_dependencies_single_level(self):
        tasks = [
            {"id": 1, "title": "A"},
            {"id": 2, "title": "B"},
            {"id": 3, "title": "C"},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 1
        assert len(levels[0]) == 3

    def test_linear_dependency_chain(self):
        tasks = [
            {"id": 1, "title": "A", "dependencies": []},
            {"id": 2, "title": "B", "dependencies": [1]},
            {"id": 3, "title": "C", "dependencies": [2]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 3
        assert levels[0][0]["id"] == 1
        assert levels[1][0]["id"] == 2
        assert levels[2][0]["id"] == 3

    def test_mixed_dependencies(self):
        tasks = [
            {"id": 1, "title": "A", "dependencies": []},
            {"id": 2, "title": "B", "dependencies": []},
            {"id": 3, "title": "C", "dependencies": [1, 2]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 2
        level0_ids = {t["id"] for t in levels[0]}
        assert level0_ids == {1, 2}
        assert levels[1][0]["id"] == 3

    def test_diamond_dependency(self):
        tasks = [
            {"id": 1, "title": "Base", "dependencies": []},
            {"id": 2, "title": "Left", "dependencies": [1]},
            {"id": 3, "title": "Right", "dependencies": [1]},
            {"id": 4, "title": "Merge", "dependencies": [2, 3]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 3
        assert levels[0][0]["id"] == 1
        assert {t["id"] for t in levels[1]} == {2, 3}
        assert levels[2][0]["id"] == 4

    def test_string_dependency_by_module_name(self):
        tasks = [
            {"id": 1, "module_name": "core"},
            {"id": 2, "module_name": "api", "dependencies": ["core"]},
        ]
        levels = BasePipeline._topological_levels(tasks)
        assert len(levels) == 2
        assert levels[0][0]["module_name"] == "core"
        assert levels[1][0]["module_name"] == "api"

    def test_empty_list(self):
        assert BasePipeline._topological_levels([]) == [[]]


# ═══════════════════════════════════════════════════════════════════════════
# Dependency validation
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateDependencies:
    def test_valid_graph_no_issues(self):
        tasks = [
            {"id": 1, "module_name": "A"},
            {"id": 2, "module_name": "B", "dependencies": ["A"]},
        ]
        issues = BasePipeline._validate_dependencies(tasks)
        assert issues == []

    def test_empty_tasks(self):
        assert BasePipeline._validate_dependencies([]) == []

    def test_duplicate_module_names(self):
        tasks = [
            {"id": 1, "module_name": "core"},
            {"id": 2, "module_name": "core"},
        ]
        issues = BasePipeline._validate_dependencies(tasks)
        assert len(issues) >= 1
        assert any("Duplicate" in i for i in issues)

    def test_dependency_cycle_detected(self):
        tasks = [
            {"id": 1, "module_name": "A", "dependencies": [2]},
            {"id": 2, "module_name": "B", "dependencies": [1]},
        ]
        issues = BasePipeline._validate_dependencies(tasks)
        assert any("cycle" in i.lower() for i in issues)


# ═══════════════════════════════════════════════════════════════════════════
# Status router
# ═══════════════════════════════════════════════════════════════════════════

class TestStatusRouter:
    def test_maps_status_to_node(self):
        flow = {"ready": "worker", "done": "finish"}
        router = BasePipeline._status_router(flow)
        assert router({"status": "ready"}) == "worker"
        assert router({"status": "done"}) == "finish"

    def test_terminal_status_returns_end(self):
        from langgraph.graph import END
        flow = {"ready": "worker"}
        router = BasePipeline._status_router(flow, terminal_errors={"crashed"})
        assert router({"status": "crashed"}) == END

    def test_unknown_status_returns_end(self):
        from langgraph.graph import END
        flow = {"ready": "worker"}
        router = BasePipeline._status_router(flow)
        assert router({"status": "unknown"}) == END


# ═══════════════════════════════════════════════════════════════════════════
# Parallel agent runner
# ═══════════════════════════════════════════════════════════════════════════

class TestRunParallelAgents:
    def test_successful_agents(self, tmpdir):
        def agent_func(item, wd, be):
            return {
                "sub_task_id": item["id"],
                "files": [f"output_{item['id']}.py"],
                "output_file": f"output_{item['id']}.py",
            }

        items = [{"id": 1}, {"id": 2}]
        outputs, succeeded, failed = BasePipeline._run_parallel_agents(
            items, agent_func, tmpdir, "deepseek",
        )
        assert succeeded == 2
        assert failed == 0
        assert len(outputs) == 2

    def test_agent_with_no_files_counts_as_failed(self, tmpdir):
        def agent_func(item, wd, be):
            return {"sub_task_id": item["id"], "files": []}

        items = [{"id": 1}]
        outputs, succeeded, failed = BasePipeline._run_parallel_agents(
            items, agent_func, tmpdir, "deepseek",
        )
        assert succeeded == 0
        assert failed == 1

    def test_agent_exception_is_caught(self, tmpdir):
        def agent_func(item, wd, be):
            raise RuntimeError("boom")

        items = [{"id": 1}]
        outputs, succeeded, failed = BasePipeline._run_parallel_agents(
            items, agent_func, tmpdir, "deepseek",
        )
        assert failed == 1
        assert "Agent exception" in outputs[0]["summary"]

    def test_retry_on_failure(self, tmpdir):
        call_count: dict[int, int] = {}

        def agent_func(item, wd, be):
            iid = item["id"]
            call_count[iid] = call_count.get(iid, 0) + 1
            # Fails on first call, succeeds on second
            if call_count[iid] < 2:
                return {"sub_task_id": iid, "files": []}
            return {"sub_task_id": iid, "files": ["ok.py"], "output_file": "ok.py"}

        items = [{"id": 1}]
        outputs, succeeded, failed = BasePipeline._run_parallel_agents(
            items, agent_func, tmpdir, "deepseek", retry_failures=True, max_retries=1,
        )
        assert succeeded == 1
        assert failed == 0


# ═══════════════════════════════════════════════════════════════════════════
# Workers with dependencies (integration with topological levels)
# ═══════════════════════════════════════════════════════════════════════════

class TestRunWorkersWithDeps:
    def test_no_deps_flat_parallelism(self, tmpdir):
        def agent_func(item, wd, be):
            return {
                "sub_task_id": item["id"],
                "files": [f"mod_{item['id']}.py"],
                "output_file": f"mod_{item['id']}.py",
            }

        items = [
            {"id": 1, "title": "A"},
            {"id": 2, "title": "B"},
        ]
        outputs, succeeded, failed = BasePipeline._run_workers_with_deps(
            items, agent_func, tmpdir, "deepseek",
        )
        assert succeeded == 2
        assert failed == 0

    def test_dependency_outputs_injected(self, tmpdir):
        """Tasks receive _dependency_outputs when deps have completed."""
        captured: dict[int, Any] = {}

        def agent_func(item, wd, be):
            captured[item["id"]] = item.get("_dependency_outputs", [])
            return {
                "sub_task_id": item["id"],
                "module_name": item.get("module_name", f"mod_{item['id']}"),
                "files": [f"out_{item['id']}.py"],
                "output_file": f"out_{item['id']}.py",
            }

        items = [
            {"id": 1, "module_name": "base", "dependencies": []},
            {"id": 2, "module_name": "child", "dependencies": ["base"]},
        ]
        BasePipeline._run_workers_with_deps(items, agent_func, tmpdir, "deepseek")
        # Child task should have received dependency output from base
        assert len(captured.get(2, [])) > 0, f"Expected child to have _dependency_outputs, got: {captured}"

    def test_stop_on_failure_blocks_downstream(self, tmpdir):
        def agent_func(item, wd, be):
            if item["id"] == 1:
                return {"sub_task_id": 1, "files": []}  # fails
            return {
                "sub_task_id": item["id"],
                "module_name": item.get("module_name", ""),
                "files": ["ok.py"],
                "output_file": "ok.py",
            }

        items = [
            {"id": 1, "title": "must_succeed_first"},
            {"id": 2, "title": "child", "dependencies": [1]},
        ]
        outputs, succeeded, failed = BasePipeline._run_workers_with_deps(
            items, agent_func, tmpdir, "deepseek",
        )
        # Child was deferred — only 1 task actually ran
        assert succeeded == 0
        assert failed == 1


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
