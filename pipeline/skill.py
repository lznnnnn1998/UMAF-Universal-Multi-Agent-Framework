"""Skill Summarizer Pipeline — scanner → 4 parallel detectors → aggregator → writer."""

from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from skill.scanner import SkillScannerRole
from skill.detectors import (ConfigDocsDetectorRole, InfraDetectorRole,
                              JSDetectorRole, PythonDetectorRole)
from skill.aggregator import SkillAggregatorRole
from skill.writer import SkillReportWriterRole
from .base import BasePipeline


_DETECTOR_CLASSES: dict[str, type] = {
    "Python": PythonDetectorRole,
    "JavaScript": JSDetectorRole,
    "Infrastructure": InfraDetectorRole,
    "Configuration & Documentation": ConfigDocsDetectorRole,
}

_DETECTOR_OUTPUT_FILES: dict[str, str] = {
    "Python": "python_report.json",
    "JavaScript": "javascript_report.json",
    "Infrastructure": "infrastructure_report.json",
    "Configuration & Documentation": "configdocs_report.json",
}


def _run_detector(item: dict[str, Any], working_dir: str, backend: str) -> dict[str, Any]:
    """Execute a single domain detector with LLM + fallback to deterministic scan."""
    domain = item.get("domain", "")
    detector_cls = _DETECTOR_CLASSES.get(domain)
    output_file = _DETECTOR_OUTPUT_FILES.get(domain, "")
    project_scan = item.get("project_scan", {})

    if detector_cls is None:
        return {"output_file": "", "domain": domain, "data": {},
                "summary": f"Unknown domain: {domain}", "files": []}

    try:
        role = detector_cls()
        try:
            report = role.execute(working_dir=working_dir, backend=backend,
                                  project_dir=".", project_scan=project_scan)
        except Exception:
            report = {}

        if not report or not report.get("domain"):
            report = role._fallback_detect(project_dir=".", working_dir=working_dir)

        report_path = os.path.join(working_dir, output_file)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        skill_count = len(report.get("skills", []))
        return {
            "output_file": output_file, "domain": domain, "data": report,
            "summary": f"Detected {skill_count} skills in the {domain} domain" if skill_count
            else f"No skills detected in the {domain} domain",
            "files": [output_file],
        }
    except Exception as exc:
        return {"output_file": "", "domain": domain, "data": {},
                "summary": f"Detector exception: {exc}", "files": []}


class SkillState(TypedDict):
    input_spec: str
    working_dir: str
    backend: str
    project_scan: dict[str, Any]
    detector_outputs: list[dict[str, Any]]
    skill_inventory: dict[str, Any]
    status: str


class SkillPipeline(BasePipeline):
    """Skill Summarizer Pipeline with 7-agent fan-out/fan-in topology.

    scanner → 4 parallel detectors → aggregator → writer → END
    """

    name = "skill"
    default_output_dir = "skill_output"

    def _decompose(self, input_spec: str) -> list[dict[str, Any]]:
        return []

    def _display_decomposition(self, sub_tasks: list[dict]):
        print("Skill Summarizer Pipeline — analyzing project to detect skills.")
        print(f"Backend: {self.backend}")

    def _build_initial_state(self, input_spec: str, sub_tasks: list[dict]) -> dict:
        return {
            "input_spec": input_spec, "working_dir": self.working_dir,
            "backend": self.backend, "project_scan": {},
            "detector_outputs": [], "skill_inventory": {},
            "status": "initialized",
        }

    def _build_graph(self):
        working_dir = self.working_dir
        backend = self.backend

        def _scanner_node(state: SkillState) -> dict:
            print("\n[scanner] Analyzing project directory structure...")
            project_dir = state.get("input_spec", "")
            existing = state.get("project_scan", {})
            if existing and existing.get("file_categories"):
                n = existing.get("total_files", "?")
                print(f"  [scanner] Using existing scan ({n} files)")
                return {"status": "scanned"}

            scan: dict[str, Any] = {}
            try:
                role = SkillScannerRole()
                scan = role.execute(working_dir=working_dir, backend=state.get("backend", backend),
                                    project_dir=project_dir)
            except Exception as exc:
                print(f"  [scanner] Agent error: {exc}")

            if not scan or not scan.get("file_categories"):
                print("  [scanner] Falling back to deterministic scanner...")
                scan = SkillScannerRole._fallback_scanner(project_dir=project_dir, working_dir=working_dir)
                scan_path = os.path.join(working_dir, "project_scan.json")
                try:
                    with open(scan_path, "w") as f:
                        json.dump(scan, f, indent=2, default=str)
                except OSError:
                    pass

            total = scan.get("total_files", 0)
            print(f"  [scanner] Found {total} files")
            return {"project_scan": scan, "status": "scanned"}

        def _detectors_node(state: SkillState) -> dict:
            print("\n[detectors] Running 4 domain detectors in parallel...")
            project_scan = state.get("project_scan", {})
            items: list[dict[str, Any]] = [
                {"domain": "Python", "project_scan": project_scan},
                {"domain": "JavaScript", "project_scan": project_scan},
                {"domain": "Infrastructure", "project_scan": project_scan},
                {"domain": "Configuration & Documentation", "project_scan": project_scan},
            ]
            outputs, succeeded, failed = BasePipeline._run_parallel_agents(
                items, _run_detector, working_dir, state.get("backend", backend),
                max_workers=4,
            )
            for out in outputs:
                of = out.get("output_file", "")
                if of:
                    fp = os.path.join(working_dir, of)
                    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
                        out["output_file"] = ""
                        if of in out.get("files", []):
                            out["files"].remove(of)

            ok = sum(1 for o in outputs if o.get("output_file"))
            print(f"  [detectors] {ok}/{len(outputs)} succeeded")
            for out in outputs:
                mark = "✓" if out.get("output_file") else "✗"
                print(f"    {mark} {out['domain']}: {out.get('summary', 'no output')[:120]}")
            return {"detector_outputs": outputs, "status": "detected" if ok > 0 else "error_no_detector_outputs"}

        def _aggregator_node(state: SkillState) -> dict:
            print("\n[aggregator] Aggregating domain reports...")
            detector_outputs = state.get("detector_outputs", [])
            project_dir = state.get("input_spec", "")
            inventory: dict[str, Any] = {}
            if any(d.get("output_file") for d in detector_outputs):
                try:
                    role = SkillAggregatorRole()
                    inventory = role.execute(working_dir=working_dir,
                                             backend=state.get("backend", backend),
                                             detector_outputs=detector_outputs)
                except Exception as exc:
                    print(f"  [aggregator] Agent error: {exc}")

            if not inventory or not inventory.get("skills"):
                print("  [aggregator] Falling back to rule-based aggregation...")
                inventory = SkillAggregatorRole._fallback_aggregator(project_dir=project_dir, working_dir=working_dir)
                inv_path = os.path.join(working_dir, "skill_inventory.json")
                try:
                    with open(inv_path, "w") as f:
                        json.dump(inventory, f, indent=2, default=str)
                except OSError:
                    pass

            total = inventory.get("summary", {}).get("total_skills", 0)
            print(f"  [aggregator] Aggregated {total} skills")
            return {"skill_inventory": inventory, "status": "aggregated"}

        def _writer_node(state: SkillState) -> dict:
            print("\n[writer] Generating final reports...")
            inventory = state.get("skill_inventory", {})
            project_dir = state.get("input_spec", "")
            proj_name = os.path.basename(project_dir.rstrip("/")) or "Project"

            if not inventory.get("skills"):
                print("  [writer] No skill inventory — nothing to write")
                return {"status": "error_no_inventory"}

            agent_ok = False
            try:
                role = SkillReportWriterRole()
                role.execute(working_dir=working_dir, backend=state.get("backend", backend),
                             project_name=proj_name, skill_inventory=inventory)
                if os.path.exists(os.path.join(working_dir, "skills.json")):
                    agent_ok = True
            except Exception as exc:
                print(f"  [writer] Agent error: {exc}")

            if not agent_ok:
                print("  [writer] Falling back to template generation...")
                skills_data = SkillReportWriterRole._fallback_skills_json(proj_name, inventory)
                SkillReportWriterRole._write_skills_json(working_dir, skills_data)
                SkillReportWriterRole._fallback_report_md(proj_name, inventory, working_dir)

            for fname in ("skills.json", "skills_report.md"):
                fpath = os.path.join(working_dir, fname)
                if os.path.exists(fpath):
                    print(f"  [writer] ✓ {fname} ({os.path.getsize(fpath)} bytes)")
                else:
                    print(f"  [writer] ✗ {fname} not found")
            return {"status": "written"}

        workflow = StateGraph(SkillState)
        workflow.add_node("scanner", _scanner_node)
        workflow.add_node("detectors", _detectors_node)
        workflow.add_node("aggregator", _aggregator_node)
        workflow.add_node("writer", _writer_node)
        workflow.set_entry_point("scanner")

        flow = {
            "initialized": "scanner", "scanned": "detectors",
            "detected": "aggregator", "detected_partial": "aggregator",
            "aggregated": "writer", "written": END,
        }
        terminal = {"error_no_detector_outputs", "error_no_inventory"}
        router = BasePipeline._status_router(flow, terminal)
        for node in ("scanner", "detectors", "aggregator", "writer"):
            workflow.add_conditional_edges(node, router, {
                "scanner": "scanner", "detectors": "detectors",
                "aggregator": "aggregator", "writer": "writer", END: END,
            })
        return workflow.compile()

    def _print_results(self, final_state: dict):
        print("-" * 60)
        status = final_state.get("status", "unknown")
        print(f"Skill Summarizer Pipeline — {status.upper()}")
        print("-" * 60)
        scan = final_state.get("project_scan", {})
        if scan:
            print(f"\nProject Scan:")
            print(f"   Total files: {scan.get('total_files', '?')}")
            print(f"   Directories: {scan.get('total_dirs', '?')}")
            cats = scan.get("file_categories", {})
            if cats:
                for cat_name in ("source", "test", "config", "docs"):
                    count = len(cats.get(cat_name, []))
                    if count:
                        print(f"   {cat_name}: {count} files")
        detectors = final_state.get("detector_outputs", [])
        if detectors:
            print(f"\nDomain Detectors:")
            for d in detectors:
                mark = "✓" if d.get("output_file") else "✗"
                data = d.get("data", {})
                sc = len(data.get("skills", []))
                print(f"   {mark} {d['domain']}: {sc} skills")
        inventory = final_state.get("skill_inventory", {})
        if inventory:
            s = inventory.get("summary", {})
            print(f"\nAggregated Inventory:")
            print(f"   Total skills: {s.get('total_skills', 0)}")
            domains = s.get("domains_covered", [])
            if domains:
                print(f"   Domains: {', '.join(domains)}")
        print(f"\nOutputs in: {self.working_dir}")
