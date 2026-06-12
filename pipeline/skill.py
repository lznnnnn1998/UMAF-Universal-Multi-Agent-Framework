"""Skill Summarizer Pipeline — scanner → 4 parallel detectors → aggregator → writer."""

from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from skill.scanner import SkillScannerRole
from skill.detectors import (DomainExpertiseDetectorRole, MethodologyDetectorRole,
                              RigorDetectorRole, TechnicalCraftDetectorRole)
from skill.aggregator import SkillAggregatorRole
from skill.writer import SkillReportWriterRole
from .base import BasePipeline


_DETECTOR_CLASSES: dict[str, type] = {
    "Domain Expertise": DomainExpertiseDetectorRole,
    "Technical Craft": TechnicalCraftDetectorRole,
    "Methodology & Tooling": MethodologyDetectorRole,
    "Depth & Rigor": RigorDetectorRole,
}

_DETECTOR_OUTPUT_FILES: dict[str, str] = {
    "Domain Expertise": "domain_expertise_report.json",
    "Technical Craft": "technical_craft_report.json",
    "Methodology & Tooling": "methodology_report.json",
    "Depth & Rigor": "rigor_report.json",
}


def _run_detector(item: dict[str, Any], working_dir: str, backend: str) -> dict[str, Any]:
    """Execute a single skill-dimension detector with LLM + fallback."""
    domain = item.get("domain", "")
    detector_cls = _DETECTOR_CLASSES.get(domain)
    output_file = _DETECTOR_OUTPUT_FILES.get(domain, "")
    artifact_analysis = item.get("artifact_analysis", {})

    if detector_cls is None:
        return {"output_file": "", "domain": domain, "data": {},
                "summary": f"Unknown domain: {domain}", "files": []}

    try:
        role = detector_cls()
        try:
            report = role.execute(working_dir=working_dir, backend=backend,
                                  project_dir=".", artifact_analysis=artifact_analysis)
        except Exception:
            report = {}

        if not report or not report.get("domain"):
            report = role._fallback_detect(project_dir=".", working_dir=working_dir)

        report_path = os.path.join(working_dir, output_file)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        tool_count = len(report.get("detected_tools", []))
        skill_count = len(report.get("inferred_skills", []))
        parts = []
        if tool_count:
            parts.append(f"{tool_count} tools")
        if skill_count:
            parts.append(f"{skill_count} skills")
        summary = f"Detected {', '.join(parts)}" if parts else f"No skills detected"
        return {
            "output_file": output_file, "domain": domain, "data": report,
            "summary": summary,
            "files": [output_file],
        }
    except Exception as exc:
        return {"output_file": "", "domain": domain, "data": {},
                "summary": f"Detector exception: {exc}", "files": []}


class SkillState(TypedDict):
    input_spec: str
    working_dir: str
    backend: str
    project_dir: str
    project_scan: dict[str, Any]
    artifact_analysis: dict[str, Any]
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
        # target_dir is the explicit --target argument; fall back to input_spec (requirement)
        target = getattr(self, "target_dir", None)
        project_dir = target or input_spec
        return {
            "input_spec": input_spec, "working_dir": self.working_dir,
            "backend": self.backend, "project_dir": project_dir,
            "project_scan": {},
            "artifact_analysis": {},
            "detector_outputs": [], "skill_inventory": {},
            "status": "initialized",
        }

    def _build_graph(self):
        working_dir = self.working_dir
        backend = self.backend

        def _scanner_node(state: SkillState) -> dict:
            print("\n[scanner] Analyzing artifact (classification + deep read)...")
            project_dir = state.get("project_dir") or state.get("input_spec", "")
            print(f"  Target directory: {project_dir}")
            existing_aa = state.get("artifact_analysis", {})

            # If we already have a deep analysis, skip
            if existing_aa and existing_aa.get("artifact_type"):
                at = existing_aa.get("artifact_type", {})
                print(f"  [scanner] Using existing analysis (type: {at.get('type', '?')})")
                return {"status": "scanned"}

            analysis: dict[str, Any] = {}
            try:
                role = SkillScannerRole()
                analysis = role.execute(working_dir=working_dir,
                                        backend=state.get("backend", backend),
                                        project_dir=project_dir)
            except Exception as exc:
                print(f"  [scanner] Agent error: {exc}")

            if not analysis or not analysis.get("artifact_type"):
                print("  [scanner] Falling back to deterministic analysis...")
                analysis = SkillScannerRole._fallback_deep_scanner(
                    project_dir=project_dir, working_dir=working_dir)

            at = analysis.get("artifact_type", {})
            print(f"  [scanner] Artifact type: {at.get('type', '?')} "
                  f"({at.get('confidence', '?')}) — {at.get('description', '')[:60]}")
            return {
                "project_scan": analysis.get("surface_scan", {}),
                "artifact_analysis": analysis,
                "status": "scanned",
            }

        def _detectors_node(state: SkillState) -> dict:
            print("\n[detectors] Running 4 skill-dimension detectors in parallel...")
            artifact_analysis = state.get("artifact_analysis", {})
            project_scan = state.get("project_scan", {})
            items: list[dict[str, Any]] = [
                {"domain": "Domain Expertise", "artifact_analysis": artifact_analysis, "project_scan": project_scan},
                {"domain": "Technical Craft", "artifact_analysis": artifact_analysis, "project_scan": project_scan},
                {"domain": "Methodology & Tooling", "artifact_analysis": artifact_analysis, "project_scan": project_scan},
                {"domain": "Depth & Rigor", "artifact_analysis": artifact_analysis, "project_scan": project_scan},
            ]
            # Limit parallelism for claude_cli backend (each detector spawns a
            # heavy claude -p subprocess; 4 in parallel causes OOM).
            max_w = 1 if state.get("backend", backend) == "claude_cli" else 4
            outputs, succeeded, failed = BasePipeline._run_parallel_agents(
                items, _run_detector, working_dir, state.get("backend", backend),
                max_workers=max_w,
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
            project_dir = state.get("project_dir") or state.get("input_spec", "")
            inventory: dict[str, Any] = {}
            if any(d.get("output_file") for d in detector_outputs):
                try:
                    role = SkillAggregatorRole()
                    inventory = role.execute(working_dir=working_dir,
                                             backend=state.get("backend", backend),
                                             detector_outputs=detector_outputs)
                except Exception as exc:
                    print(f"  [aggregator] Agent error: {exc}")

            if not inventory or not (inventory.get("tools") or inventory.get("inferred_skills") or inventory.get("skills")):
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
            project_dir = state.get("project_dir") or state.get("input_spec", "")
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
        print(f"Skill Summarizer Pipeline v2 — {status.upper()}")
        print("-" * 60)

        aa = final_state.get("artifact_analysis", {})
        if aa:
            at = aa.get("artifact_type", {})
            print(f"\nArtifact: {at.get('type', '?')} ({at.get('confidence', '?')})")
            meta = aa.get("metadata", {})
            if meta:
                print(f"   Files: {meta.get('total_files', '?')}")
                langs = meta.get("languages_detected", [])
                if langs:
                    print(f"   Languages: {', '.join(langs)}")

        detectors = final_state.get("detector_outputs", [])
        if detectors:
            print(f"\nSkill Dimension Detectors:")
            for d in detectors:
                mark = "✓" if d.get("output_file") else "✗"
                data = d.get("data", {})
                tc = len(data.get("detected_tools", []))
                sc = len(data.get("inferred_skills", []))
                parts = []
                if tc:
                    parts.append(f"{tc} tools")
                if sc:
                    parts.append(f"{sc} skills")
                print(f"   {mark} {d['domain']}: {', '.join(parts) if parts else 'no output'}")

        inventory = final_state.get("skill_inventory", {})
        if inventory:
            s = inventory.get("summary", {})
            print(f"\nAggregated Inventory:")
            print(f"   Tools: {s.get('total_tools', 0)}")
            print(f"   Inferred skills: {s.get('total_inferred_skills', 0)}")
            dims = s.get("dimensions_covered", [])
            if dims:
                print(f"   Dimensions: {', '.join(dims)}")
        print(f"\nOutputs in: {self.working_dir}")
