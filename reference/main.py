import asyncio
import sys
import os
import uuid
import dataclasses
import subprocess
from pathlib import Path

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.apps.app import App

from core.database import init_db, read_findings, read_risk_scores, update_status
from core.sandbox import build_sandbox
from core.graph_loader import load_workflow_from_json, DEFAULT_SEED_PROMPT
from core.context import RunContext, current_run_context

APP_NAME = "mantis_graph"
USER_ID = "user1"

async def execute_sub_task(
    runner: Runner,
    session_service: InMemorySessionService,
    filepath: str,
    run_id: str,
    db_path: str = "",
    status_map: dict[str, str] | None = None,
    seed_prompt_template: str = DEFAULT_SEED_PROMPT,
) -> bool:
    """Executes the workflow graph for a single target file. Returns True if an error was encountered."""
    session_id = f"session_run_{run_id}_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    
    query_text = seed_prompt_template.format(filepath=filepath)
    new_message = types.Content(
        parts=[types.Part.from_text(text=query_text)],
        role="user"
    )
    
    print(f"\n[GRAPH EXECUTION] Triggered via: {filepath}")
    print("-" * 60)
    
    errored: set[str] = set()
    stamped_nodes: set[str] = set()
    last_banner: tuple[str | None, str | None] = (None, None)

    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=new_message):
            node_path = getattr(getattr(event, "node_info", None), "path", None)
            route = getattr(getattr(event, "actions", None), "route", None)

            if node_path:
                node_name = node_path.split("/")[-1].split("@")[0]
                if status_map and db_path and node_name in status_map and node_name not in stamped_nodes:
                    stamped_nodes.add(node_name)
                    update_status(db_path, filepath, run_id, status_map[node_name])

            banner = (node_path, route)
            if (node_path or route) and banner != last_banner:
                if node_path and route:
                    print(f"\n-- {node_path} -> {route}")
                elif node_path:
                    print(f"\n-- {node_path}")
                elif route:
                    print(f"\n-- {last_banner[0] or ''} -> {route}")
                last_banner = banner

            if getattr(event, "error_code", None):
                node_key = node_path or "unknown"
                errored.add(node_key)
                err_msg = getattr(event, "error_message", None) or f"ADK Event error: {event.error_code}"
                print(f"\n[EVENT ERROR {event.error_code}] {err_msg}", file=sys.stderr)
            else:
                if node_path and node_path in errored:
                    errored.discard(node_path)
                if hasattr(event, 'content') and event.content:
                    for part in getattr(event.content, "parts", []) or []:
                        if hasattr(part, 'text') and part.text:
                            print(part.text, end="", flush=True)
    finally:
        try:
            await session_service.delete_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        except Exception:
            pass
            
    print("\n" + "-" * 60)
    return bool(errored)

def is_binary_file(path: Path, block_size: int = 1024) -> bool:
    """Returns True if the file contains null bytes in its initial block."""
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(block_size)
    except OSError:
        return True

def discover_files(target: Path, db_path: str = "") -> list[str]:
    """Source files under `target`. Uses git's own view when available —
    a repo already declares what isn't source. Excludes binary files."""
    if target.is_file():
        return [str(target)] if not is_binary_file(target) else []
    try:
        out = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
             "-C", str(target), "ls-files", "-z",
             "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=10, check=True
        ).stdout
        paths = [target / p for p in out.split("\0") if p]
        if paths:
            return [
                str(p) for p in sorted(paths)
                if p.is_file() and str(p) != db_path and not is_binary_file(p)
            ]
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return [
        str(p) for p in sorted(target.rglob("*"))
        if p.is_file() and str(p) != db_path and not any(part.startswith(".") for part in p.parts) and not is_binary_file(p)
    ]

async def pipeline(scan_target: str) -> int:
    pipeline_dir = os.path.realpath(os.path.dirname(__file__))
    workflow_path = os.path.join(pipeline_dir, "workflow.json")
    
    if not os.path.exists(workflow_path):
        print(f"Error: Could not find workflow.json at {workflow_path}", file=sys.stderr)
        return 1

    try:
        workflow, config = load_workflow_from_json(workflow_path)
    except ValueError as e:
        print(f"Workflow Configuration Error:\n{e}", file=sys.stderr)
        return 2
        
    target_path = Path(scan_target).resolve()
    if not target_path.exists():
        print(f"Error: Target path does not exist: {target_path}", file=sys.stderr)
        return 1

    db_rel = config.get("db_path", "findings.db")
    db_path = os.path.realpath(os.path.join(pipeline_dir, db_rel))
    init_db(db_path)

    files_to_scan = discover_files(target_path, db_path)
    jail_dir = str(target_path)

    if not files_to_scan:
        print(f"No files found in target: {target_path}")
        return 0

    print(f"Compiling Graph Pipeline. Target: {target_path}")
    
    try:
        test_sandbox = build_sandbox(config.get("sandbox", {}), files_to_scan[0])
        await test_sandbox.aclose()
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Sandbox Configuration Error: {e}", file=sys.stderr)
        return 2

    run_app = App(
        name=APP_NAME,
        root_agent=workflow
    )
    
    session_service = InMemorySessionService()
    runner = Runner(
        app=run_app,
        session_service=session_service
    )

    run_id = str(uuid.uuid4())
    base_ctx = RunContext(
        jail_dir=jail_dir,
        db_path=db_path,
        target_file="",
        run_id=run_id
    )

    print(f"\n🚀 Engaging JSON Graph over {len(files_to_scan)} discrete inputs (Run ID: {run_id})...")
    
    failures = 0
    successes = 0
    try:
        for filepath in files_to_scan:
            sandbox = build_sandbox(config.get("sandbox", {}), filepath)
            branch_ctx = dataclasses.replace(base_ctx, target_file=filepath, sandbox=sandbox)
            current_run_context.set(branch_ctx)
            try:
                task_failed = await execute_sub_task(
                    runner,
                    session_service,
                    filepath,
                    run_id,
                    db_path=db_path,
                    status_map=config.get("on_enter_status", {}),
                    seed_prompt_template=config.get("seed_prompt", DEFAULT_SEED_PROMPT)
                )
                if task_failed:
                    failures += 1
                else:
                    successes += 1
            except Exception as e:
                print(f"PIPELINE CRITICAL ABORT IN TASK ({filepath}): {e}", file=sys.stderr)
                failures += 1
            finally:
                await sandbox.aclose()
    finally:
        await runner.close()
        findings = read_findings(db_path, run_id=run_id)
        scores = read_risk_scores(db_path, run_id=run_id)
        print(f"\n📊 Summary: {len(findings)} vulnerability finding(s) recorded across {successes} scanned file(s).")
        for f in findings:
            lines_str = f" (Lines: {f.get('line_numbers')})" if f.get('line_numbers') else ""
            mark = " (suppressed at review)" if f.get("status") == "reported" else ""
            print(f"  - [{f.get('severity', 'Unknown')}] {f.get('filepath')}: {f.get('title')}{lines_str}{mark}")
        if scores:
            print("\n🎯 Risk Calibration Scores:")
            for s in scores:
                print(f"  - {s.get('filepath')}: {s.get('score')}/100 - {s.get('reasoning')}")

        if failures > 0:
            print(f"\n⚠️ Pipeline completed with {failures} failure(s) ({successes} succeeded).")
        else:
            print(f"\n🎉 Pipeline Execution Completed ({successes} file(s) scanned successfully).")

    if failures > 0:
        return 1
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./run.sh <directory_or_file_to_scan>")
        sys.exit(1)
        
    target = sys.argv[1]
    try:
        exit_code = asyncio.run(pipeline(target))
        sys.exit(exit_code)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nProcess aborted by user.")
        sys.exit(130)
