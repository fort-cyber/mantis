import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from pydantic import Field
from google import adk
from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.apps.app import App
from google.adk.workflow import DEFAULT_ROUTE

from core.config import get_llm_kwargs, DEFAULT_MODEL
from core.context import RunContext, current_run_context
from core.database import (
    init_db,
    write_findings,
    read_findings,
    record_calibration,
    read_risk_scores,
    update_status,
)
from core.schemas import VulnerabilityFinding
from core.graph_loader import (
    create_classifier,
    load_workflow_from_json,
    GlobalConfig,
    AgentNode,
)
from core.sandbox import StaticOnlySandbox, SANDBOXES, build_sandbox
from core.sandboxes.gvisor import GvisorSandbox
from main import APP_NAME, USER_ID, execute_sub_task, discover_files, is_binary_file
from pathlib import Path
from tools.research_tools import read_file
from tools.sandbox_tools import run_sandbox, apply_patch


class TestMantisReferenceSuite(unittest.IsolatedAsyncioTestCase):

    async def test_shipped_graph_execution_covers_all_edges(self):
        """Exercises all edges of the shipped workflow using ScriptedLlm across 3 scripts."""
        workflow_path = os.path.join(os.path.dirname(__file__), "workflow.json")

        scripts = [
            # Script 1: Confirmed bug & successful repro & patch -> 7 nodes -> dynamic_confirmed
            (
                [
                    "Found SQL injection in query handler.",
                    json.dumps({"route": "confirmed", "reason": "Review completed."}),
                    json.dumps({"route": "success", "reason": "Exploit successfully reproduced vulnerability."}),
                    "Patch created and applied successfully.",
                    "Calibration score: 90",
                ],
                [
                    "researcher", "reviewer", "reviewer_classifier",
                    "reproducer", "repro_classifier", "patcher", "calibrator"
                ],
                "dynamic_confirmed"
            ),
            # Script 2: False positive -> 3 nodes -> reported (suppressed)
            (
                [
                    "Found potential buffer overflow.",
                    json.dumps({"route": "false_positive", "reason": "Input is bounded."}),
                    "Calibration score: 0",
                ],
                [
                    "researcher", "reviewer", "reviewer_classifier"
                ],
                "reported"
            ),
            # Script 3: Repro fails, retries once, exceeds max_visits -> calibrator -> static_confirmed
            (
                [
                    "Found logic bug.",
                    json.dumps({"route": "confirmed", "reason": "Analysis done."}),
                    json.dumps({"route": "failed_repro", "reason": "Exploit attempt 1 failed."}),  # repro 1
                    json.dumps({"route": "failed_repro", "reason": "Exploit attempt 2 failed."}),  # repro 2 (retry)
                    "Calibration score: 15",
                ],
                [
                    "researcher", "reviewer", "reviewer_classifier",
                    "reproducer", "repro_classifier",
                    "reproducer", "repro_classifier",
                    "calibrator"
                ],
                "static_confirmed"
            ),
        ]

        for script_replies, expected_node_order, expected_status in scripts:
            queue = list(script_replies)

            class ScriptedLlm(BaseLlm):
                async def generate_content_async(self, llm_request, stream: bool = False):
                    text = queue.pop(0) if queue else "done"
                    yield LlmResponse(content=types.Content(parts=[types.Part.from_text(text=text)]))

            with patch.dict(os.environ, {"VERTEXAI_PROJECT": "test-project"}):
                with patch("core.graph_loader.LiteLlm", lambda **_: ScriptedLlm(model="scripted")):
                    wf, cfg = load_workflow_from_json(workflow_path)

            app = App(name=APP_NAME, root_agent=wf)
            ss = InMemorySessionService()
            sess_id = f"sess_{expected_node_order[1]}"
            run_id = f"run_{expected_node_order[1]}"
            target_file = "file.py"

            temp_dir = tempfile.mkdtemp()
            try:
                db_path = os.path.join(temp_dir, "test.db")
                init_db(db_path)
                f = VulnerabilityFinding(
                    title="Flaw", severity="High", description="desc", line_numbers=[1], remediation="rem"
                )
                write_findings(db_path, target_file, [f], run_id=run_id)
                self.assertEqual(read_findings(db_path, target_file, run_id=run_id)[0]["status"], "reported")

                await ss.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=sess_id)
                runner = Runner(app=app, session_service=ss)

                status_map = cfg.get("on_enter_status", {})
                msg = types.Content(parts=[types.Part.from_text(text="Evaluate file.py")], role="user")
                executed_nodes = []
                async for ev in runner.run_async(user_id=USER_ID, session_id=sess_id, new_message=msg):
                    path = getattr(getattr(ev, "node_info", None), "path", None)
                    if path:
                        node_name = path.split("/")[-1].split("@")[0]
                        if status_map and node_name in status_map:
                            update_status(db_path, target_file, run_id, status_map[node_name])
                        if not executed_nodes or executed_nodes[-1] != node_name:
                            executed_nodes.append(node_name)
                await runner.close()
                self.assertEqual(executed_nodes, expected_node_order)

                findings = read_findings(db_path, target_file, run_id=run_id)
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["status"], expected_status)
            finally:
                shutil.rmtree(temp_dir)

    async def test_execute_sub_task_status_lifecycle(self):
        """Verifies execute_sub_task status lifecycle propagation across all paths (dynamic_confirmed, static_confirmed, reported)."""
        workflow_path = os.path.join(os.path.dirname(__file__), "workflow.json")

        scenarios = [
            (
                [
                    "Analysis done.",
                    json.dumps({"route": "confirmed", "reason": "Analysis done."}),
                    json.dumps({"route": "success", "reason": "Exploit verified."}),
                    "Patch applied",
                    "Score: 90",
                ],
                "dynamic_confirmed",
                True,
            ),
            (
                [
                    "Analysis done.",
                    json.dumps({"route": "confirmed", "reason": "Analysis done."}),
                    json.dumps({"route": "failed_repro", "reason": "Exploit failed."}),
                    json.dumps({"route": "failed_repro", "reason": "Exploit failed."}),
                    "Score: 20",
                ],
                "static_confirmed",
                False,
            ),
            (
                [
                    "Analysis done.",
                    json.dumps({"route": "false_positive", "reason": "Score: 0"}),
                    "Score: 0",
                ],
                "reported",
                False,
            ),
        ]

        for replies, expected_status, sb_exec in scenarios:
            with self.subTest(expected_status=expected_status):
                queue = list(replies)

                class ScriptedLlm(BaseLlm):
                    async def generate_content_async(self, llm_request, stream: bool = False):
                        text = queue.pop(0) if queue else "done"
                        yield LlmResponse(content=types.Content(parts=[types.Part.from_text(text=text)]))

                with patch.dict(os.environ, {"VERTEXAI_PROJECT": "test-project"}):
                    with patch("core.graph_loader.LiteLlm", lambda **_: ScriptedLlm(model="scripted")):
                        wf, cfg = load_workflow_from_json(workflow_path)

                app = App(name=APP_NAME, root_agent=wf)
                ss = InMemorySessionService()
                runner = Runner(app=app, session_service=ss)

                temp_dir = tempfile.mkdtemp()
                try:
                    db_path = os.path.join(temp_dir, "test.db")
                    init_db(db_path)
                    target_file = "test_target.py"
                    run_id = f"run-{expected_status}"

                    f = VulnerabilityFinding(
                        title="SQL Injection", severity="Critical", description="raw query", line_numbers=[42], remediation="use ORM"
                    )
                    write_findings(db_path, target_file, [f], run_id=run_id)
                    self.assertEqual(read_findings(db_path, target_file, run_id=run_id)[0]["status"], "reported")

                    ctx = RunContext(jail_dir=temp_dir, db_path=db_path, target_file=target_file, run_id=run_id, sandbox_executed=sb_exec)
                    tok = current_run_context.set(ctx)
                    try:
                        err = await execute_sub_task(
                            runner=runner,
                            session_service=ss,
                            filepath=target_file,
                            run_id=run_id,
                            db_path=db_path,
                            status_map=cfg.get("on_enter_status", {}),
                        )
                        self.assertFalse(err)

                        findings = read_findings(db_path, target_file, run_id=run_id)
                        self.assertEqual(len(findings), 1)
                        self.assertEqual(findings[0]["status"], expected_status)
                    finally:
                        current_run_context.reset(tok)
                finally:
                    await runner.close()
                    shutil.rmtree(temp_dir)

    def test_workflow_loader_validations(self):
        """Validates graph loader error paths and token diagnostics."""
        with patch.dict(os.environ, {"VERTEXAI_PROJECT": "test-project"}):
            temp_dir = tempfile.mkdtemp()
            try:
                prompts_dir = os.path.join(temp_dir, "prompts")
                os.makedirs(prompts_dir, exist_ok=True)
                with open(os.path.join(prompts_dir, "prompt.md"), "w") as f:
                    f.write("Instructions")

                # 1. Unknown node reference in edge
                bad_edge_cfg = {
                    "nodes": [{"id": "a", "type": "agent", "system_prompt": "prompts/prompt.md"}],
                    "edges": [{"from": "START", "to": "a"}, {"from": "a", "to": "non_existent_node"}]
                }
                path = os.path.join(temp_dir, "bad_edge.json")
                with open(path, "w") as f:
                    json.dump(bad_edge_cfg, f)
                with self.assertRaises(ValueError) as ctx:
                    load_workflow_from_json(path)
                self.assertIn("non_existent_node", str(ctx.exception))

                # 2. Duplicate node id
                dup_cfg = {
                    "nodes": [
                        {"id": "node_x", "type": "agent", "system_prompt": "prompts/prompt.md"},
                        {"id": "node_x", "type": "agent", "system_prompt": "prompts/prompt.md"}
                    ],
                    "edges": [{"from": "START", "to": "node_x"}]
                }
                path = os.path.join(temp_dir, "dup.json")
                with open(path, "w") as f:
                    json.dump(dup_cfg, f)
                with self.assertRaises(ValueError) as ctx:
                    load_workflow_from_json(path)
                self.assertIn("node_x", str(ctx.exception))

                # 3. Undeclared route in edge
                bad_route_cfg = {
                    "nodes": [
                        {"id": "cls", "type": "classifier", "routes": ["confirmed", "false_positive"]},
                        {"id": "cal", "type": "agent", "system_prompt": "prompts/prompt.md"}
                    ],
                    "edges": [
                        {"from": "START", "to": "cls"},
                        {"from": "cls", "to": "cal", "on": "unrecognized_route"}
                    ]
                }
                path = os.path.join(temp_dir, "bad_route.json")
                with open(path, "w") as f:
                    json.dump(bad_route_cfg, f)
                with self.assertRaises(ValueError) as ctx:
                    load_workflow_from_json(path)
                self.assertIn("unrecognized_route", str(ctx.exception))

                # 4. Unknown output_schema
                bad_schema_cfg = {
                    "nodes": [
                        {"id": "agent_bad_schema", "type": "agent", "system_prompt": "prompts/prompt.md", "output_schema": "NonExistentSchema"}
                    ],
                    "edges": [{"from": "START", "to": "agent_bad_schema"}]
                }
                path = os.path.join(temp_dir, "bad_schema.json")
                with open(path, "w") as f:
                    json.dump(bad_schema_cfg, f)
                with self.assertRaises(ValueError) as ctx:
                    load_workflow_from_json(path)
                self.assertIn("unknown output_schema 'NonExistentSchema'", str(ctx.exception))
            finally:
                shutil.rmtree(temp_dir)

    async def test_classifier_edge_cases(self):
        """Table-driven unit testing for classifier structured verdict reading and max_visits."""
        test_cases = [
            ({"route": "success", "reason": "verified"}, ["success", "failed_repro"], 1, "success"),
            ({"route": "false_positive", "reason": "benign"}, ["confirmed", "false_positive"], 1, "false_positive"),
            ({"route": "confirmed", "reason": "flaw"}, ["confirmed", "false_positive"], 1, "confirmed"),
            ({"route": "unrecognized", "reason": "unknown"}, ["confirmed", "false_positive"], 1, DEFAULT_ROUTE),
            ({}, ["confirmed", "false_positive"], 1, DEFAULT_ROUTE),
            (None, ["confirmed", "false_positive"], 1, DEFAULT_ROUTE),
        ]
        for verdict, routes, max_v, expected_route in test_cases:
            with self.subTest(verdict=verdict, expected_route=expected_route):
                c = create_classifier("test_cls", routes, max_visits=max_v)
                ctx = MagicMock()
                ctx.state = {"verdict": verdict} if verdict is not None else {}
                evt = await c._func(ctx, node_input="ignored")
                self.assertEqual(evt.actions.route, expected_route)
                self.assertEqual(evt.output, "ignored")

        # Object with .route attribute (e.g. ReviewVerdict or ReproVerdict)
        from core.schemas import ReviewVerdict
        c_obj = create_classifier("test_obj_cls", ["confirmed"])
        ctx_obj = MagicMock()
        ctx_obj.state = {"verdict": ReviewVerdict(route="confirmed", reason="exploit verified")}
        evt_obj = await c_obj._func(ctx_obj)
        self.assertEqual(evt_obj.actions.route, "confirmed")

        # max_visits lifecycle test
        c_multi = create_classifier("repro_cls", ["success"], max_visits=2)
        ctx = MagicMock()
        ctx.state = {"verdict": {"route": "failed_repro", "reason": "attempt 1"}}
        evt1 = await c_multi._func(ctx)
        self.assertEqual(evt1.actions.route, DEFAULT_ROUTE)
        ctx.state["repro_cls_visits"] = 1
        evt2 = await c_multi._func(ctx)
        self.assertEqual(evt2.actions.route, "exceeded")

    def test_database_deduplication_and_normalization(self):
        """Tests SQLite uniqueness deduplication, line sorting normalization, risk recording, and status lifecycle."""
        temp_dir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(temp_dir, "test.db")
            target = os.path.join(temp_dir, "test.py")
            init_db(db_path)

            # 1. Default status is 'reported'
            f1 = VulnerabilityFinding(title="XSS", severity="High", description="d1", line_numbers=[20, 10], remediation="r1")
            write_findings(db_path, target, [f1], run_id="run-1")
            rows = read_findings(db_path, target, run_id="run-1")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "reported")
            self.assertEqual(rows[0]["line_numbers"], [10, 20])

            # Deduplication on (filepath, title, description, line_numbers, run_id)
            f2 = VulnerabilityFinding(title="XSS", severity="Critical", description="d1", line_numbers=[10, 20], remediation="r2")
            write_findings(db_path, target, [f2], run_id="run-1")
            rows = read_findings(db_path, target, run_id="run-1")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["severity"], "Critical")

            # 2. Distinct description creates new row
            f3 = VulnerabilityFinding(title="XSS", severity="Low", description="d2", line_numbers=[10, 20], remediation="r3")
            write_findings(db_path, target, [f3], run_id="run-1")
            self.assertEqual(len(read_findings(db_path, target, run_id="run-1")), 2)

            # 3. Status lifecycle update
            update_status(db_path, target, "run-1", "static_confirmed")
            rows_static = read_findings(db_path, target, run_id="run-1", status="static_confirmed")
            self.assertEqual(len(rows_static), 2)
            self.assertEqual(len(read_findings(db_path, target, run_id="run-1", status="reported")), 0)

            update_status(db_path, target, "run-1", "dynamic_confirmed")
            rows_dynamic = read_findings(db_path, target, run_id="run-1", status="dynamic_confirmed")
            self.assertEqual(len(rows_dynamic), 2)

            # 4. None status in finding payload safely defaults to 'reported'
            f_none = {"title": "CSRF", "severity": "Medium", "description": "no csrf token", "line_numbers": [5], "remediation": "add token", "status": None}
            write_findings(db_path, target, [f_none], run_id="run-2")
            rows_none = read_findings(db_path, target, run_id="run-2")
            self.assertEqual(len(rows_none), 1)
            self.assertEqual(rows_none[0]["status"], "reported")

            # 5. Risk scores
            record_calibration(db_path, target, 85, "High risk flaw", run_id="run-1")
            scores = read_risk_scores(db_path, target, run_id="run-1")
            self.assertEqual(len(scores), 1)
            self.assertEqual(scores[0]["score"], 85)
        finally:
            shutil.rmtree(temp_dir)

    def test_read_file_jail_security(self):
        """Tests directory traversal prevention and boundary enforcement in read_file."""
        temp_dir = tempfile.mkdtemp()
        try:
            jail_dir = os.path.join(temp_dir, "jail")
            os.makedirs(jail_dir)
            inside_file = os.path.join(jail_dir, "inside.txt")
            with open(inside_file, "w") as f:
                f.write("content_inside")

            outside_file = os.path.join(temp_dir, "outside.txt")
            with open(outside_file, "w") as f:
                f.write("secret")

            # 1. No context
            self.assertIn("No active execution context", read_file("inside.txt"))

            ctx = RunContext(jail_dir=jail_dir, db_path="", target_file=inside_file)
            tok = current_run_context.set(ctx)
            try:
                # 2. Inside jail
                self.assertEqual(read_file("inside.txt"), "content_inside")
                # 3. Path traversal escape
                self.assertIn("Permission denied", read_file("../outside.txt"))
                # 4. Absolute path outside jail
                self.assertIn("Permission denied", read_file(outside_file))
                # 5. Non-existent file
                self.assertIn("File not found", read_file("missing.txt"))
            finally:
                current_run_context.reset(tok)
        finally:
            shutil.rmtree(temp_dir)

    async def test_sandbox_tools_with_context(self):
        """Verifies sandbox tool delegators correctly plumb through current_run_context."""
        self.assertIn("No active sandbox", await run_sandbox("echo 1"))
        self.assertIn("No active sandbox", await apply_patch("diff"))

        mock_sb = AsyncMock()
        mock_sb.execute.return_value = "exit=0\nok"
        mock_sb.apply_patch.return_value = "exit=0\npitched"

        ctx = RunContext(jail_dir="/tmp", db_path="", sandbox=mock_sb)
        tok = current_run_context.set(ctx)
        try:
            res_exec = await run_sandbox("echo test")
            self.assertEqual(res_exec, "exit=0\nok")
            mock_sb.execute.assert_called_once_with("echo test")
            self.assertTrue(ctx.sandbox_executed)

            res_patch = await apply_patch("test_diff")
            self.assertEqual(res_patch, "exit=0\npitched")
            mock_sb.apply_patch.assert_called_once_with("test_diff")
        finally:
            current_run_context.reset(tok)

        # Failure string (e.g. SANDBOX-UNAVAILABLE) does NOT set sandbox_executed
        mock_sb_unavail = AsyncMock()
        mock_sb_unavail.execute.return_value = "SANDBOX-UNAVAILABLE: no sandbox configured; nothing was executed."
        ctx_unavail = RunContext(jail_dir="/tmp", db_path="", sandbox=mock_sb_unavail)
        tok = current_run_context.set(ctx_unavail)
        try:
            self.assertFalse(ctx_unavail.sandbox_executed)
            res_fail = await run_sandbox("echo test")
            self.assertIn("SANDBOX-UNAVAILABLE", res_fail)
            self.assertFalse(ctx_unavail.sandbox_executed)
        finally:
            current_run_context.reset(tok)


    async def test_sandbox_seam(self):
        """Tests sandbox dispatch, StaticOnlySandbox, custom plugin seam, and gVisor/Microsandbox platform checks."""
        with self.assertRaises(ValueError):
            build_sandbox({"type": "invalid_type"})

        static_sb = build_sandbox({"type": "static-only"})
        self.assertIsInstance(static_sb, StaticOnlySandbox)
        await static_sb.preflight()
        self.assertIn("SANDBOX-UNAVAILABLE", await static_sb.execute("whoami"))

        class CustomSeam:
            def __init__(self, target_path: str = "", **_): pass
            async def execute(self, cmd: str) -> str: return "custom_ok"
            async def apply_patch(self, diff: str) -> str: return "custom_patch"
            async def preflight(self) -> None: pass
            async def aclose(self): pass

        SANDBOXES["custom"] = CustomSeam
        try:
            custom_sb = build_sandbox({"type": "custom"})
            await custom_sb.preflight()
            self.assertEqual(await custom_sb.execute("test"), "custom_ok")
        finally:
            SANDBOXES.pop("custom", None)

        # 1. When docker/podman is NOT on PATH (e.g. clean macOS or minimal Linux host)
        with patch("shutil.which", return_value=None):
            with self.assertRaises(ValueError) as ctx_missing:
                GvisorSandbox()
            self.assertIn("requires 'docker' or 'podman'", str(ctx_missing.exception))

        # 2. When docker/podman is present on PATH
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}" if x in ("docker", "podman") else None):
            with self.assertRaises(ValueError):
                GvisorSandbox(container_tool="missing_tool_xyz")

            # Test build_sandbox for gvisor
            gv_built = build_sandbox({"type": "gvisor"})
            self.assertIsInstance(gv_built, GvisorSandbox)
            self.assertEqual(gv_built.tool, "docker")

            # 2a. Preflight fails when docker daemon is unreachable
            with patch("subprocess.run") as mock_subproc:
                mock_subproc.return_value = MagicMock(returncode=1, stdout="", stderr="Cannot connect to Docker daemon")
                with self.assertRaises(RuntimeError) as ctx_daemon:
                    await gv_built.preflight()
                self.assertIn("Could not connect to docker daemon", str(ctx_daemon.exception))

            # 2b. Preflight fails when runtime is not registered in docker
            with patch("subprocess.run") as mock_subproc:
                mock_subproc.return_value = MagicMock(returncode=0, stdout='{"runc": {}}', stderr="")
                with self.assertRaises(RuntimeError) as ctx_runsc:
                    await gv_built.preflight()
                self.assertIn("not a registered docker runtime", str(ctx_runsc.exception))

            # 2c. Preflight fails when sandbox image is missing in docker cache
            with patch("subprocess.run") as mock_subproc:
                mock_subproc.side_effect = [
                    MagicMock(returncode=0, stdout='{"runsc": {}}', stderr=""),
                    MagicMock(returncode=1, stdout="", stderr="Error: No such image"),
                ]
                with self.assertRaises(RuntimeError) as ctx_img:
                    await gv_built.preflight()
                self.assertIn("sandbox image 'mantis-sandbox:latest' not found in the local docker cache", str(ctx_img.exception))

            # 2d. Preflight succeeds when runtime and image are present
            with patch("subprocess.run") as mock_subproc:
                mock_subproc.side_effect = [
                    MagicMock(returncode=0, stdout='{"runsc": {}}', stderr=""),
                    MagicMock(returncode=0, stdout="[ok]", stderr=""),
                ]
                await gv_built.preflight()

            # Mocked containerized gVisor execution test
            gv = GvisorSandbox(container_tool="docker")
            gv._run_cmd = MagicMock()
            # 1. create -> 0, start -> 0
            gv._run_cmd.side_effect = [
                (0, "container_id"),
                (0, ""),
                (0, "hello gvisor\n"),
                (0, "patch applied\n"),
                (0, ""),
            ]
            out_exec = await gv.execute("echo hello")
            self.assertIn("hello gvisor", out_exec)
            out_patch = await gv.apply_patch("diff_text")
            self.assertIn("patch applied", out_patch)
            await gv.aclose()

        # 3. Microsandbox KVM access check on Linux
        from core.sandboxes.microsandbox import MicrosandboxSandbox
        from microsandbox import ImageNotFoundError
        with patch("sys.platform", "linux"):
            with patch("os.access", return_value=False):
                with self.assertRaises(RuntimeError) as ctx_kvm:
                    MicrosandboxSandbox()
                self.assertIn("Hardware virtualization unavailable", str(ctx_kvm.exception))
                self.assertIn("static-only", str(ctx_kvm.exception))

            with patch("os.access", return_value=True):
                msb = MicrosandboxSandbox()
                self.assertEqual(msb.image, "mantis-sandbox:latest")

            # 4. Microsandbox missing image failure in preflight
            with patch("os.access", return_value=True):
                sb_missing = MicrosandboxSandbox(image="missing-image-not-in-cache:latest")
                with patch("microsandbox.Image.get", side_effect=ImageNotFoundError("image not found")):
                    with self.assertRaises(RuntimeError) as ctx_img:
                        await sb_missing.preflight()
                    self.assertIn("sandbox image 'missing-image-not-in-cache:latest' not found in the local cache", str(ctx_img.exception))

                # Preflight success when Image.get succeeds
                with patch("microsandbox.Image.get", AsyncMock()):
                    await sb_missing.preflight()

                # 5. Verify MsbSandbox.create passes pull_policy=PullPolicy.NEVER
                from microsandbox import PullPolicy
                mock_msb_instance = AsyncMock()
                mock_msb_instance.fs.mkdir = AsyncMock()
                mock_msb_instance.fs.copy_from_host = AsyncMock()
                with patch("microsandbox.Sandbox.create", AsyncMock(return_value=mock_msb_instance)) as mock_msb_create:
                    sb_created = MicrosandboxSandbox(image="mantis-sandbox:latest")
                    await sb_created._ensure()
                    mock_msb_create.assert_called_once()
                    _, kwargs_create = mock_msb_create.call_args
                    self.assertEqual(kwargs_create.get("pull_policy"), PullPolicy.NEVER)
                    self.assertEqual(kwargs_create.get("image"), "mantis-sandbox:latest")


    def test_get_llm_kwargs_resolution_and_precedence(self):
        """Tests LLM resolution precedence for model_id and api_base across all tiers."""
        # 1. Defaults
        with patch.dict(os.environ, {"VERTEXAI_PROJECT": "proj-1", "VERTEXAI_LOCATION": "loc-1"}, clear=True):
            mid, kwargs = get_llm_kwargs()
            self.assertEqual(mid, DEFAULT_MODEL)
            self.assertEqual(kwargs["model"], DEFAULT_MODEL)
            self.assertEqual(kwargs["vertex_project"], "proj-1")
            self.assertEqual(kwargs["vertex_location"], "loc-1")
            self.assertNotIn("api_base", kwargs)

        # 2. MODEL_ID environment variable
        with patch.dict(os.environ, {"MODEL_ID": "openai/gpt-4o", "VERTEXAI_PROJECT": "proj-1"}, clear=True):
            mid, kwargs = get_llm_kwargs()
            self.assertEqual(mid, "openai/gpt-4o")
            self.assertEqual(kwargs["model"], "openai/gpt-4o")
            self.assertNotIn("api_base", kwargs)

        # 3. Explicit node model_id overrides MODEL_ID env and default
        with patch.dict(os.environ, {"MODEL_ID": "openai/gpt-4o"}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3", default_model="fallback-model")
            self.assertEqual(mid, "ollama/llama3")
            self.assertEqual(kwargs["model"], "ollama/llama3")

        # 4. LLM_API_BASE environment variable
        with patch.dict(os.environ, {"LLM_API_BASE": "http://env-api-base:8000"}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3")
            self.assertEqual(kwargs["api_base"], "http://env-api-base:8000")

        # 5. default_api_base is used when LLM_API_BASE is unset
        with patch.dict(os.environ, {}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3", default_api_base="http://config-api-base:7000")
            self.assertEqual(kwargs["api_base"], "http://config-api-base:7000")

        # 6. LLM_API_BASE env overrides default_api_base (config)
        with patch.dict(os.environ, {"LLM_API_BASE": "http://env-api-base:8000"}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3", default_api_base="http://config-api-base:7000")
            self.assertEqual(kwargs["api_base"], "http://env-api-base:8000")

        # 7. Explicit api_base parameter (node override) overrides LLM_API_BASE env AND default_api_base
        with patch.dict(os.environ, {"LLM_API_BASE": "http://env-api-base:8000"}, clear=True):
            mid, kwargs = get_llm_kwargs(
                model_id="ollama/llama3",
                api_base="http://param-api-base:9000",
                default_api_base="http://config-api-base:7000"
            )
            self.assertEqual(kwargs["api_base"], "http://param-api-base:9000")

        # 8. LLM_TIMEOUT environment variable
        with patch.dict(os.environ, {"LLM_TIMEOUT": "300"}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3")
            self.assertEqual(kwargs["timeout"], 300.0)

        # 9. LLM_REQUEST_TIMEOUT environment variable fallback
        with patch.dict(os.environ, {"LLM_REQUEST_TIMEOUT": "450.5"}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3")
            self.assertEqual(kwargs["timeout"], 450.5)

        # 10. default_timeout is used when LLM_TIMEOUT is unset
        with patch.dict(os.environ, {}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3", default_timeout=600.0)
            self.assertEqual(kwargs["timeout"], 600.0)

        # 11. LLM_TIMEOUT env overrides default_timeout
        with patch.dict(os.environ, {"LLM_TIMEOUT": "180"}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3", default_timeout=600.0)
            self.assertEqual(kwargs["timeout"], 180.0)

        # 12. Explicit timeout parameter (node override) overrides LLM_TIMEOUT env AND default_timeout
        with patch.dict(os.environ, {"LLM_TIMEOUT": "180"}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3", timeout=90.0, default_timeout=600.0)
            self.assertEqual(kwargs["timeout"], 90.0)

        # 13. Non-vertex model does not require VERTEXAI_PROJECT
        with patch.dict(os.environ, {}, clear=True):
            mid, kwargs = get_llm_kwargs(model_id="ollama/llama3")
            self.assertEqual(mid, "ollama/llama3")
            self.assertNotIn("vertex_project", kwargs)

    def test_per_node_model_and_api_base_in_workflow(self):
        """Validates that GlobalConfig parses api_base and AgentNode supports per-node model and api_base overrides."""
        temp_dir = tempfile.mkdtemp()
        try:
            prompts_dir = os.path.join(temp_dir, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)
            prompt_file = os.path.join(prompts_dir, "researcher.md")
            with open(prompt_file, "w") as f:
                f.write("Evaluate input")

            workflow_def = {
                "name": "custom_workflow",
                "config": {
                    "api_base": "http://custom-proxy.internal:8080",
                    "default_model": "vertex_ai/gemini-3.6-flash",
                    "timeout": 600.0
                },
                "nodes": [
                    {
                        "id": "agent_override",
                        "type": "agent",
                        "model": "ollama/deepseek-r1",
                        "api_base": "http://node-custom.internal:11434",
                        "timeout": 120.0,
                        "system_prompt": "prompts/researcher.md"
                    },
                    {
                        "id": "agent_default",
                        "type": "agent",
                        "system_prompt": "prompts/researcher.md"
                    },
                    {
                        "id": "agent_model_only",
                        "type": "agent",
                        "model": "openai/gpt-4o",
                        "system_prompt": "prompts/researcher.md"
                    }
                ],
                "edges": [
                    {"from": "START", "to": "agent_override"},
                    {"from": "agent_override", "to": "agent_default"},
                    {"from": "agent_default", "to": "agent_model_only"}
                ]
            }
            wf_path = os.path.join(temp_dir, "workflow.json")
            with open(wf_path, "w") as f:
                json.dump(workflow_def, f)

            captured_agent_calls = []

            def fake_agent(name, model, instruction, tools, *args, **kwargs):
                captured_agent_calls.append({"name": name, "model": model})
                mock_a = MagicMock()
                mock_a.name = name
                return mock_a

            with patch.dict(os.environ, {"VERTEXAI_PROJECT": "test-project"}):
                with patch("google.adk.Agent", side_effect=fake_agent):
                    wf, cfg = load_workflow_from_json(wf_path)

            self.assertEqual(cfg.get("api_base"), "http://custom-proxy.internal:8080")
            self.assertEqual(len(captured_agent_calls), 3)

            # Node 1: per-node model, api_base, and timeout overrides
            self.assertEqual(captured_agent_calls[0]["name"], "agent_override")
            self.assertEqual(captured_agent_calls[0]["model"].model, "ollama/deepseek-r1")
            self.assertEqual(captured_agent_calls[0]["model"]._additional_args["api_base"], "http://node-custom.internal:11434")
            self.assertEqual(captured_agent_calls[0]["model"]._additional_args["timeout"], 120.0)

            # Node 2: inherits global default_model, global api_base, and global timeout
            self.assertEqual(captured_agent_calls[1]["name"], "agent_default")
            self.assertEqual(captured_agent_calls[1]["model"].model, "vertex_ai/gemini-3.6-flash")
            self.assertEqual(captured_agent_calls[1]["model"]._additional_args["api_base"], "http://custom-proxy.internal:8080")
            self.assertEqual(captured_agent_calls[1]["model"]._additional_args["timeout"], 600.0)

            # Node 3: per-node model override, inherits global api_base and global timeout
            self.assertEqual(captured_agent_calls[2]["name"], "agent_model_only")
            self.assertEqual(captured_agent_calls[2]["model"].model, "openai/gpt-4o")
            self.assertEqual(captured_agent_calls[2]["model"]._additional_args["api_base"], "http://custom-proxy.internal:8080")
            self.assertEqual(captured_agent_calls[2]["model"]._additional_args["timeout"], 600.0)
        finally:
            shutil.rmtree(temp_dir)

    def test_global_config_forbids_unknown_fields(self):
        """Ensures GlobalConfig extra='forbid' still rejects unrecognized fields."""
        with self.assertRaises(Exception):
            GlobalConfig(unknown_field="invalid")

    def test_discover_files(self):
        """Verifies discover_files handles single files, git repos, hidden directories, db_path exclusion, and binary filtering."""
        temp_dir = tempfile.mkdtemp()
        try:
            p_dir = Path(temp_dir)
            f1 = p_dir / "app.py"
            f1.write_text("print(1)")
            
            # 1. Single file target
            self.assertEqual(discover_files(f1), [str(f1)])

            # 2. Unicode text file with non-ASCII characters (Japanese, Chinese, Emoji, Accents)
            f_unicode = p_dir / "unicode_app.py"
            f_unicode.write_text("# 日本語テスト 🚀 \n# 漏洞分析 \nprint('crème brûlée')", encoding="utf-8")
            self.assertFalse(is_binary_file(f_unicode))

            # 3. Binary file containing null bytes (.pyc, compiled object, image)
            f_binary = p_dir / "compiled.pyc"
            f_binary.write_bytes(b"\x61\x0d\x0d\x0a\x00\x00\x00\x00\x7fELF\x02\x01\x01\x00")
            self.assertTrue(is_binary_file(f_binary))

            # 4. Directory with hidden files and subdirectories
            hidden_dir = p_dir / ".venv" / "lib"
            hidden_dir.mkdir(parents=True)
            (hidden_dir / "secret.py").write_text("hidden")

            f2 = p_dir / "utils.py"
            f2.write_text("def helper(): pass")

            db_file = p_dir / "findings.db"
            db_file.write_bytes(b"SQLite format 3\x00")

            discovered = discover_files(p_dir, db_path=str(db_file))
            self.assertEqual(discovered, [str(f1), str(f_unicode), str(f2)])
            self.assertNotIn(str(f_binary), discovered)
            self.assertNotIn(str(hidden_dir / "secret.py"), discovered)
            self.assertNotIn(str(db_file), discovered)
        finally:
            shutil.rmtree(temp_dir)

    async def test_schemas_and_dynamic_confirmed_gating(self):
        """Tests ReviewVerdict and ReproVerdict schema validation and dynamic_confirmed gating in execute_sub_task."""
        from core.schemas import ReviewVerdict, ReproVerdict
        rv = ReviewVerdict(route="confirmed", reason="Exploitable vulnerability found.")
        self.assertEqual(rv.route, "confirmed")
        with self.assertRaises(Exception):
            ReviewVerdict(route="invalid_route", reason="bad")

        rp = ReproVerdict(route="success", reason="Exploit passed.")
        self.assertEqual(rp.route, "success")
        with self.assertRaises(Exception):
            ReproVerdict(route="invalid_route", reason="bad")

        # Test execute_sub_task dynamic_confirmed gating when sandbox_executed is False vs True
        workflow_path = os.path.join(os.path.dirname(__file__), "workflow.json")
        queue = [
            "Analysis done.",
            json.dumps({"route": "confirmed", "reason": "Analysis done."}),
            json.dumps({"route": "success", "reason": "Exploit verified."}),
            "Patch applied",
            "Score: 90",
        ]

        class ScriptedLlm(BaseLlm):
            async def generate_content_async(self, llm_request, stream: bool = False):
                text = queue.pop(0) if queue else "done"
                yield LlmResponse(content=types.Content(parts=[types.Part.from_text(text=text)]))

        with patch.dict(os.environ, {"VERTEXAI_PROJECT": "test-project"}):
            with patch("core.graph_loader.LiteLlm", lambda **_: ScriptedLlm(model="scripted")):
                wf, cfg = load_workflow_from_json(workflow_path)

        app = App(name=APP_NAME, root_agent=wf)
        ss = InMemorySessionService()
        runner = Runner(app=app, session_service=ss)

        temp_dir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(temp_dir, "test.db")
            init_db(db_path)
            target_file = "test_target.py"
            run_id = "run-gated"

            f = VulnerabilityFinding(
                title="SQL Injection", severity="Critical", description="raw query", line_numbers=[42], remediation="use ORM"
            )
            write_findings(db_path, target_file, [f], run_id=run_id)

            # 1. When sandbox_executed is False in RunContext, finding status is kept at static_confirmed
            ctx_gated = RunContext(jail_dir=temp_dir, db_path=db_path, target_file=target_file, run_id=run_id, sandbox_executed=False)
            tok = current_run_context.set(ctx_gated)
            try:
                err = await execute_sub_task(
                    runner=runner,
                    session_service=ss,
                    filepath=target_file,
                    run_id=run_id,
                    db_path=db_path,
                    status_map=cfg.get("on_enter_status", {}),
                )
                self.assertFalse(err)
                findings = read_findings(db_path, target_file, run_id=run_id)
                self.assertEqual(findings[0]["status"], "static_confirmed")
            finally:
                current_run_context.reset(tok)

            # 2. When sandbox_executed is True in RunContext, finding status is elevated to dynamic_confirmed
            queue = [
                "Analysis done.",
                json.dumps({"route": "confirmed", "reason": "Analysis done."}),
                json.dumps({"route": "success", "reason": "Exploit verified."}),
                "Patch applied",
                "Score: 90",
            ]
            run_id_dyn = "run-gated-dyn"
            write_findings(db_path, target_file, [f], run_id=run_id_dyn)
            ctx_dyn = RunContext(jail_dir=temp_dir, db_path=db_path, target_file=target_file, run_id=run_id_dyn, sandbox_executed=True)
            tok = current_run_context.set(ctx_dyn)
            try:
                err = await execute_sub_task(
                    runner=runner,
                    session_service=ss,
                    filepath=target_file,
                    run_id=run_id_dyn,
                    db_path=db_path,
                    status_map=cfg.get("on_enter_status", {}),
                )
                self.assertFalse(err)
                findings = read_findings(db_path, target_file, run_id=run_id_dyn)
                self.assertEqual(findings[0]["status"], "dynamic_confirmed")
            finally:
                current_run_context.reset(tok)

            # 3. When current_run_context is None (no context), finding status is NOT elevated to dynamic_confirmed
            queue = [
                "Analysis done.",
                json.dumps({"route": "confirmed", "reason": "Analysis done."}),
                json.dumps({"route": "success", "reason": "Exploit verified."}),
                "Patch applied",
                "Score: 90",
            ]
            run_id_noctx = "run-gated-noctx"
            write_findings(db_path, target_file, [f], run_id=run_id_noctx)
            err = await execute_sub_task(
                runner=runner,
                session_service=ss,
                filepath=target_file,
                run_id=run_id_noctx,
                db_path=db_path,
                status_map=cfg.get("on_enter_status", {}),
            )
            self.assertFalse(err)
            findings = read_findings(db_path, target_file, run_id=run_id_noctx)
            self.assertEqual(findings[0]["status"], "static_confirmed")
        finally:
            await runner.close()
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()

