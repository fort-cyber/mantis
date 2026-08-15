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
from core.graph_loader import create_classifier, load_workflow_from_json
from core.sandbox import NullSandbox, SANDBOXES, build_sandbox
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
                    "Review completed.\nVerdict: confirmed",
                    "Exploit successfully reproduced vulnerability.\nVerdict: success",
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
                    "Input is bounded.\nVerdict: false_positive",
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
                    "Analysis done.\nVerdict: confirmed",
                    "Exploit attempt 1 failed.",  # repro 1
                    "Exploit attempt 2 failed.",  # repro 2 (retry)
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
                ["Analysis done.", "Verdict: confirmed", "Exploit verified.\nVerdict: success", "Patch applied", "Score: 90"],
                "dynamic_confirmed"
            ),
            (
                ["Analysis done.", "Verdict: confirmed", "Exploit failed.", "Exploit failed.", "Score: 20"],
                "static_confirmed"
            ),
            (
                ["Analysis done.", "Verdict: false_positive", "Score: 0"],
                "reported"
            ),
        ]

        for replies, expected_status in scenarios:
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
            finally:
                shutil.rmtree(temp_dir)

    async def test_classifier_edge_cases(self):
        """Table-driven unit testing for classifier text normalization, strip chars, and max_visits."""
        test_cases = [
            ("success", ["success", "failed_repro"], 1, "success"),
            ("**Verdict:** `false_positive`", ["confirmed", "false_positive"], 1, "false_positive"),
            ('The verdict is: "confirmed"', ["confirmed", "false_positive"], 1, "confirmed"),
            ("- success", ["success"], 1, "success"),
            ("> confirmed", ["confirmed"], 1, "confirmed"),
            ("unrecognized response", ["confirmed", "false_positive"], 1, DEFAULT_ROUTE),
        ]
        for input_text, routes, max_v, expected_route in test_cases:
            with self.subTest(input_text=input_text, expected_route=expected_route):
                c = create_classifier("test_cls", routes, max_visits=max_v)
                ctx = MagicMock()
                ctx.state = {}
                evt = await c._func(ctx, node_input=input_text)
                self.assertEqual(evt.actions.route, expected_route)
                self.assertEqual(evt.output, input_text)

        # max_visits lifecycle test
        c_multi = create_classifier("repro_cls", ["success"], max_visits=2)
        ctx = MagicMock()
        ctx.state = {}
        evt1 = await c_multi._func(ctx, node_input="failed")
        self.assertEqual(evt1.actions.route, DEFAULT_ROUTE)
        ctx.state["repro_cls_visits"] = 1
        evt2 = await c_multi._func(ctx, node_input="failed")
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

            res_patch = await apply_patch("test_diff")
            self.assertEqual(res_patch, "exit=0\npitched")
            mock_sb.apply_patch.assert_called_once_with("test_diff")
        finally:
            current_run_context.reset(tok)

    async def test_sandbox_seam(self):
        """Tests sandbox dispatch, NullSandbox, custom plugin seam, and gVisor PATH check."""
        with self.assertRaises(ValueError):
            build_sandbox({"type": "invalid_type"})

        null_sb = build_sandbox({"type": "none"})
        self.assertIsInstance(null_sb, NullSandbox)
        self.assertIn("SANDBOX-UNAVAILABLE", await null_sb.execute("whoami"))

        class CustomSeam:
            def __init__(self, target_path: str = "", **_): pass
            async def execute(self, cmd: str) -> str: return "custom_ok"
            async def apply_patch(self, diff: str) -> str: return "custom_patch"
            async def aclose(self): pass

        SANDBOXES["custom"] = CustomSeam
        try:
            custom_sb = build_sandbox({"type": "custom"})
            self.assertEqual(await custom_sb.execute("test"), "custom_ok")
        finally:
            SANDBOXES.pop("custom", None)

        with self.assertRaises(ValueError):
            GvisorSandbox(container_tool="missing_tool_xyz")

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


if __name__ == "__main__":
    unittest.main()
