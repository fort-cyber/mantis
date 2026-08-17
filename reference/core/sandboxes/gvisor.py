import asyncio
import json
import os
import shutil
import subprocess
import uuid

class GvisorSandbox:
    """OCI container sandbox executed under gVisor runtime (docker/podman --runtime=runsc).

    * ISOLATED FILESYSTEM: Target code is copied into a clean OCI container workspace (/workspace).
    * NO HOST FS ACCESS: The container has no mounts to the host filesystem.
    * NETWORKLESS: --network=none guarantees strict network isolation.
    * STATEFUL PATCHING: State persists across execute() and apply_patch() calls within a file run.
    * Requires `docker` or `podman` configured with `runsc` OCI runtime.
    """

    def __init__(
        self,
        target_path: str = "",
        image: str = "mantis-sandbox:latest",
        runtime: str = "runsc",
        timeout_seconds: int = 30,
        container_tool: str | None = None,
    ):
        self.target_path = os.path.realpath(target_path) if target_path else ""
        self.image = image
        self.runtime = runtime
        self.timeout = timeout_seconds

        tool = container_tool
        if not tool:
            for candidate in ("docker", "podman"):
                if shutil.which(candidate) is not None:
                    tool = candidate
                    break

        if not tool or shutil.which(tool) is None:
            raise ValueError(
                f"sandbox type 'gvisor' requires 'docker' or 'podman' with '{runtime}' on PATH. "
                "Install a container engine with gVisor or set sandbox.type to 'static-only'."
            )
        self.tool = tool
        self.container_name = f"mantis-gv-{uuid.uuid4().hex[:12]}"
        self._started = False
        self._lock = asyncio.Lock()

    def _run_cmd(self, argv: list[str], stdin: str | None = None, timeout: int | None = None) -> tuple[int, str]:
        t = timeout or self.timeout
        try:
            p = subprocess.run(
                argv,
                input=stdin or "",
                capture_output=True,
                text=True,
                timeout=t,
            )
            out = (p.stdout or "") + (p.stderr or "")
            return p.returncode, out
        except subprocess.TimeoutExpired:
            return -1, f"SANDBOX-TIMEOUT: command exceeded {t}s."
        except Exception as e:
            return -1, f"SANDBOX-ERROR: {e}"

    async def _ensure(self) -> None:
        async with self._lock:
            if self._started:
                return

            def _init():
                # 1. Create container with runsc runtime and no network
                create_cmd = [
                    self.tool, "create",
                    "--name", self.container_name,
                    f"--runtime={self.runtime}",
                    "--network=none",
                    "-w", "/workspace",
                    self.image,
                    "sleep", "infinity",
                ]
                rc, out = self._run_cmd(create_cmd)
                if rc != 0:
                    raise RuntimeError(f"Failed to create gVisor container '{self.container_name}': {out}")

                # 2. Start container
                rc, out = self._run_cmd([self.tool, "start", self.container_name])
                if rc != 0:
                    self._run_cmd([self.tool, "rm", "-f", self.container_name])
                    raise RuntimeError(f"Failed to start gVisor container '{self.container_name}': {out}")

                # 3. Copy target file if provided
                if self.target_path and os.path.isfile(self.target_path):
                    target_name = os.path.basename(self.target_path)
                    rc, out = self._run_cmd([
                        self.tool, "cp",
                        self.target_path,
                        f"{self.container_name}:/workspace/{target_name}",
                    ])
                    if rc != 0:
                        self._run_cmd([self.tool, "rm", "-f", self.container_name])
                        raise RuntimeError(f"Failed to stage '{self.target_path}' into gVisor container: {out}")

            await asyncio.to_thread(_init)
            self._started = True

    async def execute(self, command: str) -> str:
        await self._ensure()

        def _exec():
            rc, out = self._run_cmd([
                self.tool, "exec",
                self.container_name,
                "/bin/sh", "-c", command,
            ])
            if rc == -1 and "SANDBOX-TIMEOUT" in out:
                return out
            return f"exit={rc}\n{out[:16000]}"

        return await asyncio.to_thread(_exec)

    async def apply_patch(self, diff: str) -> str:
        await self._ensure()

        def _patch():
            rc, out = self._run_cmd(
                [self.tool, "exec", "-i", self.container_name, "/usr/bin/patch", "--batch", "--force", "-p1"],
                stdin=diff,
            )
            if rc == 127:
                return f"exit=127\npatch(1) unavailable in image '{self.image}'"
            if rc != 0:
                rc_retry, out_retry = self._run_cmd(
                    [self.tool, "exec", "-i", self.container_name, "/usr/bin/patch", "--batch", "--force"],
                    stdin=diff,
                )
                if rc_retry == 0:
                    return f"exit=0\n{out_retry[:16000]}"
                return f"exit={rc_retry}\n{out_retry[:16000]}"
            return f"exit={rc}\n{out[:16000]}"

        return await asyncio.to_thread(_patch)

    async def preflight(self) -> None:
        def _check():
            out = subprocess.run(
                [self.tool, "info", "--format", "{{json .Runtimes}}"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                err_msg = (out.stderr or "").strip() or f"{self.tool} daemon not reachable"
                raise RuntimeError(
                    f"Could not connect to {self.tool} daemon ({err_msg}). "
                    f"Ensure {self.tool} is running, or set sandbox.type to 'static-only'."
                )

            try:
                runtimes = json.loads(out.stdout or "{}")
            except json.JSONDecodeError:
                runtimes = {}

            if self.runtime not in runtimes:
                raise RuntimeError(
                    f"'{self.runtime}' is not a registered {self.tool} runtime. "
                    f"Run `sudo runsc install && sudo systemctl restart {self.tool}`, "
                    "or set sandbox.type to 'microsandbox' or 'static-only'."
                )

            img_out = subprocess.run(
                [self.tool, "image", "inspect", self.image],
                capture_output=True, text=True, timeout=10,
            )
            if img_out.returncode != 0:
                raise RuntimeError(
                    f"sandbox image '{self.image}' not found in the local {self.tool} cache. "
                    f"Run ./install.sh to build it, or set sandbox.type to 'static-only'."
                )
        await asyncio.to_thread(_check)

    async def aclose(self) -> None:
        if self._started:
            def _cleanup():
                self._run_cmd([self.tool, "rm", "-f", self.container_name])
            await asyncio.to_thread(_cleanup)
            self._started = False
