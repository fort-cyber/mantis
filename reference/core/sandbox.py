import os
from typing import Protocol
from core.sandboxes.gvisor import GvisorSandbox
from core.sandboxes.microsandbox import MicrosandboxSandbox

class Sandbox(Protocol):
    """Executes commands in isolation on behalf of the reproducer and patcher nodes.

    STATE PERSISTENCE IS IMPLEMENTATION-DEFINED. Some sandboxes are one-shot: each
    call gets a fresh environment, so apply_patch() has no effect on any later
    execute(), and the patcher node is informational only. Others hold a session
    for the lifetime of one scanned file. Each implementation states which it is.
    """
    async def execute(self, command: str) -> str: ...
    async def apply_patch(self, diff: str) -> str: ...
    async def preflight(self) -> None: ...
    async def aclose(self) -> None: ...

class StaticOnlySandbox:
    """No-op sandbox for static-only vulnerability pipelines. Dynamic execution is disabled."""

    def __init__(self, target_path: str = "", **_):
        self.target_path = os.path.realpath(target_path) if target_path else ""

    async def execute(self, command: str) -> str:
        return "SANDBOX-UNAVAILABLE: static-only sandbox; no dynamic commands are executed."

    apply_patch = execute

    async def preflight(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

SANDBOXES: dict[str, type] = {
    "static-only": StaticOnlySandbox,
    "gvisor": GvisorSandbox,
    "microsandbox": MicrosandboxSandbox,
}

def build_sandbox(cfg: dict, target_path: str = "") -> Sandbox:
    kind = cfg.get("type", "static-only")
    if kind not in SANDBOXES:
        raise ValueError(
            f"Unknown sandbox type '{kind}'. Available: {sorted(SANDBOXES)}"
        )
    return SANDBOXES[kind](target_path, **cfg.get("options", {}))
