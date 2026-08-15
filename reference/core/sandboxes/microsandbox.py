import os
import sys
from microsandbox import Sandbox as MsbSandbox
from microsandbox.types import Network

MAX_OUTPUT = 16000


class MicrosandboxSandbox:
    """Networkless microVM sandbox (libkrun). STATEFUL for the lifetime of one file:
    a patch applied by apply_patch() IS visible to later execute() calls.

    * NETWORKLESS. Network.none() at creation; no egress, no DNS, no metadata endpoints.
    * FULLY CONFINED. The workspace is copied INTO the VM from an OCI image base.
      Neither reads nor writes reach the host unless the harness explicitly copies
      back, which it does not.
    * NO PACKAGE INSTALLS. With no network the image must already contain the tools
      you invoke.
    * Requires hardware virtualisation: Linux+KVM, macOS Apple Silicon, Windows+WHP.
    """

    def __init__(self, target_path: str = "", image: str = "mantis-sandbox:latest",
                 timeout_seconds: float = 30.0, workdir: str = "/workspace"):
        if sys.platform.startswith("linux") and not os.access("/dev/kvm", os.R_OK | os.W_OK):
            raise RuntimeError(
                "Hardware virtualization unavailable: '/dev/kvm' is not readable/writable. "
                "Ensure KVM is enabled and the user is in the 'kvm' group, or set sandbox.type to 'none'."
            )
        self.target_path = os.path.realpath(target_path) if target_path else ""
        self.image = image
        self.timeout = timeout_seconds
        self.workdir = workdir
        self._sb = None
        self._name = None
        self._stage_error = None

    async def _ensure(self):
        if self._stage_error is not None:
            raise self._stage_error
        if self._sb is None:
            name = f"mantis-{os.getpid()}-{abs(hash(self.target_path)) % 10**8}"
            sb = await MsbSandbox.create(
                name=name,
                image=self.image, network=Network.none(), replace=True,
            )
            await sb.fs.mkdir(self.workdir)

            if self.target_path and os.path.isfile(self.target_path):
                fname = os.path.basename(self.target_path)
                guest_dest = f"{self.workdir}/{fname}"
                try:
                    await sb.fs.copy_from_host(self.target_path, guest_dest)
                except Exception as e:
                    self._stage_error = e
                    self._sb = sb
                    self._name = name
                    raise e

            self._name = name
            self._sb = sb
        return self._sb

    async def _shell(self, command: str) -> str:
        try:
            sb = await self._ensure()
            out = await sb.shell(command, cwd=self.workdir, timeout=self.timeout)
        except Exception as e:
            return f"SANDBOX-ERROR: {type(e).__name__}: {e}"
        text = (out.stdout_text or "") + (getattr(out, "stderr_text", "") or "")
        return f"exit={out.exit_code}\n{text[:MAX_OUTPUT]}"

    async def execute(self, command: str) -> str:
        return await self._shell(command)

    async def apply_patch(self, diff: str) -> str:
        try:
            sb = await self._ensure()
            data = diff.encode("utf-8") if isinstance(diff, str) else diff
            await sb.fs.write(f"{self.workdir}/.mantis.patch", data)
            res = await self._shell("patch -p1 -i .mantis.patch < /dev/null")
            if res.startswith("exit=127"):
                return f"SANDBOX-ERROR: patch(1) unavailable in image '{self.image}'\n{res}"
            if not res.startswith("exit=0"):
                res_no_p1 = await self._shell("patch -i .mantis.patch < /dev/null")
                if res_no_p1.startswith("exit=127"):
                    return f"SANDBOX-ERROR: patch(1) unavailable in image '{self.image}'\n{res_no_p1}"
                return res_no_p1
            return res
        except Exception as e:
            return f"SANDBOX-ERROR: {type(e).__name__}: {e}"

    async def aclose(self) -> None:
        if self._sb is not None:
            try:
                await self._sb.stop()
            except Exception:
                pass
            try:
                if self._name:
                    await MsbSandbox.remove(self._name)
            except Exception:
                pass
            finally:
                self._sb = None
