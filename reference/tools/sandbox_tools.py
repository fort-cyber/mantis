from core.context import current_run_context

async def run_sandbox(command: str) -> str:
    """Executes a command securely inside the configured sandbox. Use this to compile or run the reproduction script."""
    ctx = current_run_context.get()
    if ctx is None or ctx.sandbox is None:
        return "Sandbox Error: No active sandbox environment."
    try:
        return await ctx.sandbox.execute(command)
    except Exception as e:
        return f"Sandbox Error: {e}"

async def apply_patch(diff_content: str) -> str:
    """Applies a specific code patch to the sandbox context. Code modifications only exist inside the sandbox."""
    ctx = current_run_context.get()
    if ctx is None or ctx.sandbox is None:
        return "Sandbox Error: No active sandbox environment."
    try:
        return await ctx.sandbox.apply_patch(diff_content)
    except Exception as e:
        return f"Sandbox Error: {e}"
